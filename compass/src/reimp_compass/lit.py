"""LightningModule: triplets, loss and optimization for COMPASS pretraining.

Batches arrive as log TPM — `tpm_unstranded` with the shared DataModule's
`log1p` transform. COMPASS takes log2(TPM + 1), but the per-gene min-max
scaling that follows divides out the base. At `setup("fit")` the module
binds the hierarchy's genes to the DataModule's columns by symbol, fits
the min-max scaler on the fold's training samples, and keeps a reference
to the loaded matrix to draw negatives from.

Each step scales the batch, stacks it twice (anchor, positive) over the
negatives drawn for it, augments all three views independently, and runs
one forward pass. Training draws come from the global RNG (seeded by
`seed_everything`). Validation draws — augmentations and negatives, from
validation patients — come from a generator reset to `seed` at the start
of each loop, so the early-stopping loss compares like with like across
epochs. Predictions are unaugmented and deterministic.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from torch import Tensor

from reimp_compass.hierarchy import load_hierarchy
from reimp_compass.model import Compass, cosine_distance, triplet_loss
from reimp_compass.triplets import Negatives, NegativeSampler, augment
from reimp_shared.data import ExpressionData


class LitCompass(L.LightningModule):
    """COMPASS pretraining with the released model's settings, less the cancer token.

    Defaults: one layer, d = 32, 2 heads of width 32, FFN 64, dropout 0.2;
    triplet margin 1; mask p = 0.1 (the paper's) or jitter σ = 0.1; Adam at
    lr 1e-3 with weight decay 1e-4.
    """

    def __init__(
        self,
        n_genes: int,
        hierarchy_path: str | None = None,
        d_model: int = 32,
        n_heads: int = 2,
        head_dim: int = 32,
        dim_ff: int = 64,
        n_layers: int = 1,
        dropout: float = 0.2,
        attention_chunk_size: int | None = None,
        margin: float = 1.0,
        mask_prob: float = 0.1,
        jitter_std: float = 0.1,
        no_augment_prob: float = 0.1,
        negatives: Negatives = "any",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        """`hierarchy_path=None` reads COMPASS's packaged gene-set table.

        `attention_chunk_size` attends that many query genes at a time (see
        `EncoderLayer`); leave it unset where SDPA has a fused kernel.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hierarchy = load_hierarchy(hierarchy_path)
        self.model = Compass(
            n_genes,
            self.hierarchy,
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            dim_ff=dim_ff,
            n_layers=n_layers,
            dropout=dropout,
            attention_chunk_size=attention_chunk_size,
        )
        self._generator = torch.Generator().manual_seed(seed)
        self._values: Tensor | None = None
        self._row_of: Tensor | None = None
        self._samplers: dict[str, NegativeSampler] = {}

    @property
    def concept_names(self) -> tuple[str, ...]:
        return self.hierarchy.concepts

    @property
    def set_names(self) -> tuple[str, ...]:
        return self.hierarchy.sets

    # ---------- data ----------

    def prepare(self, data: ExpressionData) -> None:
        """Bind the concept genes to `data`'s columns and fit the scaler on its training rows."""
        if data.values.shape[1] != self.hparams.n_genes:
            raise ValueError(f"model has {self.hparams.n_genes} genes, data {data.values.shape[1]}")
        names = data.genes["gene_name"].tolist()
        missing = self.hierarchy.missing(names)
        n_genes = len(self.hierarchy.genes)
        rank_zero_info(f"COMPASS: {n_genes - len(missing)} of {n_genes} concept genes selected")
        if missing:
            rank_zero_warn(f"{len(missing)} concept genes are not selected, e.g. {missing[:5]}")
        self.model.projector.bind(self.hierarchy.member_positions(names))
        self.model.scaler.fit(data.values[data.rows("train")])
        self.attach(data)

    def attach(self, data: ExpressionData) -> None:
        """Keep `data` to draw negatives from: training and validation patients apart."""
        self._values = torch.from_numpy(data.values)
        sample_index = data.samples["sample_index"].to_numpy()
        row_of = np.full(sample_index.max() + 1, -1, dtype=np.int64)
        row_of[sample_index] = np.arange(len(sample_index))
        self._row_of = torch.from_numpy(row_of)
        cases, projects = data.samples["case_submitter_id"], data.samples["project_id"]
        self._samplers = {
            split: NegativeSampler(cases, projects, data.rows(split), self.hparams.negatives)
            for split in ("train", "val")
            if len(np.unique(cases.to_numpy()[data.rows(split)])) >= 2
        }

    def setup(self, stage: str) -> None:
        datamodule = self.trainer.datamodule
        if stage == "fit":
            self.prepare(datamodule.data)
        elif stage == "validate":
            self.attach(datamodule.data)

    # ---------- steps ----------

    def _triplets(
        self, batch: dict[str, Tensor], split: str, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Concept vectors of the anchor, positive and negative views of a batch."""
        values = batch["values"]
        if not values.is_floating_point():
            raise TypeError("COMPASS expects log TPM: tpm_unstranded with the log1p transform")
        if split not in self._samplers:
            raise RuntimeError(f"no {split} negatives: call prepare (or fit) first")
        rows = self._row_of[batch["sample_index"].cpu()]
        negatives = self._samplers[split].draw(rows, generator)
        negatives = self._values[negatives].to(values.device, non_blocking=True)
        x = self.model.scaler(torch.cat([values, values, negatives]))
        hp = self.hparams
        views = augment(x, hp.mask_prob, hp.jitter_std, hp.no_augment_prob, generator)
        _, concepts = self.model(views)
        return concepts.chunk(3)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        anchor, positive, negative = self._triplets(batch, "train")
        loss = triplet_loss(anchor, positive, negative, self.hparams.margin).mean()
        self.log("train/loss", loss, prog_bar=True, batch_size=len(anchor))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        anchor, positive, negative = self._triplets(batch, "val", self._generator)
        losses = triplet_loss(anchor, positive, negative, self.hparams.margin)
        n = len(anchor)
        self.log("val/loss", losses.mean(), prog_bar=True, batch_size=n)
        self.log("val/d_pos", cosine_distance(anchor, positive).mean(), batch_size=n)
        self.log("val/d_neg", cosine_distance(anchor, negative).mean(), batch_size=n)
        self.log("val/active", (losses > 0).float().mean(), batch_size=n)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Set and concept scores of the unaugmented samples."""
        sets, concepts = self.model(self.model.scaler(batch["values"]))
        return {"sample_index": batch["sample_index"], "concepts": concepts, "sets": sets}

    def configure_optimizers(self):
        # torch's Adam with L2 weight decay on every parameter, as COMPASS trains.
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
