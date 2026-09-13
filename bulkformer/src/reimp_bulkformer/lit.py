"""LightningModule: fitted graph, masking, loss and optimization for BulkFormer.

Batches arrive as log1p TPM — the shared DataModule with
`quantification: tpm_unstranded` and `transform: log1p`. Before the first
step `setup` fits the model's statistics on the fold's training samples
only (`shared/EVALS.md`, rule 3): the gene co-expression graph, and each
gene's training mean, the validation baseline. Both are buffers, so a
checkpoint carries them and embedding refits nothing.

Training masks come from the global RNG (seeded by `seed_everything`).
Validation masks come from a generator reset to `seed` at the start of each
loop, so validation metrics compare like with like across epochs.
Embeddings read the unmasked sample.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Literal

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from reimp_bulkformer.favor import FavorAttention
from reimp_bulkformer.graph import (
    DEFAULT_K,
    DEFAULT_THRESHOLD,
    coexpression_graph,
    shuffle_genes,
)
from reimp_bulkformer.model import BulkFormer, Pooling, mask_genes, masked_mse
from reimp_shared.data import ExpressionData

Graph = Literal["coexpression", "random", "none"]
GraphSpace = Literal["linear", "log1p"]
GRAPHS: tuple[str, ...] = ("coexpression", "random", "none")
GRAPH_SPACES: tuple[str, ...] = ("linear", "log1p")


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """lr multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def pearson(a: Tensor, b: Tensor) -> Tensor:
    """Row-wise Pearson correlation of two (B, K) tensors; 0 for a constant row."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


class LitBulkFormer(L.LightningModule):
    """BulkFormer at reimp's default size (`paper.md`, "For reimp").

    Defaults are d = 256, N = 1 block of one GCN and K = 4 Performer layers
    (8 heads, FFN ×4), the sample-context MLP's hidden width cut to d, 15%
    of genes masked, and AdamW at peak lr 1e-4 after a 5% linear warmup
    (the paper's), then cosine decay (ours).

    `graph` picks the GCN's graph: `coexpression` (top `graph_k` |r| ≥
    `graph_threshold` over the fold's training samples, correlated on
    `graph_space` values — linear TPM by default), `random` (the same graph
    on shuffled genes, a degree-matched control) or `none` (Performer only).
    `pooling` is how `predict_step` pools the final gene tokens.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_blocks: int = 1,
        n_layers: int = 4,
        n_heads: int = 8,
        n_features: int | None = None,
        mlp_ratio: float = 4.0,
        sample_hidden: int | None = None,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        graph: Graph = "coexpression",
        graph_k: int = DEFAULT_K,
        graph_threshold: float = DEFAULT_THRESHOLD,
        graph_space: GraphSpace = "linear",
        feature_redraw_interval: int | None = 1000,
        activation_checkpointing: bool = False,
        pooling: Pooling = "max",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_frac: float = 0.05,
        seed: int = 0,
    ) -> None:
        """`n_features` is FAVOR+'s random features per head, d_head · ln d_head when None.

        `feature_redraw_interval` redraws them every that many optimizer
        steps in training (performer-pytorch's default, 1000); None never
        does. `seed` seeds validation masks and the random graph.
        """
        super().__init__()
        self.save_hyperparameters()
        if graph not in GRAPHS:
            raise ValueError(f"unknown graph {graph!r}; expected one of {GRAPHS}")
        if graph_space not in GRAPH_SPACES:
            raise ValueError(f"unknown graph_space {graph_space!r}; expected one of {GRAPH_SPACES}")
        self.model = BulkFormer(
            n_genes=n_genes,
            d_model=d_model,
            n_blocks=n_blocks,
            n_layers=n_layers,
            n_heads=n_heads,
            n_features=n_features,
            mlp_ratio=mlp_ratio,
            sample_hidden=sample_hidden,
            dropout=dropout,
            use_graph=graph != "none",
            activation_checkpointing=activation_checkpointing,
        )
        # Fitted on the training samples by `fit_statistics`.
        self.register_buffer("gene_mean", torch.zeros(n_genes))
        self.register_buffer("fitted", torch.tensor(False))
        self._generator = torch.Generator().manual_seed(seed)
        self._redraws = 0

    # ---------- fitted statistics ----------

    def fit_statistics(self, data: ExpressionData) -> None:
        """Fit the gene graph and gene means on `data`'s training rows, and nothing else.

        `data.values` is log1p TPM; the graph correlates `expm1` of it (linear
        TPM) or the values as they are, per `graph_space`.
        """
        hp = self.hparams
        train = data.values[data.rows("train")]
        if train.shape[1] != hp.n_genes:
            raise ValueError(f"data has {train.shape[1]} genes, the model {hp.n_genes}")
        mean = torch.from_numpy(train.mean(axis=0, dtype=np.float64))
        self.gene_mean.copy_(mean.to(self.gene_mean))
        # Predictions start at the training samples' average log1p TPM
        # rather than at the model's arbitrary initial offset.
        with torch.no_grad():
            self.model.head[1][-1].bias.fill_(float(mean.mean()))
        if hp.graph != "none":
            source = np.expm1(train) if hp.graph_space == "linear" else train
            index, weight = coexpression_graph(source, hp.graph_k, hp.graph_threshold)
            if hp.graph == "random":
                index = shuffle_genes(index, hp.n_genes, torch.Generator().manual_seed(hp.seed))
            self.model.set_graph(index, weight)
        self.fitted.fill_(True)

    def setup(self, stage: str) -> None:
        if self.fitted:
            return
        datamodule = self.trainer.datamodule
        data = getattr(datamodule, "data", None)
        if data is None:
            raise RuntimeError(
                "BulkFormer fits its gene graph on the DataModule's training samples: "
                "train with an ExpressionDataModule, or call fit_statistics first"
            )
        if datamodule.hparams.transform != "log1p":
            raise ValueError("BulkFormer reads log1p TPM: set the DataModule's transform to log1p")
        self.fit_statistics(data)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # The edge count is known only once a graph is fit: size the graph
        # buffers before Lightning loads the state dict into them.
        state = checkpoint["state_dict"]
        index = state.get("model.graph_index")
        if index is not None and index.numel():
            self.model.set_graph(index, state["model.graph_weight"])

    # ---------- steps ----------

    def _mask(self, values: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if not values.is_floating_point():
            raise TypeError("BulkFormer expects log1p TPM: use the `log1p` transform")
        n_rows, n_genes = values.shape
        return mask_genes(n_rows, n_genes, self.hparams.mask_ratio, values.device, generator)

    def on_train_batch_start(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        interval = self.hparams.feature_redraw_interval
        if interval and self.global_step // interval > self._redraws:
            self._redraws = self.global_step // interval
            for module in self.modules():
                if isinstance(module, FavorAttention):
                    module.redraw_features()

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        values = batch["values"]
        mask = self._mask(values)
        loss = masked_mse(self.model(values, mask), values, mask)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(values))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Masked MSE beside the training gene means' MSE, and per-sample Pearson.

        Every row masks the same number of genes, so the masked entries
        reshape to (B, K).
        """
        values = batch["values"]
        mask = self._mask(values, self._generator)
        n = len(values)
        target = values[mask].view(n, -1)
        pred = self.model(values, mask)[mask].view(n, -1)
        baseline = self.gene_mean.expand_as(values)[mask].view(n, -1)
        self.log("val/loss", F.mse_loss(pred, target), prog_bar=True, batch_size=n)
        self.log("val/loss_gene_mean", F.mse_loss(baseline, target), batch_size=n)
        self.log("val/pearson", pearson(pred, target).mean(), batch_size=n)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Final gene tokens of the unmasked sample, pooled per `pooling`. No head."""
        embedding = self.model.embed(batch["values"], self.hparams.pooling)
        return {"sample_index": batch["sample_index"], "embedding": embedding}

    def configure_optimizers(self):
        hp = self.hparams
        total_steps = self.trainer.estimated_stepping_batches
        if math.isinf(total_steps):
            raise ValueError("the lr schedule needs a finite run: set max_epochs or max_steps")
        if total_steps < 1:
            raise ValueError("no training steps: is batch_size larger than the training split?")
        total_steps = int(total_steps)
        warmup = max(1, round(hp.warmup_frac * total_steps))
        # Decay weight matrices and embeddings; leave biases and norms alone.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": hp.weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=hp.lr)
        schedule = partial(warmup_cosine, warmup=warmup, total=total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
