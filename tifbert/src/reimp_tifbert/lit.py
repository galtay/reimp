"""LightningModule: ranking, windows, masking and optimization for TifBERT.

Batches arrive from the shared DataModule as expression values
(`tpm_unstranded`, as stored, in the configs). Before a batch leaves the
CPU, `on_before_batch_transfer` ranks each sample's genes with the model's
`GeneRanker`, so only gene tokens reach the accelerator. The ranker, and
the mask of genes expressed in training that says which genes a sentence
may hold, are fit in `setup` on the DataModule's training rows — the
fold's training samples, never its val or test ones — and travel in the
checkpoint, so embedding reuses the fold's own.

Training draws one window per sample per step, uniformly over the sample's
windows, and masks it, from the global RNG (seeded by `seed_everything`).
Validation draws windows and masks from a generator reset to `seed` at the
start of each loop, so its metrics compare like with like across epochs.
A sample's embedding is the mean over all its windows of each window's
mean-pooled hidden states; it involves no randomness.
"""

from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch
from torch import Tensor

from reimp_shared.ranking import GeneRanker, IdfScheme, RankScore
from reimp_shared.tokens import IGNORE_INDEX
from reimp_tifbert.model import TifBERT, mask_genes, mlm_loss
from reimp_tifbert.sequences import all_windows, detected_in, rank_genes, sample_windows

RANKER_KEY = "gene_ranker"


class LitTifBERT(L.LightningModule):
    """TifBERT with the paper's settings where it gives them.

    Defaults are BERT-base (12 layers, 768-d, 12 heads, dropout 0.1) over
    512-token windows at stride 256 of each sample's top 10,000 genes, 15%
    of tokens masked 80/10/10, and AdamW at lr 1e-4. Genes are ranked by
    tf-idf with the entropy weight (`paper.md`, "For reimp").
    """

    def __init__(
        self,
        n_genes: int,
        rank_score: RankScore = "tfidf",
        idf_scheme: IdfScheme = "entropy",
        detection_threshold: float = 0.0,
        max_genes: int | None = 10_000,
        window: int = 512,
        stride: int = 256,
        mask_prob: float = 0.15,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_frac: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        embed_chunk: int = 128,
        seed: int = 0,
    ) -> None:
        """`max_genes=None` keeps every gene a sample expresses.

        `embed_chunk` bounds the windows encoded at once at predict time.
        """
        super().__init__()
        if not 0 < stride <= window:
            raise ValueError(f"stride must be in (0, window={window}] to cover every gene")
        if max_genes is not None and max_genes < 1:
            raise ValueError(f"max_genes must be positive or None, got {max_genes}")
        self.save_hyperparameters()
        self.model = TifBERT(
            n_genes=n_genes,
            window=window,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=int(d_model * mlp_ratio),
            dropout=dropout,
        )
        self.ranker: GeneRanker | None = None
        # Genes some training sample expresses: the only ones a sentence holds.
        self.detected: np.ndarray | None = None
        self._generator = torch.Generator().manual_seed(seed)

    # ---------- ranking ----------

    def _new_ranker(self) -> GeneRanker:
        hp = self.hparams
        return GeneRanker(hp.rank_score, hp.idf_scheme, hp.detection_threshold)

    def fit_ranker(self, values: np.ndarray) -> LitTifBERT:
        """Fit the gene ranker and the detection mask on `values` (training samples x genes)."""
        if values.shape[1] != self.hparams.n_genes:
            raise ValueError(f"{values.shape[1]} genes, but the model has {self.hparams.n_genes}")
        self.ranker = self._new_ranker().fit(values)
        self.detected = detected_in(values)
        return self

    def setup(self, stage: str) -> None:
        """Fit the ranker on the fold's training samples, unless a checkpoint brought one."""
        if self.ranker is not None:
            return
        data = getattr(self.trainer.datamodule, "data", None)
        if data is None:
            raise RuntimeError(
                "the gene ranker is fit on the DataModule's training rows: "
                "pass an ExpressionDataModule, or call fit_ranker first"
            )
        self.fit_ranker(data.values[data.rows("train")])

    def rank(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """(B, G) values -> ranked gene tokens (B, L) and each row's length (B,)."""
        if self.ranker is None:
            raise RuntimeError("fit the gene ranker first: fit_ranker, or trainer.fit")
        genes, lengths = rank_genes(
            self.ranker,
            values.cpu().numpy(),
            self.detected,
            self.hparams.max_genes,
            self.model.pad_id,
        )
        return torch.from_numpy(genes), torch.from_numpy(lengths)

    def on_before_batch_transfer(self, batch: dict[str, Tensor], dataloader_idx: int) -> dict:
        """Values -> ranked gene tokens, on the CPU: no expression value goes further."""
        genes, lengths = self.rank(batch["values"])
        return {"genes": genes, "lengths": lengths, "sample_index": batch["sample_index"]}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ranker is not None:
            checkpoint[RANKER_KEY] = {
                "offset": torch.from_numpy(self.ranker.offset_),
                "weight": torch.from_numpy(self.ranker.weight_),
                "detected": torch.from_numpy(self.detected),
            }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint.get(RANKER_KEY)
        if state is not None:
            self.ranker = self._new_ranker()
            self.ranker.offset_ = state["offset"].cpu().numpy()
            self.ranker.weight_ = state["weight"].cpu().numpy()
            self.detected = state["detected"].cpu().numpy()

    # ---------- steps ----------

    def _masked_step(
        self, batch: dict[str, Tensor], generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, int]:
        """One window per sample, masked: (loss, correct predictions, masked tokens)."""
        hp = self.hparams
        tokens, _ = sample_windows(
            batch["genes"], batch["lengths"], hp.window, hp.stride, self.model.pad_id, generator
        )
        inputs, labels = mask_genes(tokens, hp.n_genes, hp.mask_prob, generator)
        selected = labels != IGNORE_INDEX
        logits = self.model.logits(self.model(inputs)[selected])
        targets = labels[selected]
        correct = (logits.argmax(dim=-1) == targets).sum()
        return mlm_loss(logits, targets), correct, len(targets)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, _, n_masked = self._masked_step(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=max(n_masked, 1))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, correct, n_masked = self._masked_step(batch, self._generator)
        # Weighted by masked tokens, so epoch values are per-token means.
        n = max(n_masked, 1)
        self.log("val/loss", loss, prog_bar=True, batch_size=n)
        self.log("val/accuracy", correct / n, batch_size=n)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Each window mean-pooled, then the windows of each sample averaged."""
        hp = self.hparams
        genes = batch["genes"]
        tokens, rows = all_windows(genes, batch["lengths"], hp.window, hp.stride, self.model.pad_id)
        pooled = torch.zeros(0, hp.d_model, device=genes.device)
        if len(tokens):
            pooled = torch.cat([self.model.embed(chunk) for chunk in tokens.split(hp.embed_chunk)])
        total = pooled.new_zeros(len(genes), hp.d_model).index_add_(0, rows, pooled)
        count = torch.zeros(len(genes), device=genes.device).index_add_(
            0, rows, torch.ones(len(rows), device=genes.device)
        )
        # A sample with no expressed genes has no window, and embeds as zeros.
        embedding = total / count.clamp_min(1.0).unsqueeze(1)
        return {"sample_index": batch["sample_index"], "embedding": embedding}

    def configure_optimizers(self):
        hp = self.hparams
        total_steps = self.trainer.estimated_stepping_batches
        if math.isinf(total_steps):
            raise ValueError("the lr schedule needs a finite run: set max_epochs or max_steps")
        if total_steps < 1:
            raise ValueError("no training steps: is batch_size larger than the training split?")
        # Decay weight matrices and embeddings; leave biases and norms alone, as BERT does.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": hp.weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=hp.lr, betas=tuple(hp.betas), eps=hp.eps)
        # BERT's schedule: linear warmup, then linear decay.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hp.lr,
            total_steps=int(total_steps),
            pct_start=hp.warmup_frac,
            anneal_strategy="linear",
            cycle_momentum=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
