"""LightningModule: tokens, masking, loss and optimization for MOJO.

Batches arrive as log TPM — the shared DataModule's `tpm_unstranded` with
`transform: log1p` — and are binned here by a `BinTokenizer`. Its maximum
is fit when `fit` starts, on the fold's training rows only (EVALS.md rule
3), and stored as the `token_max` hyperparameter, so a checkpoint carries it
to `mojo-embed`. Tokens are masked 15%, 80/10/10, on the accelerator.

Training masks come from the global RNG (seeded by `seed_everything`).
Validation masks come from a generator reset to `seed` at the start of each
loop and drawn on the CPU, so validation metrics compare like with like
across epochs and accelerators. Embeddings read unmasked tokens.
"""

from __future__ import annotations

import math
from typing import Literal

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from reimp_mojo.model import MOJO, masked_token_loss
from reimp_shared.data import ExpressionData
from reimp_shared.tokens import DEFAULT_N_BINS, BinTokenizer, mask_tokens

GeneOrder = Literal["dataset", "genome"]
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]


def genome_order(genes: pd.DataFrame) -> np.ndarray:
    """Row positions of `genes` sorted by chromosome (1–22, X, Y, M) and start.

    Genes on other contigs or without a position go last, in dataset order.
    """
    rank = genes["chromosome"].map({c: i for i, c in enumerate(CHROMOSOMES)})
    rank = rank.fillna(len(CHROMOSOMES)).to_numpy(dtype=float)
    start = genes["start"].to_numpy(dtype=float)
    return np.lexsort((np.nan_to_num(start, nan=np.inf), rank))


class LitMOJO(L.LightningModule):
    """MOJO's RNA half with BulkRNABert's tokens and objective.

    Defaults are sized for ~8,000 training samples per fold: a 128-d token
    and gene embedding, channels rising from 128 to 256 over 8 halving
    blocks (19,944 genes -> 78 pooled positions), 4 transformer layers of 8
    heads with a 512-wide SwiGLU, trained by AdamW under a one-cycle cosine
    schedule.
    """

    def __init__(
        self,
        n_genes: int,
        n_bins: int = DEFAULT_N_BINS,
        mask_prob: float = 0.15,
        token_max: float | None = None,
        gene_order: GeneOrder = "dataset",
        embed_dim: int = 128,
        conv_channels: int = 128,
        d_model: int = 256,
        n_down: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        stem_kernel: int = 15,
        kernel_size: int = 5,
        dropout: float = 0.0,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_frac: float = 0.05,
        betas: tuple[float, float] = (0.9, 0.999),
        seed: int = 0,
    ) -> None:
        """`token_max=None` fits the tokenizer's maximum on the training rows when `fit` starts.

        `gene_order="genome"` reorders the genes by chromosome and position
        before the convolutions (the paper's ablation); the default keeps
        the dataset's order.
        """
        super().__init__()
        if gene_order not in ("dataset", "genome"):
            raise ValueError(f"unknown gene_order {gene_order!r}; expected 'dataset' or 'genome'")
        self.save_hyperparameters()
        self.model = MOJO(
            n_genes=n_genes,
            n_bins=n_bins,
            embed_dim=embed_dim,
            conv_channels=conv_channels,
            d_model=d_model,
            n_down=n_down,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=int(d_model * mlp_ratio),
            stem_kernel=stem_kernel,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.tokenizer = BinTokenizer(n_bins)
        if token_max is not None:
            self.tokenizer.max_ = float(token_max)
        # Model position -> dataset column, set in `setup` for genome order.
        self.register_buffer("gene_perm", None, persistent=False)
        self._generator = torch.Generator().manual_seed(seed)

    def _data(self) -> ExpressionData:
        data = getattr(self.trainer.datamodule, "data", None)
        if data is None:
            raise RuntimeError("LitMOJO reads its genes and training rows from the DataModule")
        return data

    def setup(self, stage: str) -> None:
        if self.hparams.gene_order == "genome":
            self.gene_perm = torch.from_numpy(genome_order(self._data().genes))
        if stage == "fit" and self.hparams.token_max is None:
            data = self._data()
            self.tokenizer.fit(data.values[data.rows("train")])
            self.hparams.token_max = self.tokenizer.max_

    def _tokens(self, values: Tensor) -> Tensor:
        """(B, G) log TPM -> (B, G) bin tokens, in the model's gene order."""
        if not values.is_floating_point():
            raise TypeError("MOJO expects log TPM: use tpm_unstranded with the log1p transform")
        if self.tokenizer.max_ is None:
            raise RuntimeError("the tokenizer is not fit: train with `fit` or pass token_max")
        tokens = self.tokenizer.tokens(values)
        return tokens if self.gene_perm is None else tokens[:, self.gene_perm.to(tokens.device)]

    def _mask(
        self, tokens: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Corrupted inputs and labels; with a `generator`, drawn on the CPU."""
        hp = self.hparams
        if generator is None:
            return mask_tokens(tokens, self.model.mask_id, hp.n_bins, hp.mask_prob)
        inputs, labels = mask_tokens(
            tokens.cpu(), self.model.mask_id, hp.n_bins, hp.mask_prob, generator=generator
        )
        return inputs.to(tokens.device), labels.to(tokens.device)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        inputs, labels = self._mask(self._tokens(batch["values"]))
        loss, accuracy = masked_token_loss(self.model(inputs), labels)
        n = len(inputs)
        self.log("train/loss", loss, prog_bar=True, batch_size=n)
        self.log("train/accuracy", accuracy, batch_size=n)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        inputs, labels = self._mask(self._tokens(batch["values"]), self._generator)
        loss, accuracy = masked_token_loss(self.model(inputs), labels)
        n = len(inputs)
        self.log("val/loss", loss, prog_bar=True, batch_size=n)
        self.log("val/accuracy", accuracy, batch_size=n)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Mean over pooled positions of the last transformer layer. No masking, no decoder."""
        embedding = self.model.embed(self._tokens(batch["values"]))
        return {"sample_index": batch["sample_index"], "embedding": embedding}

    def configure_optimizers(self):
        hp = self.hparams
        total_steps = self.trainer.estimated_stepping_batches
        if math.isinf(total_steps):
            raise ValueError("the lr schedule needs a finite run: set max_epochs or max_steps")
        if total_steps < 1:
            raise ValueError("no training steps: is batch_size larger than the training split?")
        # Decay weight matrices, convolution kernels and embeddings; leave
        # biases and norms alone.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": hp.weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=hp.lr, betas=tuple(hp.betas))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hp.lr,
            total_steps=int(total_steps),
            pct_start=hp.warmup_frac,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
