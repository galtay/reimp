"""LightningModule: tokenizer, masking, loss and optimization for BulkRNABert.

Batches arrive as log TPM — the shared DataModule's `tpm_unstranded` with
`transform: log1p`. The `BinTokenizer`'s maximum is fit when training
starts, on the fold's training rows only, and kept in the hyperparameters,
so a checkpoint carries it and embedding bins every sample by the same
training-set statistic. Tokens and BERT's 80/10/10 corruption are made
here, on the accelerator.

Training masks come from the global RNG (seeded by `seed_everything`).
Validation masks come from a CPU generator reset to `seed` at the start of
each validation loop, so validation metrics compare like with like across
epochs and devices. Embedding needs no mask: every gene's true token goes
in, as at the paper's inference.
"""

from __future__ import annotations

import math

import lightning as L
import torch
from torch import Tensor

from reimp_bulkrnabert.model import BulkRNABert, mlm_accuracy, mlm_loss
from reimp_shared.tokens import DEFAULT_N_BINS, BinTokenizer, mask_tokens


class LitBulkRNABert(L.LightningModule):
    """BulkRNABert with the paper's settings where it gives them (§3.1).

    Defaults are the paper's encoder (4 layers, 8 heads, d = 256, FFN 512)
    over 64 expression bins, with 15% of tokens selected for corruption.
    The optimizer, AdamW, is the paper's; its learning rate, weight decay
    and one-cycle cosine schedule with 10% warmup are ours.
    """

    def __init__(
        self,
        n_genes: int,
        n_bins: int = DEFAULT_N_BINS,
        token_max: float | None = None,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.0,
        attn_chunk: int | None = None,
        mask_prob: float = 0.15,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_frac: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        seed: int = 0,
    ) -> None:
        """`token_max=None` fits the tokenizer's maximum on the training rows when fit starts.

        A checkpoint records the fitted value here, so a model loaded from
        it tokenizes as it did in training. `attn_chunk` computes attention
        in chunks of that many query genes: the same model, in kernels small
        enough for MPS (`model.Block`).
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BulkRNABert(
            n_genes=n_genes,
            n_bins=n_bins,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=dim_ff,
            dropout=dropout,
            attn_chunk=attn_chunk,
        )
        self.tokenizer = BinTokenizer(n_bins)
        if token_max is not None:
            self.tokenizer.max_ = float(token_max)
        self._generator = torch.Generator().manual_seed(seed)

    def fit_tokenizer(self, values) -> None:
        """Fit the tokenizer's maximum on `values` (training samples x genes)."""
        self.tokenizer.fit(values)
        self.hparams.token_max = self.tokenizer.max_

    def setup(self, stage: str) -> None:
        """Before training, fit the tokenizer on the fold's training rows unless already set."""
        if stage != "fit" or self.tokenizer.max_ is not None:
            return
        data = getattr(self.trainer.datamodule, "data", None)
        if data is None:
            raise RuntimeError(
                "fitting the tokenizer needs the shared ExpressionDataModule (or pass token_max)"
            )
        self.fit_tokenizer(data.values[data.rows("train")])

    def _tokens(self, values: Tensor) -> Tensor:
        if self.tokenizer.max_ is None:
            raise RuntimeError("the tokenizer is not fit: train the model or pass token_max")
        return self.tokenizer.tokens(values)

    def _corrupt(
        self, values: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """(B, G) log TPM -> corrupted input tokens and MLM labels, on the model's device."""
        tokens = self._tokens(values)
        if generator is not None:
            tokens = tokens.to(generator.device)
        inputs, labels = mask_tokens(
            tokens, self.model.mask_id, self.hparams.n_bins, self.hparams.mask_prob, generator
        )
        return inputs.to(values.device), labels.to(values.device)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        inputs, labels = self._corrupt(batch["values"])
        loss = mlm_loss(self.model(inputs), labels)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(inputs))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        inputs, labels = self._corrupt(batch["values"], self._generator)
        logits = self.model(inputs)
        n = len(inputs)
        self.log("val/loss", mlm_loss(logits, labels), prog_bar=True, batch_size=n)
        self.log("val/accuracy", mlm_accuracy(logits, labels), batch_size=n)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Mean over genes of the final hidden states, from uncorrupted tokens."""
        embedding = self.model.embed(self._tokens(batch["values"]))
        return {"sample_index": batch["sample_index"], "embedding": embedding}

    def configure_optimizers(self):
        hp = self.hparams
        total_steps = self.trainer.estimated_stepping_batches
        if math.isinf(total_steps):
            raise ValueError("the lr schedule needs a finite run: set max_epochs or max_steps")
        if total_steps < 1:
            raise ValueError("no training steps: is batch_size larger than the training split?")
        # Decay weight matrices and embeddings; leave biases and norms alone.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": hp.weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=hp.lr, betas=tuple(hp.betas), eps=hp.eps)
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
