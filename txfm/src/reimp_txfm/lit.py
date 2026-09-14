"""LightningModule: masking, loss and optimization for TxFM.

Batches arrive library-size normalized and log1p'd — the shared
DataModule's `lognorm` transform, with the same `library_size` the output
activation is bounded by (the CLI links the two). The K-of-G mask is drawn
here, on the accelerator.

Training masks come from the global RNG (seeded by `seed_everything`).
Validation and embedding masks come from a generator reset to `seed` at the
start of each loop, so validation metrics compare like with like across
epochs and embeddings are reproducible.
"""

from __future__ import annotations

import math

import lightning as L
import torch
from torch import Tensor

from reimp_txfm.metrics import holdout_metrics
from reimp_txfm.model import TxFM, poisson_loss, sample_unmasked


class LitTxFM(L.LightningModule):
    """TxFM with the paper's settings where it gives them (§3, §A.1, Table 6).

    Defaults are the TxFM-S backbone (6 blocks, 6 heads, 384-d, stochastic
    depth 0.1) with K = 2048 unmasked genes, L = 1e5, a 4-layer decoder, and
    AdamW at max lr 1e-3 under a one-cycle cosine schedule with 10% warmup.
    """

    def __init__(
        self,
        n_genes: int,
        n_unmasked: int = 2048,
        library_size: float = 1e5,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        decoder_layers: int = 4,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        layer_scale_init: float | None = 1e-4,
        lr: float = 1e-3,
        weight_decay: float | None = None,
        warmup_frac: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        embed_draws: int = 1,
        seed: int = 0,
    ) -> None:
        """`weight_decay=None` applies the paper's rule, 1 / (lr · total steps).

        `embed_draws` averages each CLS embedding over that many independent
        masks at predict time.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = TxFM(
            n_genes=n_genes,
            library_size=library_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=int(d_model * mlp_ratio),
            decoder_layers=decoder_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
        )
        self._generator = torch.Generator().manual_seed(seed)

    def _mask(
        self, values: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """(B, G) log-normalized values -> unmasked positions (B, K) and their values."""
        if not values.is_floating_point():
            raise TypeError("TxFM expects log-normalized values: use the `lognorm` transform")
        n_rows, n_genes = values.shape
        k = min(self.hparams.n_unmasked, n_genes)
        idx = sample_unmasked(n_rows, n_genes, k, device=values.device, generator=generator)
        return idx, values.gather(1, idx)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        target = batch["values"]
        loss = poisson_loss(self.model(*self._mask(target)), target).mean()
        self.log("train/loss", loss, prog_bar=True, batch_size=len(target))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        target = batch["values"]
        idx, unmasked = self._mask(target, self._generator)
        x_hat = self.model(idx, unmasked)
        n = len(target)
        self.log("val/loss", poisson_loss(x_hat, target).mean(), prog_bar=True, batch_size=n)
        for name, value in holdout_metrics(x_hat, target, idx).items():
            self.log(f"val/{name}", value, batch_size=n)

    def on_predict_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """CLS embeddings, averaged over `embed_draws` masks. No decoder."""
        draws = self.hparams.embed_draws
        values = batch["values"]
        embedding = sum(
            self.model.encoder(*self._mask(values, self._generator)) for _ in range(draws)
        )
        return {"sample_index": batch["sample_index"], "embedding": embedding / draws}

    def configure_optimizers(self):
        hp = self.hparams
        total_steps = self.trainer.estimated_stepping_batches
        if math.isinf(total_steps):
            raise ValueError("the lr schedule needs a finite run: set max_epochs or max_steps")
        if total_steps < 1:
            raise ValueError("no training steps: is batch_size larger than the training split?")
        total_steps = int(total_steps)
        weight_decay = hp.weight_decay if hp.weight_decay is not None else 1 / (hp.lr * total_steps)
        # Decay weight matrices and embeddings; leave biases, norms and
        # LayerScale gains alone.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=hp.lr, betas=tuple(hp.betas), eps=hp.eps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hp.lr,
            total_steps=total_steps,
            pct_start=hp.warmup_frac,
            anneal_strategy="cos",
            # Cycling momentum would move AdamW's beta1 off the paper's 0.9.
            cycle_momentum=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
