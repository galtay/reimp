"""LightningModule: losses, KL warm-up and optimization for Tybalt and the MMD-AE.

One module, two configurations (`configs/tybalt.yaml`, `configs/mmdae_*.yaml`):

  reconstruction  `bce`: sigmoid output, per-gene BCE summed over genes
                  (Tybalt; needs min-max scaled inputs). `mse`: linear
                  output, squared error averaged over genes (MMD-AE).
  regularizer     `kl`: β · KL with β warmed up by κ per epoch (Tybalt).
                  `mmd`: MMD between the batch's sampled z and
                  `mmd_prior_samples` N(0, I) draws, weight 1 (MMD-AE).
  n_classes       > 0 adds a classifier head on the sampled z, trained with
                  cross-entropy at weight 1; linked from the DataModule's
                  `supervision`.

The validation loss is the full objective — β = 1 whatever the warm-up —
so early stopping compares like with like across epochs. Training noise
comes from the global RNG (seeded by `seed_everything`); validation noise
(ε and the prior draws) from a generator reset to `seed` at the start of
each validation loop. The embedding is the posterior mean μ, no sampling.
"""

from __future__ import annotations

from typing import Literal

import lightning as L
import torch
from torch import Tensor
from torch.nn import functional as F

from reimp_vae.model import (
    Autoencoder,
    Heads,
    bce_loss,
    kl_divergence,
    kl_weight,
    mmd,
    mse_loss,
    standard_normal,
)

Reconstruction = Literal["bce", "mse"]
Regularizer = Literal["kl", "mmd"]
RECONSTRUCTIONS: tuple[str, ...] = ("bce", "mse")
REGULARIZERS: tuple[str, ...] = ("kl", "mmd")


class LitVAE(L.LightningModule):
    """A Gaussian-latent autoencoder. Defaults are Tybalt's (paper.md, "For reimp").

    Tybalt: 256-d latent (the paper's 100, widened to reimp's common size),
    no hidden layer, BatchNorm + ReLU on both heads,
    sigmoid decoder with BCE, KL warmed up with κ = 1, Glorot init, Adam at
    5e-4. The MMD-AE config sets a 121-d latent, one hidden layer of
    0.2 · n_genes, linear heads, MSE, MMD, Flexynesis's Xavier init
    (`xavier_init`, `glorot_init: false`) and Adam at 1.72e-3.
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int = 0,
        latent_dim: int = 256,
        hidden_dim: int | None = None,
        hidden_factor: float = 0.0,
        heads: Heads = "bn_relu",
        reconstruction: Reconstruction = "bce",
        regularizer: Regularizer = "kl",
        kappa: float = 1.0,
        mmd_prior_samples: int = 200,
        logvar_max: float | None = None,
        class_hidden: int = 32,
        class_dropout: float = 0.1,
        glorot_init: bool = True,
        xavier_init: bool = False,
        lr: float = 5e-4,
        seed: int = 0,
    ) -> None:
        """The hidden layer is `hidden_dim` wide if given, else round(`hidden_factor` · n_genes);
        0 means none (Tybalt). `n_classes` 0 means no classifier head. `logvar_max` caps
        the log-variance before it is exponentiated (see `Autoencoder`); None leaves it free."""
        super().__init__()
        if reconstruction not in RECONSTRUCTIONS:
            raise ValueError(
                f"unknown reconstruction {reconstruction!r}; expected one of {RECONSTRUCTIONS}"
            )
        if regularizer not in REGULARIZERS:
            raise ValueError(f"unknown regularizer {regularizer!r}; expected one of {REGULARIZERS}")
        self.save_hyperparameters()
        width = hidden_dim if hidden_dim is not None else round(hidden_factor * n_genes)
        self.model = Autoencoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=width,
            heads=heads,
            n_classes=n_classes,
            class_hidden=class_hidden,
            class_dropout=class_dropout,
            glorot_init=glorot_init,
            xavier_init=xavier_init,
            logvar_max=logvar_max,
        )
        self._generator = torch.Generator().manual_seed(seed)

    def regularizer_weight(self) -> float:
        """The regularizer's weight this epoch in training: β for `kl`, 1 for `mmd`."""
        if self.hparams.regularizer == "kl":
            return kl_weight(self.current_epoch, self.hparams.kappa)
        return 1.0

    def losses(
        self, batch: dict[str, Tensor], generator: torch.Generator | None = None
    ) -> dict[str, Tensor]:
        """Batch means of each term: `recon`, `kl` or `mmd`, and `class` and `accuracy`."""
        hp = self.hparams
        x = batch["values"]
        out = self.model(x, generator)
        if hp.reconstruction == "bce":
            recon = bce_loss(out["output"], x)
        else:
            recon = mse_loss(out["output"], x)
        losses = {"recon": recon.mean()}
        if hp.regularizer == "kl":
            losses["kl"] = kl_divergence(out["mu"], out["logvar"]).mean()
        else:
            prior = standard_normal((hp.mmd_prior_samples, hp.latent_dim), x.device, generator)
            losses["mmd"] = mmd(out["z"], prior)
        if "logits" in out:
            if "label" not in batch:
                raise KeyError("a supervised model needs `label` in every batch")
            losses["class"] = F.cross_entropy(out["logits"], batch["label"])
            losses["accuracy"] = (out["logits"].argmax(-1) == batch["label"]).float().mean()
        return losses

    def objective(self, losses: dict[str, Tensor], weight: float) -> Tensor:
        """recon + weight · regularizer + class, the unweighted sum at weight 1."""
        loss = losses["recon"] + weight * losses[self.hparams.regularizer]
        return loss + losses["class"] if "class" in losses else loss

    def _log(self, stage: str, losses: dict[str, Tensor], loss: Tensor, n: int) -> None:
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=n)
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, batch_size=n)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x = batch["values"]
        if batch_idx == 0 and self.hparams.reconstruction == "bce":
            if not x.is_floating_point() or x.min() < 0 or x.max() > 1:
                raise ValueError("BCE needs values in [0, 1]: use the DataModule's minmax scaling")
        losses = self.losses(batch)
        weight = self.regularizer_weight()
        loss = self.objective(losses, weight)
        self._log("train", losses, loss, len(x))
        if self.hparams.regularizer == "kl":
            self.log("train/beta", weight, batch_size=len(x))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        losses = self.losses(batch, self._generator)
        self._log("val", losses, self.objective(losses, 1.0), len(batch["values"]))

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Posterior means, the published embedding of both papers."""
        mu, _ = self.model.encoder(batch["values"])
        return {"sample_index": batch["sample_index"], "embedding": mu}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
