"""LightningModule: one MLP encoder, one of four objectives.

  scarf  the clean profile and one SCARF-corrupted copy through encoder and
         projection head, NT-Xent between them.
  vime   one VIME-corrupted copy through the encoder; decoders estimate the
         corruption mask and the clean profile.
  byol   the clean profile and a VIME-corrupted copy through the online
         network; each predicts the EMA target network's projection of the
         other.
  none   no training: the encoder at its random initialization, the
         untrained-encoder control. One pass over the training samples
         sets the BatchNorm running statistics (an exact average, not an
         exponential one) and nothing else.

Batches arrive per-sample transformed (`lognorm`, or `log1p` of TPM) from
the shared DataModule. At the start of `fit` this module fits, on the
fold's training samples only: a per-gene z-score (`GeneScaler`, saved with
the model), each gene's scaled [min, max] for SCARF's uniform replacement,
and the pool of scaled training samples every marginal replacement is drawn
from (rebuilt at each fit, not saved). Every scaled input, at training and
at embedding, is clipped to that [min, max] (`scale`): a gene nearly
constant over training otherwise gives a held-out sample z-scores in the
thousands. Embeddings are the encoder's output on scaled, uncorrupted
inputs, in eval mode.

Training corruptions come from the global RNG (seeded by `seed_everything`)
and are redrawn every batch. Validation corruptions come from a generator
reset to `seed` at the start of each validation loop, so the validation
loss compares like with like across epochs.
"""

from __future__ import annotations

import copy
from typing import Literal

import lightning as L
import numpy as np
import torch
from torch import Tensor, nn

from reimp_tabssl.corrupt import (
    bernoulli_mask,
    corrupt,
    fixed_count_mask,
    marginal_draws,
    uniform_draws,
)
from reimp_tabssl.model import (
    Decoder,
    GeneScaler,
    MLPEncoder,
    ProjectionHead,
    byol_loss,
    ema_update,
    nt_xent,
    vime_losses,
)

Objective = Literal["scarf", "vime", "byol", "none"]
OBJECTIVES: tuple[str, ...] = ("scarf", "vime", "byol", "none")
Replacement = Literal["uniform", "marginal"]
REPLACEMENTS: tuple[str, ...] = ("uniform", "marginal")

# Learning rates of Table S3: SCARF Adam 1e-4, VIME RMSprop 1e-3, BYOL
# Adam 1e-4 (Table S3 says SGD; Adam is what the paper's code runs).
DEFAULT_LR = {"scarf": 1e-4, "vime": 1e-3, "byol": 1e-4}


class LitTabSSL(L.LightningModule):
    """SCARF, VIME or BYOL on the paper's 4 × 256 MLP, with its settings (Table S3).

    Defaults: corruption rate 0.3 for every objective; SCARF replaces from
    Uniform[min, max] (`scarf_replacement="marginal"` for the original
    SCARF's empirical marginal) under NT-Xent at τ = 1.0; VIME weighs the
    reconstruction by α = 2.0; BYOL uses 4096-wide heads and an EMA decay
    of 0.9. Batch size and epochs belong to the DataModule and Trainer.
    """

    def __init__(
        self,
        n_genes: int,
        objective: Objective = "scarf",
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.2,
        corruption_rate: float = 0.3,
        scarf_replacement: Replacement = "uniform",
        temperature: float = 1.0,
        decoder_layers: int = 4,
        vime_alpha: float = 2.0,
        byol_hidden_dim: int = 4096,
        byol_ema: float = 0.9,
        lr: float | None = None,
        seed: int = 0,
    ) -> None:
        """`lr=None` takes the objective's learning rate from `DEFAULT_LR`."""
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
        if scarf_replacement not in REPLACEMENTS:
            raise ValueError(
                f"unknown scarf_replacement {scarf_replacement!r}; expected one of {REPLACEMENTS}"
            )
        if not 0.0 <= corruption_rate <= 1.0:
            raise ValueError(f"corruption_rate must be in [0, 1], got {corruption_rate}")
        self.save_hyperparameters()
        self.scaler = GeneScaler(n_genes)
        self.encoder = MLPEncoder(n_genes, hidden_dim, n_layers, dropout)
        # Each gene's scaled range over the training samples: SCARF's
        # uniform replacement range, and the clip on every scaled input.
        self.register_buffer("low", torch.zeros(n_genes))
        self.register_buffer("high", torch.zeros(n_genes))
        # Scaled training samples, the source of marginal replacements.
        # Rebuilt from the data at every fit, so kept out of checkpoints.
        self.register_buffer("pool", torch.empty(0, n_genes), persistent=False)

        if objective == "scarf":
            self.head = ProjectionHead(hidden_dim, hidden_dim, hidden_dim)
        elif objective == "vime":
            self.mask_decoder = Decoder(hidden_dim, n_genes, hidden_dim, decoder_layers)
            self.feature_decoder = Decoder(hidden_dim, n_genes, hidden_dim, decoder_layers)
        elif objective == "byol":
            self.projector = ProjectionHead(hidden_dim, byol_hidden_dim, hidden_dim)
            self.predictor = ProjectionHead(hidden_dim, byol_hidden_dim, hidden_dim)
            self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
            self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        else:
            # A cumulative average: one training pass gives the exact mean
            # of the batch statistics.
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.momentum = None
        self._generator = torch.Generator().manual_seed(seed)

    @property
    def needs_pool(self) -> bool:
        hp = self.hparams
        return hp.objective in ("vime", "byol") or (
            hp.objective == "scarf" and hp.scarf_replacement == "marginal"
        )

    # ------------------------------------------------------------------
    # Statistics fit on the training samples

    def fit_statistics(self, train_values) -> None:
        """Fit the scaler, SCARF's ranges and the replacement pool on training samples.

        `train_values` is (n_train, n_genes), as the DataModule transforms
        them: the fold's training rows and nothing else.
        """
        train_values = np.asarray(train_values)
        if not np.issubdtype(train_values.dtype, np.floating):
            raise TypeError(
                "tabssl z-scores log expression: use the `lognorm` or `log1p` transform"
            )
        self.scaler.fit(train_values)
        device = self.scaler.mean.device
        scaled = self.scaler(torch.as_tensor(train_values, dtype=torch.float32, device=device))
        self.low.copy_(scaled.min(dim=0).values)
        self.high.copy_(scaled.max(dim=0).values)
        self.pool = scaled if self.needs_pool else scaled.new_empty(0, scaled.shape[1])

    def scale(self, values: Tensor) -> Tensor:
        """z-scored `values`, clipped to each gene's scaled range over the training samples.

        The clip leaves training samples as they are. A held-out sample can
        lie far outside: a gene expressed in one training sample has std
        ~1/√n_train of its peak, so a higher held-out value scales to
        thousands of standard deviations and swamps the first layer.
        """
        return torch.clamp(self.scaler(values), self.low, self.high)

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return
        data = getattr(self.trainer.datamodule, "data", None)
        if data is None:
            raise RuntimeError(
                "tabssl fits its statistics on the training samples of an ExpressionDataModule: "
                "pass one to fit"
            )
        self.fit_statistics(data.values[data.rows("train")])

    # ------------------------------------------------------------------
    # Corruptions and objectives, on scaled values

    def _vime_corrupt(self, x: Tensor, generator: torch.Generator | None) -> tuple[Tensor, Tensor]:
        """VIME's corruption: the corrupted copy, and where it differs from `x`."""
        n, g = x.shape
        mask = bernoulli_mask(n, g, self.hparams.corruption_rate, x.device, generator)
        x_tilde = corrupt(x, mask, marginal_draws(self.pool, n, generator))
        return x_tilde, x_tilde != x

    def _scarf_corrupt(self, x: Tensor, generator: torch.Generator | None) -> Tensor:
        n, g = x.shape
        mask = fixed_count_mask(n, g, self.hparams.corruption_rate, x.device, generator)
        if self.hparams.scarf_replacement == "uniform":
            replacement = uniform_draws(self.low, self.high, n, generator)
        else:
            replacement = marginal_draws(self.pool, n, generator)
        return corrupt(x, mask, replacement)

    def _scarf(self, x: Tensor, generator: torch.Generator | None) -> dict[str, Tensor]:
        z_clean = self.head(self.encoder(x))
        z_corrupt = self.head(self.encoder(self._scarf_corrupt(x, generator)))
        return {"loss": nt_xent(z_clean, z_corrupt, self.hparams.temperature)}

    def _vime(self, x: Tensor, generator: torch.Generator | None) -> dict[str, Tensor]:
        x_tilde, changed = self._vime_corrupt(x, generator)
        h = self.encoder(x_tilde)
        bce, mse = vime_losses(self.mask_decoder(h), self.feature_decoder(h), changed, x)
        return {"loss": bce + self.hparams.vime_alpha * mse, "mask_bce": bce, "feature_mse": mse}

    def _byol(self, x: Tensor, generator: torch.Generator | None) -> dict[str, Tensor]:
        x_tilde, _ = self._vime_corrupt(x, generator)
        p_clean = self.predictor(self.projector(self.encoder(x)))
        p_corrupt = self.predictor(self.projector(self.encoder(x_tilde)))
        with torch.no_grad():
            t_clean = self.target_projector(self.target_encoder(x))
            t_corrupt = self.target_projector(self.target_encoder(x_tilde))
        loss = byol_loss(p_clean, t_corrupt) + byol_loss(p_corrupt, t_clean)
        return {"loss": loss.mean()}

    def losses(self, values: Tensor, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """The objective's loss (and its parts) on a batch of transformed values."""
        x = self.scale(values)
        objective = self.hparams.objective
        if objective == "scarf":
            return self._scarf(x, generator)
        if objective == "vime":
            return self._vime(x, generator)
        if objective == "byol":
            return self._byol(x, generator)
        raise ValueError("objective `none` has no loss")

    # ------------------------------------------------------------------
    # Lightning hooks

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor | None:
        values = batch["values"]
        if self.hparams.objective == "none":
            # BatchNorm running statistics only; returning None skips the step.
            with torch.no_grad():
                self.encoder(self.scale(values))
            return None
        losses = self.losses(values)
        n = len(values)
        self.log("train/loss", losses["loss"], prog_bar=True, batch_size=n)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"train/{name}", value, batch_size=n)
        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.hparams.objective == "byol":
            decay = self.hparams.byol_ema
            ema_update(self.target_encoder, self.encoder, decay)
            ema_update(self.target_projector, self.projector, decay)

    def on_validation_epoch_start(self) -> None:
        self._generator.manual_seed(self.hparams.seed)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        if self.hparams.objective == "none":
            return
        values = batch["values"]
        for name, value in self.losses(values, self._generator).items():
            self.log(f"val/{name}", value, prog_bar=name == "loss", batch_size=len(values))

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Encoder embeddings of scaled, clipped, uncorrupted inputs. No head."""
        embedding = self.encoder(self.scale(batch["values"]))
        return {"sample_index": batch["sample_index"], "embedding": embedding}

    def configure_optimizers(self):
        hp = self.hparams
        if hp.objective == "none":
            return None
        lr = hp.lr if hp.lr is not None else DEFAULT_LR[hp.objective]
        params = [p for p in self.parameters() if p.requires_grad]
        if hp.objective == "vime":
            # Keras's RMSprop defaults (rho 0.9, eps 1e-7): the paper's VIME is Keras.
            return torch.optim.RMSprop(params, lr=lr, alpha=0.9, eps=1e-7)
        return torch.optim.Adam(params, lr=lr)
