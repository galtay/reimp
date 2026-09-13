"""Autoencoders with a Gaussian latent: architecture and losses for both papers.

  Tybalt (Way and Greene 2018, as in greenelab/tybalt's code)
    encoder   genes -> latent directly; the mean and log-variance heads are
              each Dense -> BatchNorm -> ReLU, so posterior means are
              non-negative and every posterior variance is at least 1.
    decoder   latent -> genes, sigmoid output; Glorot-uniform weights.
    loss      per-gene BCE summed over genes + β · KL(q(z|x) ‖ N(0, I)),
              β warmed up from 0 by κ per epoch.

  MMD-AE (Pande et al. 2026, Flexynesis `supervised_vae`)
    encoder   genes -> hidden (Linear, LeakyReLU(0.2), BatchNorm) -> linear
              mean and log-variance heads.
    decoder   latent -> hidden (the same block) -> genes, linear output.
    loss      MSE averaged over genes + MMD(z, N(0, I) draws), plus, when
              supervised, the cross-entropy of a classifier head on the
              sampled z: an unweighted sum.

Both sample z = μ + exp(logvar / 2) · ε and embed a sample as μ. The
MMD-AE's code used the raw log-variance head as a standard deviation and a
sigmoid output against z-scored targets; both are fixed here (see README).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Heads = Literal["bn_relu", "linear"]
HEADS: tuple[str, ...] = ("bn_relu", "linear")


# ---------- losses ----------


def bce_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Per-sample binary cross-entropy summed over genes, from decoder logits.

    Tybalt's `original_dim · binary_crossentropy`: the mean over genes times
    the number of genes. Computed from logits for stability; the
    reconstruction itself is `sigmoid(logits)`. Targets must lie in [0, 1].
    """
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)


def mse_loss(x_hat: Tensor, target: Tensor) -> Tensor:
    """Per-sample squared error averaged over genes; its batch mean is Flexynesis's MSE."""
    return (x_hat - target).pow(2).mean(-1)


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Per-sample KL(N(μ, e^logvar) ‖ N(0, I)), summed over latent dimensions."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)


def kl_weight(epoch: int, kappa: float) -> float:
    """KL warm-up (Sønderby et al. 2016): β = 0 in epoch 0, raised by κ each epoch, capped at 1."""
    return min(1.0, epoch * kappa)


def gaussian_kernel(x: Tensor, y: Tensor) -> Tensor:
    """(n, m) kernel exp(−mean_k (x_k − y_k)² / d) between rows of x (n, d) and y (m, d).

    InfoVAE's kernel as Flexynesis computes it: the mean over dimensions is
    divided by d again, so this is exp(−‖x − y‖² / d²).
    """
    d = x.shape[-1]
    return torch.exp(-(x.unsqueeze(1) - y.unsqueeze(0)).pow(2).mean(-1) / d)


def mmd(x: Tensor, y: Tensor) -> Tensor:
    """Squared maximum mean discrepancy between samples x and y under `gaussian_kernel`.

    The biased (V-statistic) estimate over every pair, diagonals included,
    as Flexynesis computes it; zero when x and y are the same sample.
    """
    return (
        gaussian_kernel(x, x).mean()
        + gaussian_kernel(y, y).mean()
        - 2 * gaussian_kernel(x, y).mean()
    )


# ---------- sampling ----------


def standard_normal(
    shape: tuple[int, ...],
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """N(0, 1) draws. With a `generator` they are drawn on its device, then moved,
    so a seeded draw is the same on any accelerator."""
    draw_device = generator.device if generator is not None else device
    return torch.randn(shape, generator=generator, device=draw_device).to(device)


def reparameterize(mu: Tensor, logvar: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """z = μ + exp(logvar / 2) · ε, ε ~ N(0, I)."""
    eps = standard_normal(tuple(mu.shape), mu.device, generator)
    return mu + torch.exp(0.5 * logvar) * eps


# ---------- architecture ----------


def hidden_block(n_in: int, n_out: int) -> nn.Sequential:
    """Flexynesis's hidden layer: Linear -> LeakyReLU(0.2) -> BatchNorm."""
    return nn.Sequential(nn.Linear(n_in, n_out), nn.LeakyReLU(0.2), nn.BatchNorm1d(n_out))


def latent_head(n_in: int, latent_dim: int, heads: Heads) -> nn.Module:
    """A mean or log-variance head: linear, or Tybalt's Dense -> BatchNorm -> ReLU."""
    if heads == "linear":
        return nn.Linear(n_in, latent_dim)
    if heads == "bn_relu":
        return nn.Sequential(nn.Linear(n_in, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU())
    raise ValueError(f"unknown heads {heads!r}; expected one of {HEADS}")


class Encoder(nn.Module):
    """genes -> [hidden] -> (μ, logvar). `hidden_dim=0` connects the heads to the genes."""

    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: int = 0, heads: Heads = "linear"):
        super().__init__()
        self.hidden = hidden_block(n_genes, hidden_dim) if hidden_dim else nn.Identity()
        width = hidden_dim or n_genes
        self.mean = latent_head(width, latent_dim, heads)
        self.logvar = latent_head(width, latent_dim, heads)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.hidden(x)
        return self.mean(h), self.logvar(h)


class Decoder(nn.Module):
    """latent -> [hidden] -> genes, the encoder mirrored. Returns the pre-activation output."""

    def __init__(self, latent_dim: int, n_genes: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                hidden_block(latent_dim, hidden_dim), nn.Linear(hidden_dim, n_genes)
            )
        else:
            self.net = nn.Linear(latent_dim, n_genes)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class ClassifierHead(nn.Module):
    """Flexynesis's MLP: Linear -> BatchNorm -> ReLU -> Dropout -> Linear, to class logits."""

    def __init__(
        self, latent_dim: int, n_classes: int, hidden_dim: int = 32, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    """Encoder, reparameterized sample, decoder, and an optional classifier on the sample."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int = 0,
        heads: Heads = "linear",
        n_classes: int = 0,
        class_hidden: int = 32,
        class_dropout: float = 0.1,
        glorot_init: bool = False,
        logvar_max: float | None = None,
    ) -> None:
        """`logvar_max` caps the log-variance before it is exponentiated.

        Without a KL term nothing holds the log-variance down: under the MMD
        and a decoder that BatchNorm makes blind to one sample's scale, it
        drifts past 100 within a few dozen steps and exp(logvar / 2)
        overflows. A KL term's equilibrium for an uninformative dimension is
        logvar = 0, the prior's width.
        """
        super().__init__()
        self.logvar_max = logvar_max
        self.encoder = Encoder(n_genes, latent_dim, hidden_dim, heads)
        self.decoder = Decoder(latent_dim, n_genes, hidden_dim)
        self.classifier = (
            ClassifierHead(latent_dim, n_classes, class_hidden, class_dropout)
            if n_classes
            else None
        )
        if glorot_init:
            # Keras's Dense default, which Tybalt kept: Glorot-uniform weights, zero biases.
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """`mu`, `logvar`, the sample `z`, the decoder `output` and, if supervised, `logits`."""
        mu, logvar = self.encoder(x)
        if self.logvar_max is not None:
            logvar = logvar.clamp(max=self.logvar_max)
        z = reparameterize(mu, logvar, generator)
        out = {"mu": mu, "logvar": logvar, "z": z, "output": self.decoder(z)}
        if self.classifier is not None:
            out["logits"] = self.classifier(z)
        return out
