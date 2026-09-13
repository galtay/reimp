"""The shared MLP encoder, each objective's pretext head, the input scaler and the losses.

  encoder     4 × [Linear 256 → BatchNorm → ReLU → Dropout 0.2]; the
              embedding is the 256-d output of the last block (paper §2.1,
              Table S4).
  SCARF       projection head Linear-BN-ReLU-Linear, 256 → 256; NT-Xent
              with cosine similarity, symmetric over both views.
  VIME        mask and feature decoders, each 4 × [Linear 256 → ReLU] then
              Linear G; BCE on the mask + α · MSE on the features.
  BYOL        projector and predictor Linear 4096 → BN → ReLU → Linear 256;
              2 − 2·cos between online prediction and EMA target,
              symmetric over both views.

`GeneScaler` z-scores each gene by a mean and standard deviation fit on
training samples; it is a module, so its statistics are saved with the
model and embed-time inputs are scaled exactly as training inputs were.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GeneScaler(nn.Module):
    """Per-gene z-score: (x − mean) / std, both fit on training samples.

    A gene constant over the training samples gets std 1, so it scales to
    x − mean rather than dividing by zero.
    """

    def __init__(self, n_genes: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_genes))
        self.register_buffer("std", torch.ones(n_genes))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, values) -> GeneScaler:
        """Fit on `values`, (samples, genes) — the training samples only."""
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean):
            raise ValueError(f"expected (n, {len(self.mean)}) values, got {x.shape}")
        if len(x) < 2:
            raise ValueError("fitting a scaler needs at least two samples")
        std = x.std(axis=0)
        std[std == 0] = 1.0
        self.mean.copy_(torch.from_numpy(x.mean(axis=0)))
        self.std.copy_(torch.from_numpy(std))
        self.fitted.fill_(True)
        return self

    def forward(self, x: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("the scaler is not fit: fit it on the training samples first")
        return (x - self.mean) / self.std


def mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Linear → BatchNorm → ReLU → Dropout, the encoder's unit."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout)
    )


class MLPEncoder(nn.Module):
    """`n_layers` blocks of `mlp_block`; the last block's output is the embedding."""

    def __init__(
        self, n_genes: int, hidden_dim: int = 256, n_layers: int = 4, dropout: float = 0.2
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        dims = [n_genes] + [hidden_dim] * n_layers
        self.blocks = nn.Sequential(
            *(mlp_block(a, b, dropout) for a, b in zip(dims[:-1], dims[1:], strict=True))
        )
        self.out_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """(B, G) scaled values -> (B, hidden_dim) embedding."""
        return self.blocks(x)


class ProjectionHead(nn.Module):
    """Linear → BatchNorm → ReLU → Linear: SCARF's head, BYOL's projector and predictor."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """VIME's decoders: `n_layers` × [Linear `hidden_dim` → ReLU], then Linear to G.

    Returns the raw output: logits for the mask decoder, z-scored values
    for the feature decoder.
    """

    def __init__(self, in_dim: int, n_genes: int, hidden_dim: int = 256, n_layers: int = 4) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], n_genes))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def nt_xent(z1: Tensor, z2: Tensor, temperature: float = 1.0) -> Tensor:
    """Symmetric NT-Xent (SimCLR) over 2N views: row i's positive is its other view.

    Cosine similarities divided by `temperature`; each of the 2N views is
    scored against the 2N − 1 others (its positive and 2N − 2 negatives),
    and the cross-entropies are averaged — both directions of the paper's
    Eq. 1, as its code computes.
    """
    n = len(z1)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    logits = z @ z.T / temperature
    self_pairs = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_pairs, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(logits, targets)


def vime_losses(
    mask_logits: Tensor, features: Tensor, mask: Tensor, target: Tensor
) -> tuple[Tensor, Tensor]:
    """VIME's two terms: BCE of the mask estimate, and MSE of the reconstruction.

    `mask` is 1 where the corrupted input differs from `target`, the clean
    values. The reconstruction is scored on every gene, corrupted or not,
    as VIME's and the paper's code do. The objective is `bce + α · mse`.
    """
    bce = F.binary_cross_entropy_with_logits(mask_logits, mask.to(mask_logits.dtype))
    return bce, F.mse_loss(features, target)


def byol_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Per sample, 2 − 2·cos(prediction, target): the squared distance of the unit vectors.

    No gradient reaches `target`.
    """
    return 2 - 2 * F.cosine_similarity(prediction, target.detach(), dim=-1)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float) -> None:
    """target ← decay · target + (1 − decay) · online, parameter by parameter.

    Buffers (BatchNorm running statistics) are left to each network's own
    forward passes.
    """
    for t, o in zip(target.parameters(), online.parameters(), strict=True):
        t.lerp_(o, 1 - decay)
