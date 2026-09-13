"""Validation reconstruction metrics, split by what the encoder saw.

The training loss averages over all G genes, which mixes two different
things: the K genes the encoder was given (their values are in the tokens,
so reconstructing them is close to a copy) and the G - K masked genes (the
actual prediction problem). These metrics report the two separately, plus
per-sample Pearson and R² over the masked genes.
"""

from __future__ import annotations

import torch
from torch import Tensor

from reimp_txfm.model import poisson_loss


def holdout_metrics(x_hat: Tensor, target: Tensor, unmasked_idx: Tensor) -> dict[str, Tensor]:
    """Scalars `loss_visible`, `loss_holdout`, `pearson_holdout`, `r2_holdout`.

    `x_hat` and `target` are (B, G) in log1p space; `unmasked_idx` is the
    (B, K) positions the encoder saw. Pearson and R² are per sample over its
    masked genes, then averaged over samples where they are defined.
    """
    visible = torch.zeros_like(target, dtype=torch.bool).scatter_(1, unmasked_idx, True)
    per_gene = poisson_loss(x_hat, target)
    holdout = (~visible).to(target.dtype)
    n = holdout.sum(dim=1)

    dx = (x_hat - (x_hat * holdout).sum(dim=1, keepdim=True) / n[:, None]) * holdout
    dt = (target - (target * holdout).sum(dim=1, keepdim=True) / n[:, None]) * holdout
    ss_x = dx.pow(2).sum(dim=1)
    ss_t = dt.pow(2).sum(dim=1)
    pearson = (dx * dt).sum(dim=1) / (ss_x * ss_t).sqrt()
    r2 = 1 - ((x_hat - target).pow(2) * holdout).sum(dim=1) / ss_t
    # Zero variance leaves a sample's correlation undefined (inf or NaN).
    pearson = pearson.where(ss_x * ss_t > 0, torch.nan)
    r2 = r2.where(ss_t > 0, torch.nan)

    return {
        "loss_visible": per_gene[visible].mean(),
        "loss_holdout": per_gene[~visible].mean(),
        "pearson_holdout": pearson.nanmean(),
        "r2_holdout": r2.nanmean(),
    }
