"""Feature corruption: which entries of a (batch, genes) matrix to replace, and with what.

  SCARF  exactly ⌊rate · G⌋ genes per sample, uniform without replacement
         (`fixed_count_mask`), replaced by draws from Uniform[low_j, high_j],
         the gene's range over the training samples (`uniform_draws`), or —
         the original SCARF — from the gene's empirical marginal
         (`marginal_draws`).
  VIME   each entry independently with probability `rate`
         (`bernoulli_mask`), replaced by the same gene's value in a random
         training sample (`marginal_draws`). BYOL's second view is the same.

Every replacement comes from a `pool` of training samples, or from the
range it spans, so a validation or test sample never feeds another's
corruption. With a `generator` the draw runs on the generator's device and
is then moved to the data's, so a seeded corruption is the same on any
accelerator.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _rand(
    shape: tuple[int, ...], device: torch.device, generator: torch.Generator | None
) -> Tensor:
    draw_device = generator.device if generator is not None else device
    return torch.rand(shape, generator=generator, device=draw_device).to(device)


def fixed_count_mask(
    n_rows: int,
    n_genes: int,
    rate: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(n_rows, n_genes) bool: exactly ⌊rate · n_genes⌋ True per row, without replacement."""
    device = torch.device(device or "cpu")
    k = int(rate * n_genes)
    mask = torch.zeros(n_rows, n_genes, dtype=torch.bool, device=device)
    if k == 0:
        return mask
    idx = _rand((n_rows, n_genes), device, generator).topk(k, dim=1).indices
    return mask.scatter_(1, idx, True)


def bernoulli_mask(
    n_rows: int,
    n_genes: int,
    rate: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(n_rows, n_genes) bool: each entry True with probability `rate`, independently."""
    device = torch.device(device or "cpu")
    return _rand((n_rows, n_genes), device, generator) < rate


def uniform_draws(
    low: Tensor, high: Tensor, n_rows: int, generator: torch.Generator | None = None
) -> Tensor:
    """(n_rows, G): entry (i, j) drawn from Uniform[low_j, high_j]."""
    u = _rand((n_rows, len(low)), low.device, generator)
    return low + (high - low) * u


def marginal_draws(pool: Tensor, n_rows: int, generator: torch.Generator | None = None) -> Tensor:
    """(n_rows, G): entry (i, j) is gene j's value in a uniformly drawn row of `pool`.

    Every entry draws its own row, so this samples each gene's empirical
    marginal independently — VIME's column-wise permutation, without
    tying a draw to the batch it lands in.
    """
    n_pool, n_genes = pool.shape
    if n_pool == 0:
        raise RuntimeError("no training pool to draw replacements from: fit statistics first")
    draw_device = generator.device if generator is not None else pool.device
    rows = torch.randint(n_pool, (n_rows, n_genes), generator=generator, device=draw_device)
    return pool.gather(0, rows.to(pool.device))


def corrupt(values: Tensor, mask: Tensor, replacement: Tensor) -> Tensor:
    """`values` with the masked entries taken from `replacement`."""
    return torch.where(mask, replacement, values)
