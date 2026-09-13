"""Binned expression tokens and masked-token corruption, for masked language models.

BulkRNABert tokenizes each gene's value on its own: log TPM divided by one
cohort-wide maximum, cut into `n_bins` equal-width bins. MOJO reads the
same tokens with another backbone. A `BinTokenizer` measures the maximum
on the samples it is fit on — the training samples — so a test sample is
binned by a statistic it did not help set; values above it land in the
top bin.

Bin 0 holds exact zeros only; bins 1 … n_bins − 1 split (0, max] evenly.
Pass the values as the model reads them (BulkRNABert: `tpm_unstranded`
with `transform="log1p"`). Natural-log `log1p` gives the same bins as
the paper's log10, since the tokenizer divides by a maximum in the same
units.

`mask_tokens` is BERT's corruption: a fraction of positions is selected,
and of those 80% become the mask token, 10% a random bin and 10% stay.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

DEFAULT_N_BINS = 64
IGNORE_INDEX = -100  # torch's cross-entropy default


class BinTokenizer:
    """One maximum fit on training samples; `tokens` bins every value by it."""

    def __init__(self, n_bins: int = DEFAULT_N_BINS) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")
        self.n_bins = n_bins
        self.max_: float | None = None

    def fit(self, values) -> BinTokenizer:
        """Measure the maximum of `values` (samples x genes, non-negative)."""
        top = float(np.max(np.asarray(values)))
        if not top > 0:
            raise ValueError("cannot fit a tokenizer on values with no positive entry")
        self.max_ = top
        return self

    def tokens(self, values):
        """Bin of each value, int64: 0 for exact zeros, else 1 … n_bins − 1.

        Takes a numpy array or a torch tensor and returns the same kind.
        """
        if self.max_ is None:
            raise RuntimeError("call fit before tokenizing")
        top = self.n_bins - 1
        if isinstance(values, Tensor):
            scaled = values.to(torch.float32) / self.max_
            return torch.ceil(scaled * top).clamp(0, top).to(torch.long)
        scaled = np.asarray(values, dtype=np.float32) / self.max_
        return np.clip(np.ceil(scaled * top), 0, top).astype(np.int64)


def mask_tokens(
    tokens: Tensor,
    mask_id: int,
    n_bins: int,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """BERT-style corruption of a (batch, genes) token tensor.

    Selects each position with probability `mask_prob`; a selected position
    becomes `mask_id` with probability 0.8, a uniform random bin in
    [0, n_bins) with 0.1, and keeps its token with 0.1. Returns the
    corrupted inputs and the labels: the original token at selected
    positions, `IGNORE_INDEX` everywhere else.
    """
    draw = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    selected = draw < mask_prob
    # Reuse the draw for the 80/10/10 split: within the selected positions
    # draw / mask_prob is uniform on [0, 1).
    within = draw / mask_prob
    to_mask = selected & (within < 0.8)
    to_random = selected & (within >= 0.8) & (within < 0.9)

    inputs = tokens.clone()
    inputs[to_mask] = mask_id
    random_bins = torch.randint(
        n_bins, tokens.shape, generator=generator, device=tokens.device, dtype=tokens.dtype
    )
    inputs[to_random] = random_bins[to_random]
    labels = torch.where(selected, tokens, torch.full_like(tokens, IGNORE_INDEX))
    return inputs, labels
