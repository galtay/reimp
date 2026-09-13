"""A sample as a sentence: its genes, ranked, cut into overlapping windows.

  rank     a fitted `reimp_shared.ranking.GeneRanker` orders each sample's
           genes, highest score first. A gene's token is its identity — its
           position in the loaded gene selection — and its rank is where it
           stands in the sentence; no expression value is kept.
  keep     only genes the sample expresses (value > 0) that the ranker can
           weigh (weight > 0: seen expressed in training). A zero has no
           rank among the others, and a gene never expressed in training has
           no rarity to score. At most `max_genes` genes, the top-ranked.
  windows  `window` tokens every `stride`, starting at 0, until a window
           reaches the end of the sentence: 1 + ceil((n − window) / stride)
           windows for n genes, the last padded. The paper's ~10,000 genes
           in 512-token windows at stride 256 make 39.

Sentences are padded with `pad_id` to a common length, and every function
here takes that padded (B, L) array and each row's length.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from reimp_shared.ranking import GeneRanker


def rank_genes(
    ranker: GeneRanker, values: np.ndarray, max_genes: int | None, pad_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """(B, G) values -> ranked gene positions (B, L) and each row's length (B,).

    L = min(`max_genes`, G). Rows are the kept genes in score order, then
    `pad_id`. Scores that can go negative (`z`) may rank an unexpressed
    gene above an expressed one; it is dropped all the same.
    """
    values = np.asarray(values)
    order = ranker.order(values)
    keep = (values > 0) & (ranker.weight_ > 0)
    # Move the kept genes to the front, each group still in score order.
    kept_first = np.argsort(~np.take_along_axis(keep, order, axis=1), axis=1, kind="stable")
    order = np.take_along_axis(order, kept_first, axis=1)
    lengths = keep.sum(axis=1)
    if max_genes is not None:
        order, lengths = order[:, :max_genes], np.minimum(lengths, max_genes)
    order[np.arange(order.shape[1]) >= lengths[:, None]] = pad_id
    return order.astype(np.int64), lengths.astype(np.int64)


def n_windows(lengths: Tensor, window: int, stride: int) -> Tensor:
    """Windows per sentence: 1 + ceil((n − window) / stride), at least 1; 0 for n = 0."""
    overhang = (lengths - window).clamp_min(0)
    return torch.where(lengths > 0, 1 + (overhang + stride - 1) // stride, 0)


def _gather(genes: Tensor, rows: Tensor, starts: Tensor, window: int, pad_id: int) -> Tensor:
    """`window` tokens of `genes[rows]` from each start; past the end is padding."""
    padded = F.pad(genes, (0, window), value=pad_id)
    columns = starts[:, None] + torch.arange(window, device=genes.device)
    return padded[rows[:, None], columns]


def sample_windows(
    genes: Tensor,
    lengths: Tensor,
    window: int,
    stride: int,
    pad_id: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """One window per sentence, uniform over its windows: (tokens (B', window), rows (B',)).

    Sentences with no genes have no window and are left out; `rows` says
    which sentence each window came from. With a `generator` the draw runs
    on the generator's device, so a seeded draw is the same on any
    accelerator.
    """
    counts = n_windows(lengths, window, stride)
    rows = torch.nonzero(counts > 0).squeeze(1)
    draw_device = generator.device if generator is not None else genes.device
    u = torch.rand(len(rows), generator=generator, device=draw_device).to(genes.device)
    starts = (u * counts[rows]).long().clamp_max(counts[rows] - 1) * stride
    return _gather(genes, rows, starts, window, pad_id), rows


def all_windows(
    genes: Tensor, lengths: Tensor, window: int, stride: int, pad_id: int
) -> tuple[Tensor, Tensor]:
    """Every window of every sentence, in order: (tokens (M, window), rows (M,))."""
    counts = n_windows(lengths, window, stride)
    rows = torch.repeat_interleave(torch.arange(len(genes), device=genes.device), counts)
    first = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    starts = (torch.arange(len(rows), device=genes.device) - first) * stride
    return _gather(genes, rows, starts, window, pad_id), rows
