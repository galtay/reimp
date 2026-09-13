"""Triplets for COMPASS's contrastive pretraining: augmented views and negatives.

The anchor and the positive are two augmented views of one sample; the
negative is an augmented view of a sample from another patient. Each view
is left as it is with probability `no_augment_prob`, and otherwise gets one
of two augmentations, with equal odds — COMPASS's `MaskJitterAugmentor`:

  mask    each gene set to 0, its training minimum in min-max units, with
          probability `mask_prob`
  jitter  Gaussian noise of standard deviation `jitter_std` on every gene

A `NegativeSampler` draws each anchor's negative from one split's samples,
never the anchor's own patient:

  any           any other patient's sample: COMPASS's released code (K = 1)
  same_project  another patient in the anchor's project: the paper's
                description, a variant here since it reads a label
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

Negatives = Literal["any", "same_project"]
NEGATIVES: tuple[str, ...] = ("any", "same_project")


def augment(
    x: Tensor,
    mask_prob: float,
    jitter_std: float,
    no_augment_prob: float = 0.1,
    generator: torch.Generator | None = None,
) -> Tensor:
    """One augmented view of each row of `x` (samples x genes, min-max scaled).

    With a `generator` the draws run on the generator's device and are then
    moved to `x`'s, so a seeded view is the same on any accelerator.
    """
    device = generator.device if generator is not None else x.device
    n, g = x.shape
    kind = torch.rand(n, 1, generator=generator, device=device)
    split = no_augment_prob + (1 - no_augment_prob) / 2
    masked = (kind >= no_augment_prob) & (kind < split)
    jittered = kind >= split
    drop = torch.rand(n, g, generator=generator, device=device) < mask_prob
    noise = torch.randn(n, g, generator=generator, device=device) * jitter_std
    x = x.masked_fill((drop & masked).to(x.device), 0.0)
    return x + (noise * jittered).to(x.device)


class NegativeSampler:
    """For each anchor row, a row of another patient from a pool of rows.

    `cases` and `projects` label every row of the data; `pool` holds the
    rows to draw from (one split's). With `same_project` a negative comes
    from the anchor's project, unless the pool has a single patient there,
    in which case it comes from the whole pool.
    """

    def __init__(
        self,
        cases: Sequence[str],
        projects: Sequence[str],
        pool: np.ndarray,
        negatives: Negatives = "any",
    ) -> None:
        if negatives not in NEGATIVES:
            raise ValueError(f"unknown negatives {negatives!r}; expected one of {NEGATIVES}")
        case = pd.factorize(np.asarray(cases))[0]
        project = pd.factorize(np.asarray(projects))[0]
        pool = np.asarray(pool, dtype=np.int64)
        if len(np.unique(case[pool])) < 2:
            raise ValueError("negatives need at least two patients in the pool")

        # Candidate rows as consecutive slices: the whole pool first, then,
        # for same_project, one slice per project; each row draws from one.
        slices = [pool]
        start = np.zeros(len(case), dtype=np.int64)
        count = np.full(len(case), len(pool), dtype=np.int64)
        if negatives == "same_project":
            offset = len(pool)
            for p in np.unique(project[pool]):
                rows = pool[project[pool] == p]
                if len(np.unique(case[rows])) < 2:
                    continue
                slices.append(rows)
                start[project == p], count[project == p] = offset, len(rows)
                offset += len(rows)
        self.candidates = torch.from_numpy(np.concatenate(slices))
        self.start, self.count = torch.from_numpy(start), torch.from_numpy(count)
        self.case = torch.from_numpy(case)

    def draw(
        self, anchors: Tensor, generator: torch.Generator | None = None, max_tries: int = 100
    ) -> Tensor:
        """One negative row per anchor row (CPU, int64)."""
        anchors = anchors.cpu()
        start, count = self.start[anchors], self.count[anchors]
        negatives = torch.empty_like(anchors)
        todo = torch.arange(len(anchors))
        for _ in range(max_tries):
            u = torch.rand(len(todo), generator=generator)
            pick = start[todo] + (u * count[todo]).long().clamp_max(count[todo] - 1)
            negatives[todo] = self.candidates[pick]
            todo = todo[self.case[negatives[todo]] == self.case[anchors[todo]]]
            if len(todo) == 0:
                return negatives
        raise RuntimeError(f"no negative of another patient in {max_tries} draws")
