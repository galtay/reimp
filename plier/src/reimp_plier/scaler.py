"""Per-gene z-scores with statistics from one fold's training samples.

`GeneScaler.fit` takes each gene's mean, SD (n - 1 denominator) and range
over the training samples. `varying` names the genes that are not constant
there, `subset` keeps some of them, and `transform` clips any sample to the
training range before z-scoring it, so no held-out z-score leaves the range
the training ones span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeneScaler:
    """Training statistics per gene: `mean`, `sd` (ddof = 1), and the range `low`..`high`."""

    mean: np.ndarray
    sd: np.ndarray
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> GeneScaler:
        """Statistics of `values`, samples x genes: the training rows only."""
        values = np.asarray(values)
        if len(values) < 2:
            raise ValueError("an SD needs at least two training samples")
        return cls(
            mean=values.mean(axis=0, dtype=np.float64),
            sd=values.std(axis=0, ddof=1, dtype=np.float64),
            low=values.min(axis=0),
            high=values.max(axis=0),
        )

    def varying(self) -> np.ndarray:
        """Indices of the genes whose training SD is not 0."""
        return np.flatnonzero(self.sd > 0)

    def subset(self, genes: np.ndarray) -> GeneScaler:
        return GeneScaler(self.mean[genes], self.sd[genes], self.low[genes], self.high[genes])

    def transform(self, values: np.ndarray) -> np.ndarray:
        """`values`, samples x genes, clipped to the training range and z-scored, in float64."""
        out = np.clip(values, self.low, self.high).astype(np.float64)
        out -= self.mean
        out /= self.sd
        return out
