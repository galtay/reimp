"""Per-gene selection and scaling, fit on a fold's training samples.

Both models scale each gene before training, and Tybalt first keeps its
5,000 most variable genes. `GeneScaler` fits all of it on the rows it is
given — the fold's training rows — and applies it to every row, so a val or
test sample is scaled by statistics it did not help set (shared/EVALS.md,
rule 3):

  genes    the `n_genes` genes with the largest median absolute deviation,
           median |x − median(x)|, as Tybalt's paper and BioBombe describe
           it (Tybalt's code used pandas' *mean* absolute deviation); all
           genes when `n_genes` is None
  minmax   (x − min) / (max − min), Tybalt's MinMaxScaler. Every row is then
           clipped to [0, 1], which only moves val and test values outside
           the training range: a sigmoid output and BCE cannot represent them.
  zscore   (x − mean) / sd, the MMD-AE's StandardScaler. Every row is then
           clipped to the gene's range over the training rows, which again
           only moves val and test values: a gene expressed in a handful of
           training samples has a near-zero SD, and puts a held-out sample
           that expresses it far beyond anything the model saw (fold 0: |z|
           up to 238 in val and test, 91 in training).
  none     values as given

A gene constant over the training rows gets scale 1, as in scikit-learn's
scalers, so it maps to 0 in every row.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

Scaling = Literal["minmax", "zscore", "none"]
SCALINGS: tuple[str, ...] = ("minmax", "zscore", "none")


def median_absolute_deviation(values: np.ndarray) -> np.ndarray:
    """Per-gene median |x − median(x)| over the rows of `values` (samples x genes)."""
    values = np.asarray(values)
    return np.median(np.abs(values - np.median(values, axis=0)), axis=0)


def top_genes_by_mad(values: np.ndarray, n_genes: int) -> np.ndarray:
    """Column positions of the `n_genes` genes with the largest MAD, in column order.

    Always exactly min(`n_genes`, columns) genes, so a model's width is
    known before the fit; ties, including genes with no deviation at all,
    go to the earlier column.
    """
    mad = median_absolute_deviation(values)
    return np.sort(np.argsort(-mad, kind="stable")[:n_genes])


class GeneScaler:
    """Top genes by MAD, then a per-gene affine scaling; `fit` on training rows only."""

    def __init__(self, scaling: Scaling = "minmax", n_genes: int | None = None) -> None:
        if scaling not in SCALINGS:
            raise ValueError(f"unknown scaling {scaling!r}; expected one of {SCALINGS}")
        if n_genes is not None and n_genes < 1:
            raise ValueError(f"n_genes must be positive or None, got {n_genes}")
        self.scaling = scaling
        self.n_genes = n_genes
        self.genes_: np.ndarray | None = None
        self.offset_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.low_: np.ndarray | None = None  # zscore: the training range, scaled
        self.high_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> GeneScaler:
        """Choose genes and measure their scaling from `values` (samples x genes)."""
        values = np.asarray(values)
        n_columns = values.shape[1]
        if self.n_genes is None or self.n_genes >= n_columns:
            self.genes_ = np.arange(n_columns)
        else:
            self.genes_ = top_genes_by_mad(values, self.n_genes)
        x = values[:, self.genes_]
        if self.scaling == "minmax":
            offset = x.min(axis=0).astype(np.float64)
            scale = x.max(axis=0).astype(np.float64) - offset
        elif self.scaling == "zscore":
            offset = x.mean(axis=0, dtype=np.float64)
            scale = x.std(axis=0, dtype=np.float64)
        else:
            offset, scale = np.zeros(len(self.genes_)), np.ones(len(self.genes_))
        self.offset_ = offset
        self.scale_ = np.where(scale > 0, scale, 1.0)
        self.low_ = self.high_ = None
        if self.scaling == "zscore":
            # Scaled exactly as `transform` scales, so no training value is clipped.
            self.low_, self.high_ = self._affine(np.stack([x.min(axis=0), x.max(axis=0)]))
        return self

    def _affine(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x -= self.offset_.astype(np.float32)
        x /= self.scale_.astype(np.float32)
        return x

    def transform(self, values: np.ndarray) -> np.ndarray:
        """The chosen genes of `values`, scaled and clipped, as float32."""
        if self.genes_ is None:
            raise RuntimeError("call fit before transform")
        x = self._affine(np.asarray(values)[:, self.genes_])
        if self.scaling == "minmax":
            np.clip(x, 0.0, 1.0, out=x)
        elif self.low_ is not None:
            np.clip(x, self.low_, self.high_, out=x)
        return x

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The fitted statistics as tensors, for a checkpoint; empty before `fit`."""
        if self.genes_ is None:
            return {}
        state = {
            "genes": torch.from_numpy(self.genes_.astype(np.int64)),
            "offset": torch.from_numpy(self.offset_),
            "scale": torch.from_numpy(self.scale_),
        }
        if self.low_ is not None:
            state["low"], state["high"] = torch.from_numpy(self.low_), torch.from_numpy(self.high_)
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore what `state_dict` saved; an empty state leaves the scaler unfit."""
        if not state:
            return
        self.genes_ = state["genes"].numpy()
        self.offset_ = state["offset"].numpy()
        self.scale_ = state["scale"].numpy()
        # A checkpoint from before the zscore clip has no range, and is not clipped.
        self.low_ = state["low"].numpy() if "low" in state else None
        self.high_ = state["high"].numpy() if "high" in state else None
