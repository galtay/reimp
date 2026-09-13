"""Embeddings that need no model, for every score to be read against."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_embeddings(
    values: np.ndarray,
    train_rows: np.ndarray,
    n_components: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """PCA fit on `train_rows` only, applied to every row."""
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(values[train_rows])
    return pca.transform(values).astype(np.float32)


def hvg_embeddings(values: np.ndarray, train_rows: np.ndarray, n_genes: int = 5000) -> np.ndarray:
    """The `n_genes` genes most variable over `train_rows`, each standardized on those rows.

    Expression itself, cut to its most variable genes — Gross et al. 2024's
    "Identity" representation, their strongest per-cohort survival
    baseline. Beside PCA it asks what compressing the expression buys.
    Genes keep their dataset order; genes constant over `train_rows` are
    never chosen.
    """
    train = values[train_rows]
    variance = train.var(axis=0, dtype=np.float64)
    varying = np.flatnonzero(variance > 0)
    top = np.sort(varying[np.argsort(-variance[varying], kind="stable")[:n_genes]])
    mean = train[:, top].mean(axis=0, dtype=np.float64)
    sd = train[:, top].std(axis=0, dtype=np.float64)
    return ((values[:, top] - mean) / sd).astype(np.float32)
