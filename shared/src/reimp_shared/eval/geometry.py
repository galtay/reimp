"""Geometry of the embedding space, read against cancer type.

Linear probes ask whether cancer type can be read off an embedding. These
ask how the embedding is laid out — a model can pass a linear probe with a
nearly collapsed embedding, or fail one with a rich but curved one.

- `effective_rank`, `top_eigenvalue_share`: how many directions the
  embeddings use — exp of the entropy of the normalized covariance
  spectrum, and the largest eigenvalue's share of it (TifBERT, Table 4).
- `precision_at_k`: of a sample's k nearest training samples by cosine
  similarity, the fraction of the same cancer type (TifBERT; kNN in
  TxFM's benchmarks).
- `nmi`, `ari`: agreement between cancer type and k-means clusters, fit on
  training embeddings with one cluster per cancer type (TxFM's
  Kedzierska benchmark).
- `silhouette`: cosine silhouette of cancer types among the evaluated
  samples.

Samples and labels are the `project_id` classification task's: tumour
samples, cancer type. Embeddings are centred on the training mean first,
so cosine similarity is not dominated by a shared offset.

Each fold's model embeds into its own space: neighbours
come from the fold's own fitting samples and silhouettes from its own test
samples, and the per-sample scores are pooled; the spectrum, NMI and ARI
describe a whole space, so they are taken per fold and averaged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from reimp_shared.eval.bootstrap import (
    N_BOOTSTRAP,
    bootstrap_weights,
    summarize,
    weighted_mean,
)
from reimp_shared.eval.classification import task_labels
from reimp_shared.eval.protocol import plan
from reimp_shared.splits import CASE_KEY, DEFAULT_SALT

DEFAULT_KS = (1, 10)


def spectrum(x: np.ndarray) -> tuple[float, float]:
    """Effective rank and top-eigenvalue share of the covariance of `x`."""
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    share = singular**2 / (singular**2).sum()
    share = share[share > 0]
    return float(np.exp(-(share * np.log(share)).sum())), float(share.max())


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def neighbour_precision(
    query: np.ndarray,
    reference: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
) -> np.ndarray:
    """`(n_query, len(ks))`: the share of each query's k nearest references sharing its label.

    Nearness is cosine similarity.
    """
    similarity = _unit(query) @ _unit(reference).T
    k_max = max(ks)
    top = np.argpartition(-similarity, k_max - 1, axis=1)[:, :k_max]
    ranked = np.argsort(-np.take_along_axis(similarity, top, axis=1), axis=1, kind="stable")
    nearest = np.take_along_axis(top, ranked, axis=1)
    hits = np.asarray(reference_labels)[nearest] == np.asarray(query_labels)[:, None]
    return np.stack([hits[:, :k].mean(axis=1) for k in ks], axis=1)


def clustering_agreement(
    labels: np.ndarray, clusters: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """Adjusted Rand index and normalized mutual information (arithmetic mean).

    Both come from the labels x clusters contingency table, so under
    `weights` (replicates x rows) each replicate's table is one matrix
    product. Without `weights`, floats matching scikit-learn.
    """
    _, label_codes = np.unique(labels, return_inverse=True)
    _, cluster_codes = np.unique(clusters, return_inverse=True)
    n_labels, n_clusters = label_codes.max() + 1, cluster_codes.max() + 1
    n = len(label_codes)
    w = np.ones((1, n)) if weights is None else np.atleast_2d(weights)
    cells = np.zeros((n, n_labels * n_clusters))
    cells[np.arange(n), label_codes * n_clusters + cluster_codes] = 1.0
    table = (w @ cells).reshape(-1, n_labels, n_clusters)
    total = table.sum(axis=(1, 2))
    rows, cols = table.sum(axis=2), table.sum(axis=1)

    def pairs(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2

    agree = pairs(table).sum(axis=(1, 2))
    row_pairs, col_pairs = pairs(rows).sum(axis=1), pairs(cols).sum(axis=1)
    expected = row_pairs * col_pairs / pairs(total)
    ari = (agree - expected) / (0.5 * (row_pairs + col_pairs) - expected)

    def entropy(p: np.ndarray) -> np.ndarray:
        return -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=-1)

    joint = table / total[:, None, None]
    p_rows, p_cols = rows / total[:, None], cols / total[:, None]
    outer = p_rows[:, :, None] * p_cols[:, None, :]
    ratio = np.where(joint > 0, joint / np.where(outer > 0, outer, 1.0), 1.0)
    mutual = (joint * np.log(ratio)).sum(axis=(1, 2))
    nmi = mutual / ((entropy(p_rows) + entropy(p_cols)) / 2)

    scores = {"nmi": nmi, "ari": ari}
    return scores if weights is not None else {name: float(v[0]) for name, v in scores.items()}


def geometry_probe(
    sample_index: np.ndarray,
    embeddings: np.ndarray,
    samples: pd.DataFrame,
    ks: Sequence[int] = DEFAULT_KS,
    *,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold geometry scores; precision and silhouette with patient-bootstrap intervals.

    No interval for the spectrum scores, which describe the evaluated
    embeddings, nor for NMI and ARI: a resample's duplicated patients form
    pairs that agree by construction, which inflates both — on the PCA
    baseline, random clusters' NMI rises from 0.14 to a replicate median of
    0.22 — so their percentile interval would miss the estimate.
    Silhouettes are computed once per split and resampled, so their
    interval ignores that each silhouette depends on the others.
    """
    cv = plan(samples, sample_index, fold, salt)
    rows = cv.rows
    labels = task_labels(rows, "project_id")
    labelled = labels.notna().to_numpy()
    y = labels.to_numpy()
    cases = rows[CASE_KEY].to_numpy()

    precision = np.full((len(rows), len(ks)), np.nan)
    silhouette = np.full(len(rows), np.nan)
    whole_space = []
    for f in cv.fits:
        known, y_fold, raw = labelled[f.rows], y[f.rows], embeddings[f.rows]
        train = known & f.fit
        test = known & f.test
        if not test.any():
            continue
        x = raw - raw[train].mean(axis=0)
        unit = _unit(x)
        kmeans = KMeans(len(np.unique(y_fold[train])), n_init="auto", random_state=seed)
        kmeans.fit(unit[train])
        at = f.rows[test]
        precision[at] = neighbour_precision(x[test], x[train], y_fold[test], y_fold[train], ks)
        if 1 < len(np.unique(y_fold[test])) < test.sum():
            silhouette[at] = silhouette_samples(x[test], y_fold[test], metric="cosine")
        rank, top_share = spectrum(raw[test])
        whole_space.append(
            {
                "effective_rank": rank,
                "top_eigenvalue_share": top_share,
                **clustering_agreement(y_fold[test], kmeans.predict(unit[test])),
            }
        )

    mask = labelled & cv.tested
    if not mask.any():
        return pd.DataFrame()
    weights = bootstrap_weights(cases[mask], n_bootstrap, seed)
    scores = {
        f"precision_at_{k}": weighted_mean(weights, precision[mask, j]) for j, k in enumerate(ks)
    }
    if not np.isnan(silhouette[mask]).all():
        scores["silhouette"] = weighted_mean(weights, silhouette[mask])
    space = pd.DataFrame(whole_space).mean()
    return pd.DataFrame(
        [
            {
                "split": cv.name,
                "n": int(mask.sum()),
                "effective_rank": space["effective_rank"],
                "top_eigenvalue_share": space["top_eigenvalue_share"],
                **summarize(scores),
                "nmi": space["nmi"],
                "ari": space["ari"],
            }
        ]
    )
