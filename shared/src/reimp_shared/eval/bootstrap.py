"""Confidence intervals by resampling patients.

A replicate draws an evaluation set's patients with replacement — all of a
patient's rows together, as the split keeps them — and rescores. Rather
than materializing replicates, each is a weight per row: how many times
its patient was drawn. Metrics that are sums over rows, or over pairs of
rows, then score every replicate at once with a matrix product.

`bootstrap_weights` puts the observed data (all weights 1) in row 0 and
the replicates after it, so every metric function returns its estimate
and its replicates in one array, and `summarize` splits them apart. With
the same seed and the same patients, two models' replicates are the same
draws, so their intervals are paired.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

N_BOOTSTRAP = 1000
LEVEL = 0.95
# Suffix of the column holding a score's replicates.
BOOT = "_boot"


def bootstrap_weights(
    cases: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0
) -> np.ndarray:
    """`(1 + n_bootstrap, n_rows)` weights: all ones, then one row per replicate.

    A replicate draws as many cases as there are, with replacement; a row's
    weight is how many times its case was drawn.
    """
    unique, inverse = np.unique(np.asarray(cases), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=n_bootstrap)
    return np.vstack([np.ones(len(inverse)), draws[:, inverse]]).astype(np.float64)


def weighted_mean(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of per-row `values` under each row of `weights`, ignoring NaN rows."""
    valid = ~np.isnan(values)
    return (weights @ np.where(valid, values, 0.0)) / (weights @ valid)


def interval(replicates: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    """Percentile interval of the replicates."""
    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(replicates, [tail, 100 - tail])
    return float(low), float(high)


def summarize(scores: Mapping[str, np.ndarray], level: float = LEVEL) -> dict:
    """`name` from each array's first entry, `name_lo` / `name_hi` from the rest.

    The replicates themselves are kept as `name_boot`, so two embeddings
    scored on the same patients can be differenced replicate by replicate
    (see `reimp_shared.eval.compare`).
    """
    out = {}
    for name, values in scores.items():
        values = np.atleast_1d(values)
        out[name] = float(values[0])
        if len(values) > 1:
            out[f"{name}_lo"], out[f"{name}_hi"] = interval(values[1:], level)
            out[f"{name}{BOOT}"] = np.asarray(values[1:], dtype=np.float64)
    return out
