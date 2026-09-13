"""Invertibility: how much expression a linear map recovers from an embedding."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reimp_shared.data import ExpressionData, load_expression
from reimp_shared.eval.bootstrap import N_BOOTSTRAP, bootstrap_weights, summarize
from reimp_shared.eval.protocol import plan
from reimp_shared.preprocess import DEFAULT_LIBRARY_SIZE
from reimp_shared.splits import CASE_KEY, DEFAULT_SALT

# The expression every model is asked to recover, whatever it was trained on.
INVERTIBILITY_TARGET = {
    "quantification": "unstranded",
    "gene_types": ("protein_coding",),
    "transform": "lognorm",
    "library_size": DEFAULT_LIBRARY_SIZE,
}
DEFAULT_ALPHAS = tuple(np.logspace(-2, 4, 7))
# Rows per block when accumulating scores: a pooled cross-validation set is
# every sample, too many to hold as float64 samples x genes several times.
BLOCK_ROWS = 1024


def invertibility_target(revision: str | None = None) -> ExpressionData:
    """The fixed target: log-normalized protein-coding counts for every sample."""
    return load_expression(**INVERTIBILITY_TARGET, revision=revision)


class ReconstructionSums:
    """The weighted sums behind `reconstruction_scores`, accumulated over rows.

    Every score is a ratio of sums over rows, so an eval set can arrive in
    pieces — a cross-validation fold at a time, each with its own fit and
    its own baseline profile — and still score as one. `shift` is any
    per-gene constant near the genes' means: taking it off before squaring
    keeps SS_tot = Σw·y² − W·ȳ² free of cancellation.
    """

    def __init__(self, n_replicates: int, shift: np.ndarray) -> None:
        self.shift = np.asarray(shift, dtype=np.float64)
        shape = (n_replicates, len(self.shift))
        self.total = np.zeros((n_replicates, 1))
        self.ss_res = np.zeros(shape)
        self.first = np.zeros(shape)  # Σ w·(y − shift)
        self.second = np.zeros(shape)  # Σ w·(y − shift)²
        # Per score, the weighted sum of per-sample Pearsons and of weights.
        self.pearson = {
            name: np.zeros((2, n_replicates))
            for name in ("pearson_sample", "pearson_sample_baseline")
        }

    def add(
        self, true: np.ndarray, pred: np.ndarray, baseline: np.ndarray, weights: np.ndarray
    ) -> None:
        """Rows of `true` and `pred` (samples x genes), `weights` (replicates x rows)."""
        weights = np.atleast_2d(weights)
        for start in range(0, len(true), BLOCK_ROWS):
            block = slice(start, start + BLOCK_ROWS)
            w = weights[:, block]
            t = np.asarray(true[block], dtype=np.float64)
            p = np.asarray(pred[block], dtype=np.float64)
            self.total += w.sum(axis=1, keepdims=True)
            self.ss_res += w @ (t - p) ** 2
            centred = t - self.shift
            self.first += w @ centred
            self.second += w @ centred**2
            constant = np.broadcast_to(baseline, t.shape)
            for name, other in [("pearson_sample", p), ("pearson_sample_baseline", constant)]:
                r = _row_pearson(t, other)
                valid = ~np.isnan(r)
                self.pearson[name][0] += w @ np.where(valid, r, 0.0)
                self.pearson[name][1] += w @ valid

    def scores(self) -> dict[str, np.ndarray]:
        """Arrays over replicates, as in `reconstruction_scores`."""
        mean = self.first / self.total
        ss_tot = np.maximum(self.second - self.total * mean**2, 0.0)
        varying = ss_tot > 1e-9 * self.total
        r2_gene = np.where(varying, 1 - self.ss_res / np.where(varying, ss_tot, 1.0), np.nan)
        return {
            "r2_pooled": 1 - self.ss_res.sum(axis=1) / ss_tot.sum(axis=1),
            "r2_gene": np.nanmean(r2_gene, axis=1),
            **{name: total / count for name, (total, count) in self.pearson.items()},
        }


def reconstruction_scores(
    true: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Scores of `pred` against `true`, both samples x genes.

    - `r2_pooled`   1 − SS_res / SS_tot with both sums over every gene, so
                    each gene counts in proportion to its variance.
    - `r2_gene`     R² per gene, averaged with every gene counting the same;
                    genes constant across these samples are left out.
    - `pearson_sample`           Pearson over genes per sample, averaged.
    - `pearson_sample_baseline`  the same for `baseline`, one profile
                    predicted for every sample.

    Both R²s are 0 for predicting each gene's mean over these samples, so
    they measure sample-to-sample variation directly. Per-sample Pearson
    does not: genes differ in mean expression far more than samples differ,
    so a constant profile already scores high, and the gap to
    `pearson_sample_baseline` is what the embedding adds.

    Without `weights`, floats. With `weights` (replicates x rows), arrays
    over replicates; every sum becomes a matrix product with the weights.
    """
    true = np.asarray(true, dtype=np.float64)
    w = np.ones((1, len(true))) if weights is None else np.atleast_2d(weights)
    sums = ReconstructionSums(len(w), true.mean(axis=0))
    sums.add(true, pred, baseline, w)
    scores = sums.scores()
    return scores if weights is not None else {name: float(v[0]) for name, v in scores.items()}


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    return (a * b).sum(axis=1) / np.where(denom > 0, denom, np.nan)


def invertibility(
    sample_index: np.ndarray,
    embeddings: np.ndarray,
    target: ExpressionData,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    *,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """How much of each sample's expression a linear map recovers from its embedding.

    Ridge regression from standardized embeddings to `target.values`, fit
    per fold on its train and val samples (see `reimp_shared.eval.protocol`)
    with α chosen by leave-one-out CV within them, then scored out of fold
    by `reconstruction_scores` with patient-bootstrap intervals. The
    baseline profile is each fold's fitting mean; `alpha` is the median
    over folds.

    PCA of the target itself is the best rank-d linear code for it, so the
    PCA baseline's scores are close to the ceiling for a d-dim embedding; a
    model is not expected to beat them, only to approach them.
    """
    position = pd.Series(np.arange(len(target.samples)), index=target.samples["sample_index"])
    missing = ~np.isin(sample_index, position.index)
    if missing.any():
        raise ValueError(
            f"{missing.sum()} embedded samples are not in the target, "
            f"e.g. sample_index {sample_index[missing][:5].tolist()}"
        )
    at = position.loc[sample_index].to_numpy()  # each embedding row's row in the target
    cv = plan(target.samples, sample_index, fold, salt)
    rows = cv.rows
    if not cv.tested.any():
        return pd.DataFrame()
    cases = rows[CASE_KEY].to_numpy()
    weights = bootstrap_weights(cases[cv.tested], n_bootstrap, seed)
    column = np.cumsum(cv.tested) - 1  # each tested row's column in `weights`
    sums = ReconstructionSums(len(weights), target.values.mean(axis=0, dtype=np.float64))

    chosen = []
    for f in cv.fits:
        x, y_at = embeddings[f.rows], at[f.rows]
        y_train = target.values[y_at[f.fit]]
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(x[f.fit], y_train)
        chosen.append(model[-1].alpha_)
        if f.test.any():
            sums.add(
                target.values[y_at[f.test]],
                model.predict(x[f.test]),
                y_train.mean(axis=0, dtype=np.float64),
                weights[:, column[f.rows[f.test]]],
            )

    scores = sums.scores()
    # The baseline column is context for pearson_sample, not a result.
    baseline_pearson = scores.pop("pearson_sample_baseline")[0]
    # No interval for r2_gene: the genes that fit worst hold most of their
    # variance in a handful of samples, a resample that misses those
    # samples sends their R² far below zero, and the average over genes
    # follows. On the PCA baseline 97% of replicates fall below the
    # estimate, so a percentile interval would not cover it.
    scores["r2_gene"] = scores["r2_gene"][:1]
    return pd.DataFrame(
        [
            {
                "split": cv.name,
                "n": int(cv.tested.sum()),
                "alpha": float(np.median(chosen)),
                **summarize(scores),
                "pearson_sample_baseline": float(baseline_pearson),
            }
        ]
    )
