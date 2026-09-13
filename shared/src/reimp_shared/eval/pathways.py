"""Pathway activity: ssGSEA scores back from the embedding, within cancer type.

For each tumour sample with ssGSEA scores (`reimp_shared.labels.load_ssgsea`,
MSigDB Hallmark by default), ridge regression from the embedding to every
pathway score at once. Pathway activity differs hugely between cancer
types, and cancer type is supposed to be in the embedding, so scores and
embeddings are both centred on each project's training mean, and R² is
relative to that mean: 0 is what the cancer type alone gives.

The targets are computed from each sample's own expression, so this is a
cousin of `invertibility`: what it adds is weighting by biology. A pathway
counts as one target however many or few genes carry it, so programs that
low-variance genes carry count as much as those in the highest-variance
genes.

- `r2_pooled`: residual over total variance, both summed over pathways, so
  a pathway counts in proportion to its within-project variance.
- `r2_pathway`: R² per pathway, averaged with every pathway counting the
  same.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from reimp_shared.eval.bootstrap import N_BOOTSTRAP, bootstrap_weights, summarize
from reimp_shared.eval.classification import task_labels
from reimp_shared.eval.confounders import centre_by_project
from reimp_shared.eval.protocol import plan
from reimp_shared.splits import CASE_KEY, DEFAULT_SALT

DEFAULT_ALPHAS = tuple(np.logspace(-2, 4, 7))


def pathway_scores(true: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """`r2_pooled` and `r2_pathway` of `pred` against `true` (samples x pathways).

    `true` is already centred on its projects' training means, so predicting
    0 scores 0. Without `weights`, floats; with `weights` (replicates x
    rows), arrays over replicates.
    """
    w = np.ones((1, len(true))) if weights is None else np.atleast_2d(weights)
    ss_res = w @ (true - pred) ** 2
    ss_tot = w @ true**2
    scores = {
        "r2_pooled": 1 - ss_res.sum(axis=1) / ss_tot.sum(axis=1),
        "r2_pathway": (1 - ss_res / ss_tot).mean(axis=1),
    }
    return scores if weights is not None else {name: float(v[0]) for name, v in scores.items()}


def pathway_probe(
    sample_index: np.ndarray,
    embeddings: np.ndarray,
    samples: pd.DataFrame,
    scores: pd.DataFrame,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    *,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold within-project R², with bootstrap intervals.

    Each fold centres on its own fitting rows' project means and fits its
    own ridge (see `reimp_shared.eval.protocol`). `scores` has an
    `aliquot_id` column and one column per pathway. Samples without scores
    are left out; `n` counts those scored. `alpha` is the median over folds.
    """
    cv = plan(samples, sample_index, fold, salt)
    rows = cv.rows
    pathways = [c for c in scores.columns if c != "aliquot_id"]
    matched = rows[["aliquot_id"]].merge(scores, on="aliquot_id", how="left")
    y = matched[pathways].to_numpy(dtype=np.float64)
    has_scores = ~np.isnan(y).any(axis=1)
    tumour = task_labels(rows, "project_id").notna().to_numpy() & has_scores
    projects = rows["project_id"].to_numpy()
    cases = rows[CASE_KEY].to_numpy()

    target, pred = np.full_like(y, np.nan), np.full_like(y, np.nan)
    chosen = []
    for f in cv.fits:
        train = tumour[f.rows] & f.fit
        x = centre_by_project(embeddings[f.rows], projects[f.rows], train)
        scaled = StandardScaler().fit(x[train]).transform(x)
        centred = centre_by_project(y[f.rows], projects[f.rows], train)
        model = RidgeCV(alphas=alphas).fit(scaled[train], centred[train])
        chosen.append(model.alpha_)
        # Projects with no fitting rows have no centre and are not scored.
        test = tumour[f.rows] & f.test & ~np.isnan(centred).any(axis=1)
        target[f.rows[test]] = centred[test]
        pred[f.rows[test]] = model.predict(scaled[test])

    mask = cv.tested & ~np.isnan(target).any(axis=1)
    if not mask.any():
        return pd.DataFrame()
    weights = bootstrap_weights(cases[mask], n_bootstrap, seed)
    r2 = pathway_scores(target[mask], pred[mask], weights)
    return pd.DataFrame(
        [
            {
                "split": cv.name,
                "n": int(mask.sum()),
                "pathways": len(pathways),
                "alpha": float(np.median(chosen)),
                **summarize(r2),
            }
        ]
    )
