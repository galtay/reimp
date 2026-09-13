"""Technical confounders: how the sample was processed, as seen in its embedding.

A representation of biology should say little about how a sample was
processed. Plates, sites and library quality all differ between cancer
types, and cancer type is supposed to be in the embedding, so everything
here is measured within projects, on tumour samples:

- **Library QC** — `log_reads` (sequencing depth), `assigned_fraction`
  (share of reads assigned to genes), `strand_balance`. Ridge regression
  from the embedding, both centred on each project's training mean; R²
  relative to that mean. 0: the embedding predicts nothing about the
  covariate beyond cancer type.
- **Batch** — sequencing `plate` and tissue source site `tss`. For each
  sample, the share of its nearest same-project training neighbours
  (cosine) that share its plate or site, minus that share among all the
  project's training samples. 0: neighbours ignore the batch; positive:
  the embedding groups samples by batch.

Lower is more invariant, but not all of it is artefact: sites differ in
their patients, and plates in when and from where samples arrived. Read a
model against the PCA baseline and the other models, not against zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from reimp_shared.eval.bootstrap import (
    N_BOOTSTRAP,
    bootstrap_weights,
    summarize,
    weighted_mean,
)
from reimp_shared.eval.classification import task_labels
from reimp_shared.eval.geometry import _unit
from reimp_shared.eval.protocol import plan
from reimp_shared.splits import CASE_KEY, DEFAULT_SALT

QC_COVARIATES = ("log_reads", "assigned_fraction", "strand_balance")
BATCH_COVARIATES = ("plate", "tss")
DEFAULT_K = 10
DEFAULT_ALPHAS = tuple(np.logspace(-2, 4, 7))


def centre_by_project(
    values: np.ndarray, projects: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """`values` minus each project's mean over its `reference` rows, ignoring NaN.

    A project without reference rows becomes NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.full_like(values, np.nan)
    for project in np.unique(projects):
        rows = projects == project
        base = rows & reference
        if base.any():
            out[rows] = values[rows] - np.nanmean(values[base], axis=0)
    return out


def within_project_r2(true: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None):
    """1 − SS_res / SS_tot, with `true` already centred on its projects' training means.

    Predicting 0 — each project's training mean — scores 0.
    """
    w = np.ones((1, len(true))) if weights is None else np.atleast_2d(weights)
    r2 = 1 - (w @ (true - pred) ** 2) / (w @ true**2)
    return r2 if weights is not None else float(r2[0])


def batch_enrichment(
    x: np.ndarray,
    batches: np.ndarray,
    projects: np.ndarray,
    query: np.ndarray,
    reference: np.ndarray,
    k: int = DEFAULT_K,
) -> np.ndarray:
    """Per row: same-batch share among its k nearest same-project references, minus chance.

    Chance is the same-batch share among all the project's references. NaN
    for rows outside `query`, and for queries whose batch has no references
    in their project.
    """
    unit = _unit(np.asarray(x, dtype=np.float64))
    batches = np.asarray(batches, dtype=object)
    out = np.full(len(unit), np.nan)
    for project in np.unique(projects[query]):
        q = np.flatnonzero(query & (projects == project))
        r = np.flatnonzero(reference & (projects == project))
        if len(r) < 2:
            continue
        k_eff = min(k, len(r))
        similarity = unit[q] @ unit[r].T
        nearest = np.argpartition(-similarity, k_eff - 1, axis=1)[:, :k_eff]
        observed = (batches[r][nearest] == batches[q][:, None]).mean(axis=1)
        chance = (batches[r][None, :] == batches[q][:, None]).mean(axis=1)
        out[q] = np.where(chance > 0, observed - chance, np.nan)
    return out


def confounder_probe(
    sample_index: np.ndarray,
    embeddings: np.ndarray,
    samples: pd.DataFrame,
    covariates: pd.DataFrame,
    k: int = DEFAULT_K,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    *,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold within-project QC R² and batch enrichment, with bootstrap intervals.

    Each fold centres, regresses and finds neighbours among its own fitting
    rows (see `reimp_shared.eval.protocol`). `covariates` has one row per
    `sample_index` with the columns of
    `reimp_shared.labels.technical_covariates`.
    """
    cv = plan(samples, sample_index, fold, salt)
    rows = cv.rows
    covariates = covariates.set_index("sample_index").loc[sample_index].reset_index(drop=True)
    tumour = task_labels(rows, "project_id").notna().to_numpy()
    projects = rows["project_id"].to_numpy()
    cases = rows[CASE_KEY].to_numpy()

    qc_true = {name: np.full(len(rows), np.nan) for name in QC_COVARIATES}
    qc_pred = {name: np.full(len(rows), np.nan) for name in QC_COVARIATES}
    enrichment = {name: np.full(len(rows), np.nan) for name in BATCH_COVARIATES}
    for f in cv.fits:
        train = tumour[f.rows] & f.fit
        scored = tumour[f.rows] & f.test
        in_projects = projects[f.rows]
        x = centre_by_project(embeddings[f.rows], in_projects, train)
        scaled = StandardScaler().fit(x[train]).transform(x)
        for name in QC_COVARIATES:
            values = covariates[name].to_numpy(dtype=float)[f.rows]
            target = centre_by_project(values, in_projects, train)
            valid = ~np.isnan(target)
            model = RidgeCV(alphas=alphas).fit(scaled[train & valid], target[train & valid])
            ok = scored & valid
            qc_true[name][f.rows[ok]] = target[ok]
            qc_pred[name][f.rows[ok]] = model.predict(scaled[ok])
        for name in BATCH_COVARIATES:
            batches = covariates[name].to_numpy(dtype=object)[f.rows]
            known = pd.notna(batches)
            found = batch_enrichment(x, batches, in_projects, scored & known, train & known, k)
            enrichment[name][f.rows[scored]] = found[scored]

    mask = tumour & cv.tested
    if not mask.any():
        return pd.DataFrame()
    weights = bootstrap_weights(cases[mask], n_bootstrap, seed)
    scores = {}
    for name in QC_COVARIATES:
        true, pred = qc_true[name][mask], qc_pred[name][mask]
        ok = ~np.isnan(true)
        scores[f"{name}_r2"] = within_project_r2(true[ok], pred[ok], weights[:, ok])
    for name in BATCH_COVARIATES:
        scores[f"{name}_enrichment"] = weighted_mean(weights, enrichment[name][mask])
    return pd.DataFrame([{"split": cv.name, "n": int(mask.sum()), **summarize(scores)}])
