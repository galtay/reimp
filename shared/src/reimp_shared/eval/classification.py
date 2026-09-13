"""Classification probes: logistic regression onto sample labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from reimp_shared.eval.bootstrap import N_BOOTSTRAP, bootstrap_weights, summarize
from reimp_shared.eval.protocol import plan
from reimp_shared.splits import CASE_KEY, DEFAULT_SALT

NORMAL = "Solid Tissue Normal"
# Cancer types that share a tissue of origin. Telling them apart is far
# from saturated where the 33-way task is not. COAD and READ are close to
# one disease — TCGA's colorectal study analysed them together — so near
# chance there is expected.
ORGAN_TASKS: dict[str, tuple[str, ...]] = {
    "lung": ("TCGA-LUAD", "TCGA-LUSC"),
    "kidney": ("TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP"),
    "colorectal": ("TCGA-COAD", "TCGA-READ"),
    "glioma": ("TCGA-GBM", "TCGA-LGG"),
}
TASKS = ("project_id", "tumor_vs_normal", *ORGAN_TASKS)


def task_labels(samples: pd.DataFrame, task: str) -> pd.Series:
    """Labels for one task; NaN marks samples the task leaves out."""
    normal = samples["sample_type"].eq(NORMAL)
    if task == "project_id":
        # Cancer type, from tumour samples only: adjacent normal tissue says
        # which organ a sample came from, not which cancer.
        return samples["project_id"].where(~normal)
    if task == "tumor_vs_normal":
        return normal.map({True: "normal", False: "tumor"})
    if task in ORGAN_TASKS:
        return samples["project_id"].where(~normal & samples["project_id"].isin(ORGAN_TASKS[task]))
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def fit_linear_probe(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """Standardize, then a logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x, y)


def classification_scores(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """Accuracy, balanced accuracy, macro-F1 and weighted-F1.

    Without `weights`, floats. With `weights` (replicates x rows), arrays
    over replicates: a replicate's confusion matrix is its weighted count of
    (true, predicted) pairs, so every replicate comes from one matrix
    product. Definitions follow scikit-learn: balanced accuracy averages
    recall over classes present in `y_true`; macro-F1 averages F1 over
    classes present in either; weighted-F1 weights each class's F1 by its
    support.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    w = np.ones((1, n)) if weights is None else np.atleast_2d(weights)
    _, codes = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
    k = codes.max() + 1
    cells = np.zeros((n, k * k))
    cells[np.arange(n), codes[:n] * k + codes[n:]] = 1.0
    confusion = (w @ cells).reshape(-1, k, k)  # [replicate, true, predicted]
    tp = np.diagonal(confusion, axis1=1, axis2=2)
    support, predicted = confusion.sum(axis=2), confusion.sum(axis=1)
    total = support.sum(axis=1)
    recall = np.divide(tp, support, out=np.full_like(tp, np.nan), where=support > 0)
    either = support + predicted
    f1 = np.divide(2 * tp, either, out=np.full_like(tp, np.nan), where=either > 0)
    scores = {
        "accuracy": tp.sum(axis=1) / total,
        "balanced_accuracy": np.nanmean(recall, axis=1),
        "macro_f1": np.nanmean(f1, axis=1),
        "weighted_f1": np.nansum(f1 * support, axis=1) / total,
    }
    return scores if weights is not None else {name: float(v[0]) for name, v in scores.items()}


def classification_probe(
    sample_index: np.ndarray,
    embeddings: np.ndarray,
    samples: pd.DataFrame,
    tasks: Sequence[str] = TASKS,
    *,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold scores per task, a probe fit per fold (see `reimp_shared.eval.protocol`).

    `split` names the folds pooled, `cv` for all five. Every score comes
    with a patient-bootstrap interval (`_lo`, `_hi`). A task whose fitting
    samples hold fewer than two classes in any fold — an organ task on
    data without those projects — is skipped.
    """
    cv = plan(samples, sample_index, fold, salt)
    rows = cv.rows
    cases = rows[CASE_KEY].to_numpy()
    results = []
    for task in tasks:
        labels = task_labels(rows, task)
        labelled = labels.notna().to_numpy()
        y = labels.to_numpy()
        if any(len(np.unique(y[f.rows][labelled[f.rows] & f.fit])) < 2 for f in cv.fits):
            continue
        pred = np.empty(len(rows), dtype=object)
        for f in cv.fits:
            x, y_fold, known = embeddings[f.rows], y[f.rows], labelled[f.rows]
            clf = fit_linear_probe(x[known & f.fit], y_fold[known & f.fit])
            test = known & f.test
            pred[f.rows[test]] = clf.predict(x[test])
        mask = labelled & cv.tested
        if not mask.any():
            continue
        weights = bootstrap_weights(cases[mask], n_bootstrap, seed)
        scores = classification_scores(y[mask], pred[mask], weights)
        results.append({"task": task, "split": cv.name, "n": int(mask.sum()), **summarize(scores)})
    return pd.DataFrame(results)
