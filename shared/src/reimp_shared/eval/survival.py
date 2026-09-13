"""Survival probe: does the embedding rank patients by risk within a cancer type?

Each patient contributes one sample — the earliest primary tumour sample —
joined to the patient's endpoint, PFI by default. Liu et al. 2018 recommend
PFI over OS for most TCGA cancer types, whose follow-up is often too short
to accumulate enough deaths. A ridge-penalized Cox model, stratified by
project, is fit per fold on its train and val patients and scored out of
fold by Harrell's C-index, counting only pairs of patients from the same
project.

Only primary tumours, as TCGA-CDR (Liu et al. 2018) recommends and
SurvBoard and MultiSurv do; patients without one are not scored. The
endpoints start at diagnosis, and a metastasis or recurrence is sampled
later: scoring it would count time its patient was guaranteed to survive,
and for PFI the first progression often precedes the sample. That leaves
out most SKCM patients, who were sampled only at metastasis;
`shared/EVALS.md` has the numbers.

Stratifying is what makes the score about the embedding. Cancer type alone
predicts survival strongly, so across all projects an embedding that knew
only the cancer type would already score well. Within a project it scores
exactly 0.5, which is therefore this probe's baseline.

Two summaries: `c_index` pools comparable pairs over projects, so a
project counts in proportion to its pairs; `c_index_macro` averages each
project's own C-index (BulkRNABert's "macro C-index"), projects with at
least `MACRO_MIN_EVENTS` events counting equally. `survival_scores` also
gives each project's own C-index, from the same bootstrap draws.

Each fold has its own Cox model, whose risks are on their own scale, so
pairs are formed within a (fold, project); a project's C-index pools its
pairs over the folds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from reimp_shared.eval.bootstrap import N_BOOTSTRAP, bootstrap_weights, summarize
from reimp_shared.eval.protocol import plan
from reimp_shared.splits import CASE_KEY, DEFAULT_SALT

PRIMARY_TUMOR = ("Primary Tumor", "Primary Blood Derived Cancer - Peripheral Blood")
DEFAULT_ALPHAS = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)
N_FOLDS = 5
MACRO_MIN_EVENTS = 10


def _pairs(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pair (i, j): comparable (i had the event and t_i < t_j), and its concordance credit."""
    comparable = event[:, None] & (time[:, None] < time[None, :])
    credit = (risk[:, None] > risk[None, :]) + 0.5 * (risk[:, None] == risk[None, :])
    return comparable.astype(np.float64), comparable * credit


def concordance(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    strata: np.ndarray | None = None,
) -> tuple[float, int]:
    """Harrell's C, and the number of comparable pairs it is taken over.

    A pair (i, j) is comparable when i had the event and t_i < t_j — and,
    given `strata`, both are in the same stratum. It is concordant when
    risk_i > risk_j; a tie in risk counts half. NaN with no comparable pair.
    """
    time, risk = np.asarray(time, dtype=float), np.asarray(risk, dtype=float)
    event = np.asarray(event).astype(bool)
    strata = np.zeros(len(time)) if strata is None else np.asarray(strata)
    concordant, pairs = 0.0, 0.0
    for stratum in np.unique(strata):
        m = strata == stratum
        comparable, credit = _pairs(time[m], event[m], risk[m])
        concordant += credit.sum()
        pairs += comparable.sum()
    return (concordant / pairs if pairs else float("nan")), int(pairs)


def stratified_concordance(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    strata: np.ndarray,
    weights: np.ndarray,
    min_events: int = MACRO_MIN_EVENTS,
    groups: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """`c_index`, `c_index_macro` and `pairs` under each row of `weights`.

    A pair counts with the product of its two patients' weights. The macro
    average is over `groups` (default: the strata), each pooling the pairs
    of the strata inside it; groups with fewer than `min_events` events
    stay in `c_index` but are left out of `c_index_macro`.
    """
    time, risk = np.asarray(time, dtype=float), np.asarray(risk, dtype=float)
    event = np.asarray(event).astype(bool)
    strata = np.asarray(strata)
    groups = strata if groups is None else np.asarray(groups)
    weights = np.atleast_2d(weights)
    concordant, pairs = np.zeros(len(weights)), np.zeros(len(weights))
    by_group: dict = {}
    for stratum in np.unique(strata):
        m = strata == stratum
        comparable, credit = _pairs(time[m], event[m], risk[m])
        w = weights[:, m]
        c, p = ((w @ credit) * w).sum(axis=1), ((w @ comparable) * w).sum(axis=1)
        concordant += c
        pairs += p
        (group,) = np.unique(groups[m])  # a stratum lies inside one group
        sums = by_group.setdefault(group, [0.0, 0.0])
        sums[0] += c
        sums[1] += p
    per_group = [
        np.divide(c, p, out=np.full_like(c, np.nan), where=p > 0)
        for group, (c, p) in by_group.items()
        if event[groups == group].sum() >= min_events
    ]
    macro = np.nanmean(np.stack(per_group), axis=0) if per_group else np.full(len(weights), np.nan)
    pooled = np.divide(concordant, pairs, out=np.full_like(pairs, np.nan), where=pairs > 0)
    return {"c_index": pooled, "c_index_macro": macro, "pairs": pairs}


def cox_objective(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    alpha: float,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """β -> (loss, gradient) for a stratified Cox model with a ridge penalty.

    The loss is the negative Breslow partial log-likelihood averaged over
    patients, plus (alpha / 2)·||β||². Each stratum has its own baseline
    hazard, so risk sets never cross strata. The gradient costs two
    matrix-vector products per stratum, however wide `x` is.
    """
    # One float64 copy up front: a float32 x would be upcast on every call.
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    event = np.asarray(event).astype(bool)
    strata_rows = []
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        rows = rows[np.argsort(-time[rows], kind="stable")]
        es = event[rows]
        if not es.any():
            continue
        t = time[rows]
        # Sorted by descending time, i's risk set (everyone with t_j >= t_i)
        # is the prefix through the last patient tied with t_i.
        last = np.searchsorted(-t, -t, side="right") - 1
        xs = x[rows]
        strata_rows.append((xs, es, last[es], xs[es].sum(axis=0)))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loglik, grad = 0.0, np.zeros(d)
        for xs, es, last, x_events in strata_rows:
            eta = xs @ beta
            shift = eta.max()
            w = np.exp(eta - shift)
            s0 = np.cumsum(w)[last]  # each event's risk-set total
            loglik += (eta[es] - shift - np.log(s0)).sum()
            # The events' risk-set means of x, summed without forming one per
            # event: row j is in event i's risk set when j <= last_i, so it
            # enters with weight w_j times the sum of 1 / s0_i over those i.
            share = np.zeros(len(w))
            np.add.at(share, last, 1.0 / s0)
            share = np.cumsum(share[::-1])[::-1]
            grad += x_events - xs.T @ (w * share)
        return -loglik / n + 0.5 * alpha * beta @ beta, -grad / n + alpha * beta

    return objective


def fit_cox(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    alpha: float,
    beta0: np.ndarray | None = None,
) -> np.ndarray:
    """Coefficients of the stratified ridge Cox model (see `cox_objective`)."""
    objective = cox_objective(x, time, event, strata, alpha)
    start = np.zeros(x.shape[1]) if beta0 is None else beta0
    return minimize(objective, start, jac=True, method="L-BFGS-B").x


def select_alpha(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_folds: int = N_FOLDS,
    seed: int = 0,
) -> float:
    """The penalty with the best stratified C-index over held-out folds.

    Concordant and comparable pairs are pooled across folds. The path runs
    from the strongest penalty down, each fold's fit warm-starting its
    next, and stops once the pooled C-index falls below the best so far:
    weaker penalties only overfit further, and on wide embeddings they are
    by far the slowest to fit. Where the C-index rises to one peak along the
    path, this is the grid's best penalty.
    """
    x = np.asarray(x, dtype=np.float64)
    splits = list(KFold(n_folds, shuffle=True, random_state=seed).split(x))
    betas: list[np.ndarray | None] = [None] * len(splits)
    best, best_c = None, -np.inf
    for alpha in sorted(alphas, reverse=True):
        concordant, pairs = 0.0, 0
        for k, (fit_rows, held) in enumerate(splits):
            betas[k] = fit_cox(
                x[fit_rows], time[fit_rows], event[fit_rows], strata[fit_rows], alpha, betas[k]
            )
            c, p = concordance(time[held], event[held], x[held] @ betas[k], strata[held])
            if p:
                concordant += c * p
                pairs += p
        c = concordant / max(pairs, 1)
        if c <= best_c:
            break
        best, best_c = alpha, c
    return best


def survival_samples(rows: pd.DataFrame) -> pd.DataFrame:
    """One sample per patient: the earliest primary tumour.

    `rows` carries `case_submitter_id`, `sample_type` and `sample_index`;
    patients without a primary tumour sample are dropped.
    """
    primary = rows[rows["sample_type"].isin(PRIMARY_TUMOR)]
    return primary.sort_values("sample_index").drop_duplicates(CASE_KEY)


@dataclass(frozen=True)
class SurvivalPredictions:
    """Out-of-fold risks: one row per scored patient, and the penalty each fold chose.

    `scored` has `time`, `event`, `risk`, `project`, `fold` and `case`.
    """

    endpoint: str
    split: str
    scored: pd.DataFrame
    alphas: tuple[float, ...]


def survival_predictions(
    sample_index: np.ndarray,
    embeddings: np.ndarray,
    samples: pd.DataFrame,
    survival: pd.DataFrame,
    endpoint: str = "pfi",
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    *,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
) -> SurvivalPredictions:
    """Fit a Cox model per fold and predict its test patients' risks.

    Each fold standardizes, picks its penalty and fits on its own fitting
    patients (see `reimp_shared.eval.protocol`). `survival` has one row
    per patient with `<endpoint>_event` and `<endpoint>_time` (see
    `reimp_shared.labels.load_survival`). Patients without a primary
    tumour sample among `sample_index`, or without the endpoint, are left
    out.
    """
    event_col, time_col = f"{endpoint}_event", f"{endpoint}_time"
    cv = plan(samples, sample_index, fold, salt)
    rows = cv.rows
    labels = survival[[CASE_KEY, event_col, time_col]].dropna()
    labels = labels[labels[time_col] >= 0]

    chosen, parts = [], []
    for f in cv.fits:
        members = rows.iloc[f.rows].assign(local=np.arange(len(f.rows)))
        patients = survival_samples(members).merge(labels, on=CASE_KEY)
        local = patients["local"].to_numpy()
        time = patients[time_col].to_numpy(dtype=float)
        event = patients[event_col].to_numpy().astype(bool)
        strata = patients["project_id"].to_numpy()
        train = f.fit[local]
        x = embeddings[f.rows[local]]
        x = StandardScaler().fit(x[train]).transform(x)
        alpha = select_alpha(x[train], time[train], event[train], strata[train], alphas)
        beta = fit_cox(x[train], time[train], event[train], strata[train], alpha)
        chosen.append(alpha)
        test = f.test[local]
        parts.append(
            pd.DataFrame(
                {
                    "time": time[test],
                    "event": event[test],
                    "risk": x[test] @ beta,
                    "project": strata[test],
                    "fold": f.fold,
                    "case": patients[CASE_KEY].to_numpy()[test],
                }
            )
        )
    scored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return SurvivalPredictions(endpoint, cv.name, scored, tuple(chosen))


def survival_scores(
    predictions: SurvivalPredictions, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The summary row, and one row per project, from one set of bootstrap draws.

    `patients` and `events` report who was scored, `projects` how many
    projects enter `c_index_macro`. `alpha` is the median over folds. A project's own
    C-index is NaN where it has no comparable pair.
    """
    scored = predictions.scored
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()
    time = scored["time"].to_numpy()
    event = scored["event"].to_numpy()
    risk = scored["risk"].to_numpy()
    project = scored["project"].to_numpy()
    weights = bootstrap_weights(scored["case"].to_numpy(), n_bootstrap, seed)
    # Each fold's Cox model has its own risk scale: pair within a fold and project.
    pairing = (scored["fold"].astype(str) + ":" + scored["project"]).to_numpy()
    scores = stratified_concordance(time, event, risk, pairing, weights, groups=project)
    events_per_project = scored.groupby("project")["event"].sum()
    head = {"endpoint": predictions.endpoint, "split": predictions.split}
    summary = {
        **head,
        "patients": len(scored),
        "events": int(event.sum()),
        "pairs": int(round(scores.pop("pairs")[0])),
        "projects": int((events_per_project >= MACRO_MIN_EVENTS).sum()),
        "alpha": float(np.median(predictions.alphas)),
        **summarize(scores),
    }
    by_project = []
    for name in np.unique(project):
        m = project == name
        within = stratified_concordance(time[m], event[m], risk[m], pairing[m], weights[:, m])
        by_project.append(
            {
                **head,
                "project": name,
                "patients": int(m.sum()),
                "events": int(event[m].sum()),
                "pairs": int(round(within["pairs"][0])),
                **summarize({"c_index": within["c_index"]}),
            }
        )
    return pd.DataFrame([summary]), pd.DataFrame(by_project)


def survival_probe(
    sample_index: np.ndarray,
    embeddings: np.ndarray,
    samples: pd.DataFrame,
    survival: pd.DataFrame,
    endpoint: str = "pfi",
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    *,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold stratified C-indexes, with bootstrap intervals: the summary row.

    `survival_predictions` then `survival_scores`, which also gives each
    project's own C-index.
    """
    predictions = survival_predictions(
        sample_index,
        embeddings,
        samples,
        survival,
        endpoint,
        alphas,
        fold=fold,
        salt=salt,
    )
    return survival_scores(predictions, n_bootstrap, seed)[0]
