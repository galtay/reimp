import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.model_selection import KFold

from reimp_shared.eval.survival import (
    concordance,
    cox_objective,
    fit_cox,
    select_alpha,
    stratified_concordance,
    survival_predictions,
    survival_probe,
    survival_samples,
    survival_scores,
)
from reimp_shared.splits import sample_folds
from reimp_shared.testing import every_fold

BETA = np.array([1.0, -0.5, 0.0])


def _simulate(n: int = 3000, seed: int = 0):
    """Cox-model data with a different baseline hazard in each of 3 strata."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(BETA)))
    strata = rng.integers(0, 3, n)
    baseline = np.array([0.5, 1.0, 2.0])[strata]
    event_time = rng.exponential(1 / (baseline * np.exp(x @ BETA)))
    censor_time = rng.exponential(1.5, n)
    return x, np.minimum(event_time, censor_time), event_time <= censor_time, strata


# ---------- concordance ----------


def test_concordance_of_perfect_and_reversed_rankings() -> None:
    time, event = np.array([1.0, 2, 3, 4]), np.ones(4)
    assert concordance(time, event, -time) == (1.0, 6)
    assert concordance(time, event, time) == (0.0, 6)


def test_concordance_by_hand() -> None:
    # Only patient 0 anchors pairs: 1 is censored, 2 and 3 have nobody later.
    # Against 1: risk 3 < 5, discordant. Against 2: 3 > 1, concordant.
    # Against 3: a tie in risk, half.
    time = np.array([1, 2, 3, 3])
    event = np.array([1, 0, 1, 1])
    risk = np.array([3, 5, 1, 3])
    assert concordance(time, event, risk) == pytest.approx((0.5, 3))


def test_concordance_only_compares_within_strata() -> None:
    time, event = np.array([1.0, 2, 1, 2]), np.ones(4)
    risk = np.array([2.0, 1, 4, 3])
    strata = np.array(["a", "a", "b", "b"])
    assert concordance(time, event, risk, strata) == (1.0, 2)
    assert concordance(time, event, risk) == (0.75, 4)


def test_concordance_without_comparable_pairs_is_nan() -> None:
    c, pairs = concordance(np.array([1.0, 2]), np.zeros(2), np.array([0.0, 1]))
    assert np.isnan(c) and pairs == 0


def test_stratified_concordance_with_unit_weights_is_concordance() -> None:
    x, time, event, strata = _simulate(n=300)
    risk = x @ BETA
    scores = stratified_concordance(time, event, risk, strata, np.ones((1, 300)))
    c, pairs = concordance(time, event, risk, strata)
    assert scores["c_index"][0] == pytest.approx(c)
    assert scores["pairs"][0] == pairs


def test_integer_weights_are_duplicated_patients() -> None:
    x, time, event, strata = _simulate(n=300)
    risk = x @ BETA
    weights = np.random.default_rng(1).integers(0, 3, 300).astype(float)
    repeated = np.repeat(np.arange(300), weights.astype(int))
    weighted = stratified_concordance(time, event, risk, strata, weights[None, :])
    c, _ = concordance(time[repeated], event[repeated], risk[repeated], strata[repeated])
    assert weighted["c_index"][0] == pytest.approx(c)


def test_macro_c_index_counts_strata_equally() -> None:
    # Stratum a: 10 events ranked perfectly (45 pairs). Stratum b: 4 events
    # ranked backwards (6 pairs).
    time = np.concatenate([np.arange(1.0, 11), np.arange(1.0, 5)])
    event = np.ones(14)
    risk = np.concatenate([-np.arange(1.0, 11), np.arange(1.0, 5)])
    strata = np.array(["a"] * 10 + ["b"] * 4)
    scores = stratified_concordance(time, event, risk, strata, np.ones((1, 14)), min_events=3)
    assert scores["c_index"][0] == pytest.approx(45 / 51)
    assert scores["c_index_macro"][0] == pytest.approx(0.5)
    # Too few events and a stratum drops out of the macro average only.
    scores = stratified_concordance(time, event, risk, strata, np.ones((1, 14)), min_events=5)
    assert scores["c_index"][0] == pytest.approx(45 / 51)
    assert scores["c_index_macro"][0] == pytest.approx(1.0)


def test_macro_groups_pool_their_strata() -> None:
    # Group g holds two strata: 6 pairs ranked perfectly and 1 backwards;
    # group h one stratum of 3 pairs ranked perfectly.
    time = np.array([1.0, 2, 3, 4, 1, 2, 1, 2, 3])
    event = np.ones(9)
    risk = np.array([4.0, 3, 2, 1, 1, 2, 3, 2, 1])
    strata = np.array(["g1"] * 4 + ["g2"] * 2 + ["h"] * 3)
    groups = np.array(["g"] * 6 + ["h"] * 3)
    scores = stratified_concordance(
        time, event, risk, strata, np.ones((1, 9)), min_events=3, groups=groups
    )
    assert scores["pairs"][0] == 10
    assert scores["c_index"][0] == pytest.approx(9 / 10)
    assert scores["c_index_macro"][0] == pytest.approx((6 / 7 + 1) / 2)


# ---------- Cox model ----------


def test_cox_gradient_matches_finite_differences() -> None:
    x, time, event, strata = _simulate(n=300)
    objective = cox_objective(x, np.round(time, 1), event, strata, alpha=0.1)  # with ties
    beta = np.array([0.3, -0.2, 0.1])
    error = check_grad(lambda b: objective(b)[0], lambda b: objective(b)[1], beta)
    assert error < 1e-6


def test_stratified_cox_recovers_the_coefficients() -> None:
    x, time, event, strata = _simulate()
    np.testing.assert_allclose(fit_cox(x, time, event, strata, alpha=0.0), BETA, atol=0.1)


def test_ridge_penalty_shrinks_the_coefficients() -> None:
    x, time, event, strata = _simulate()
    small = fit_cox(x, time, event, strata, alpha=1e-3)
    large = fit_cox(x, time, event, strata, alpha=1.0)
    assert np.linalg.norm(large) < 0.5 * np.linalg.norm(small)


def test_penalty_path_stops_once_held_out_concordance_falls(monkeypatch) -> None:
    """The early stop picks the grid's best penalty, without fitting the weaker ones."""
    x, time, event, strata = _simulate(n=600)
    x = np.hstack([x, np.random.default_rng(2).normal(size=(600, 40))])  # mostly noise
    alphas = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)
    # The grid's best, from every penalty fit on its own inner folds.
    concordant, pairs = np.zeros(len(alphas)), np.zeros(len(alphas))
    for fit_rows, held in KFold(5, shuffle=True, random_state=0).split(x):
        for k, alpha in enumerate(alphas):
            beta = fit_cox(x[fit_rows], time[fit_rows], event[fit_rows], strata[fit_rows], alpha)
            c, p = concordance(time[held], event[held], x[held] @ beta, strata[held])
            concordant[k] += c * p
            pairs[k] += p
    grid_best = alphas[int(np.argmax(concordant / pairs))]

    fitted = []
    module = importlib.import_module("reimp_shared.eval.survival")
    original = module.fit_cox
    monkeypatch.setattr(module, "fit_cox", lambda *a, **k: fitted.append(a[4]) or original(*a, **k))
    chosen = select_alpha(x, time, event, strata, alphas)
    assert chosen == pytest.approx(grid_best, rel=1e-6)
    assert min(fitted) > min(alphas)  # stopped before the weakest penalty


# ---------- survival_probe ----------


def _cohort(n_patients: int = 1500, seed: int = 0):
    """Patients in 3 projects whose first embedding dim carries their log-hazard.

    Every patient has a primary tumour; some also have a normal or a second
    tumour aliquot, and a few have no endpoint.
    """
    rng = np.random.default_rng(seed)
    cases = [f"case-{i}" for i in range(n_patients)]
    project = [f"P{i % 3}" for i in range(n_patients)]
    log_hazard = rng.normal(size=n_patients)
    event_time = rng.exponential(np.exp(-log_hazard) * 1000 * (1 + np.arange(n_patients) % 3))
    censor_time = rng.uniform(200, 3000, n_patients)
    survival = pd.DataFrame(
        {
            "case_submitter_id": cases,
            "pfi_event": (event_time <= censor_time).astype(int),
            "pfi_time": np.minimum(event_time, censor_time),
        }
    )
    survival.loc[:9, ["pfi_event", "pfi_time"]] = np.nan

    records, features = [], []
    for i in range(n_patients):
        kinds = ["Primary Tumor"]
        kinds += ["Solid Tissue Normal"] if i % 5 == 0 else []
        kinds += ["Primary Tumor"] if i % 7 == 0 else []
        for kind in kinds:
            records.append((cases[i], project[i], kind))
            signal = log_hazard[i] if kind == "Primary Tumor" else rng.normal()
            features.append([signal + 0.3 * rng.normal(), *rng.normal(size=3)])
    samples = pd.DataFrame(records, columns=["case_submitter_id", "project_id", "sample_type"])
    samples.insert(0, "sample_index", np.arange(len(samples)))
    return samples, np.array(features), survival


def test_survival_probe_finds_the_signal_and_not_noise() -> None:
    samples, embeddings, survival = _cohort()
    sample_index, long, fold = every_fold(samples["sample_index"].to_numpy(), embeddings)
    scores = survival_probe(sample_index, long, samples, survival, fold=fold, n_bootstrap=200)
    assert scores["split"].tolist() == ["cv"]
    (row,) = scores.to_dict(orient="records")
    assert row["c_index"] > 0.7 and row["c_index_macro"] > 0.7
    assert row["c_index_lo"] <= row["c_index"] <= row["c_index_hi"]
    assert row["projects"] == 3

    noise = np.random.default_rng(1).normal(size=long.shape)
    chance = survival_probe(sample_index, noise, samples, survival, fold=fold, n_bootstrap=0)
    assert (chance["c_index"] - 0.5).abs().max() < 0.07


def test_one_fold_scores_one_sample_per_labelled_patient_it_tests() -> None:
    samples, embeddings, survival = _cohort()
    fold = np.full(len(samples), 2)
    scores = survival_probe(
        samples["sample_index"].to_numpy(), embeddings, samples, survival, fold=fold, n_bootstrap=0
    )
    labelled = survival.dropna()["case_submitter_id"]
    patients = samples.drop_duplicates("case_submitter_id")
    patients = patients[patients["case_submitter_id"].isin(labelled)]
    assert scores["split"].tolist() == ["fold2"]
    case_fold = dict(zip(samples["case_submitter_id"], sample_folds(samples), strict=True))
    assert scores["patients"].item() == (patients["case_submitter_id"].map(case_fold) == 2).sum()


def test_only_the_earliest_primary_tumour_is_scored() -> None:
    rows = pd.DataFrame(
        {
            "case_submitter_id": ["a", "a", "b", "b", "b", "c", "d"],
            "sample_type": [
                "Metastatic",
                "Primary Tumor",
                "Metastatic",
                "Additional Metastatic",
                "Solid Tissue Normal",
                "Recurrent Tumor",
                "Primary Blood Derived Cancer - Peripheral Blood",
            ],
            "sample_index": [0, 5, 3, 1, 2, 4, 6],
        }
    )
    chosen = survival_samples(rows).set_index("case_submitter_id")
    # a: its primary, though a metastasis came first; b and c: metastases and
    # a recurrence only, sampled after diagnosis, left out; d: a blood-derived
    # primary.
    assert chosen["sample_index"].to_dict() == {"a": 5, "d": 6}


def test_per_project_rows_split_the_summary() -> None:
    samples, embeddings, survival = _cohort()
    sample_index, long, fold = every_fold(samples["sample_index"].to_numpy(), embeddings)
    predictions = survival_predictions(sample_index, long, samples, survival, fold=fold)
    summary, projects = survival_scores(predictions, n_bootstrap=50)
    (row,) = summary.to_dict(orient="records")
    assert projects["project"].tolist() == ["P0", "P1", "P2"]
    assert projects["patients"].sum() == row["patients"]
    assert projects["events"].sum() == row["events"]
    assert projects["pairs"].sum() == row["pairs"]
    # The pooled C-index is the projects' pairs pooled.
    pooled = (projects["c_index"] * projects["pairs"]).sum() / projects["pairs"].sum()
    assert row["c_index"] == pytest.approx(pooled)
    assert (projects["c_index_lo"] <= projects["c_index"]).all()
    assert (projects["c_index"] > 0.65).all()
    # The summary is what survival_probe returns.
    probe = survival_probe(sample_index, long, samples, survival, fold=fold, n_bootstrap=50)
    assert probe["c_index"].item() == pytest.approx(row["c_index"])


def test_cv_scores_every_patient_once_and_pairs_within_folds() -> None:
    samples, embeddings, survival = _cohort()
    sample_index, long, fold = every_fold(samples["sample_index"].to_numpy(), embeddings)
    scores = survival_probe(sample_index, long, samples, survival, fold=fold, n_bootstrap=0)
    (row,) = scores.to_dict(orient="records")

    # Every labelled patient once; pairs only between patients of one fold and project.
    patients = samples.drop_duplicates("case_submitter_id").merge(
        survival.dropna(), on="case_submitter_id"
    )
    assert row["patients"] == len(patients)
    # Folds come from the whole cohort, not from the labelled patients alone.
    case_fold = dict(zip(samples["case_submitter_id"], sample_folds(samples), strict=True))
    fold_of = patients["case_submitter_id"].map(case_fold).astype(str)
    strata = patients["project_id"] + "@" + fold_of
    _, pairs = concordance(
        patients["pfi_time"], patients["pfi_event"], np.zeros(len(patients)), strata.to_numpy()
    )
    assert row["pairs"] == pairs
