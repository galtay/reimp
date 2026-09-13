import numpy as np
import pandas as pd
import pytest

from reimp_shared import hub
from reimp_shared.eval import (
    batch_enrichment,
    centre_by_project,
    confounder_probe,
    within_project_r2,
)
from reimp_shared.labels import technical_covariates
from reimp_shared.testing import every_fold


def test_centre_by_project_uses_reference_rows_only() -> None:
    values = np.array([1.0, 3.0, 100.0, 10.0, 20.0])
    projects = np.array(["a", "a", "a", "b", "b"])
    reference = np.array([True, True, False, True, False])
    np.testing.assert_allclose(centre_by_project(values, projects, reference), [-1, 1, 98, 0, 10])


def test_within_project_r2_of_perfect_and_mean_predictions() -> None:
    true = np.array([1.0, -2.0, 0.5, 3.0])
    assert within_project_r2(true, true) == pytest.approx(1.0)
    assert within_project_r2(true, np.zeros(4)) == pytest.approx(0.0)


def _batched(n_per_batch: int = 30, separation: float = 5.0, seed: int = 0):
    """Two projects, two plates each; each plate sits in its own corner when separated."""
    rng = np.random.default_rng(seed)
    x, batches, projects = [], [], []
    for p, project in enumerate(["P0", "P1"]):
        for b, sign in enumerate([1.0, -1.0]):
            centre = np.zeros(6)
            centre[p + 2] = 3.0
            centre[0] = sign * separation
            x.append(centre + rng.normal(size=(n_per_batch, 6)))
            batches += [f"{project}-plate{b}"] * n_per_batch
            projects += [project] * n_per_batch
    x = np.vstack(x)
    reference = rng.random(len(x)) < 0.7
    return x, np.array(batches), np.array(projects), ~reference, reference


def test_batch_enrichment_finds_batches_that_cluster_and_not_ones_that_do_not() -> None:
    x, batches, projects, query, reference = _batched(separation=5.0)
    clustered = batch_enrichment(x, batches, projects, query, reference, k=5)
    # Every neighbour shares the plate (1.0) against half the project (0.5).
    assert np.nanmean(clustered[query]) == pytest.approx(0.5, abs=0.05)
    assert np.isnan(clustered[reference]).all()

    x, batches, projects, query, reference = _batched(separation=0.0)
    mixed = batch_enrichment(x, batches, projects, query, reference, k=5)
    assert abs(np.nanmean(mixed[query])) < 0.1


def test_batch_enrichment_skips_batches_unseen_in_training() -> None:
    x, batches, projects, query, reference = _batched()
    batches = batches.copy()
    lone = np.flatnonzero(query)[0]
    batches[lone] = "plate-with-no-training-samples"
    assert np.isnan(batch_enrichment(x, batches, projects, query, reference)[lone])


def _cohort(n: int = 900, seed: int = 0):
    """Tumour samples in 3 projects; embedding dim 0 carries depth within project."""
    rng = np.random.default_rng(seed)
    project = np.array([f"P{i % 3}" for i in range(n)])
    depth = rng.normal(size=n)
    samples = pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "case_submitter_id": [f"case-{i}" for i in range(n)],
            "project_id": project,
            "sample_type": "Primary Tumor",
        }
    )
    covariates = pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "tss": rng.choice(list("abc"), n),
            "plate": rng.choice(list("wxyz"), n),
            # Between-project offsets the probe must not credit to the embedding.
            "log_reads": 7.5 + np.array([0.0, 0.5, 1.0])[np.arange(n) % 3] + 0.2 * depth,
            "assigned_fraction": rng.uniform(0.5, 0.8, n),
            "strand_balance": rng.uniform(0.45, 0.55, n),
        }
    )
    embeddings = np.column_stack([depth + 0.2 * rng.normal(size=n), rng.normal(size=(n, 4))])
    return samples, embeddings, covariates


def test_confounder_probe_finds_depth_and_nothing_else() -> None:
    samples, embeddings, covariates = _cohort()
    sample_index, long, fold = every_fold(samples["sample_index"].to_numpy(), embeddings)
    scores = confounder_probe(sample_index, long, samples, covariates, fold=fold, n_bootstrap=100)
    assert scores["split"].tolist() == ["cv"]
    assert scores["n"].item() == len(samples)
    assert (scores["log_reads_r2"] > 0.8).all()
    assert (scores["assigned_fraction_r2"] < 0.1).all()
    assert (scores["plate_enrichment"].abs() < 0.1).all()
    assert (scores["log_reads_r2_lo"] <= scores["log_reads_r2"]).all()


def test_confounder_probe_runs_on_the_fake_dataset(cv_pca) -> None:
    sample_index, embeddings, fold = cv_pca
    scores = confounder_probe(
        sample_index,
        embeddings,
        hub.load_samples(),
        technical_covariates(),
        fold=fold,
        n_bootstrap=50,
    )
    columns = ["log_reads_r2", "assigned_fraction_r2", "strand_balance_r2"]
    columns += ["plate_enrichment", "tss_enrichment"]
    assert set(columns) <= set(scores.columns)
