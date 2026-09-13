import numpy as np
import pandas as pd
import pytest

from reimp_shared import hub, labels
from reimp_shared.eval import pathway_probe, pathway_scores
from reimp_shared.testing import every_fold


def test_pathway_scores_of_perfect_and_mean_predictions() -> None:
    true = np.random.default_rng(0).normal(size=(30, 4))
    assert pathway_scores(true, true) == pytest.approx({"r2_pooled": 1.0, "r2_pathway": 1.0})
    assert pathway_scores(true, np.zeros_like(true)) == pytest.approx(
        {"r2_pooled": 0.0, "r2_pathway": 0.0}
    )


def test_r2_pooled_weights_pathways_by_variance_and_r2_pathway_does_not() -> None:
    rng = np.random.default_rng(0)
    true = np.column_stack([rng.normal(0, 10, 50), rng.normal(0, 1, 50)])
    pred = np.column_stack([true[:, 0], np.zeros(50)])  # exact on the wide pathway only
    scores = pathway_scores(true, pred)
    assert scores["r2_pathway"] == pytest.approx(0.5)
    assert scores["r2_pooled"] > 0.95


def _cohort(n: int = 900, seed: int = 0):
    """Tumour samples in 3 projects whose pathway scores are linear in embedding dim 0."""
    rng = np.random.default_rng(seed)
    project = np.array([f"P{i % 3}" for i in range(n)])
    signal = rng.normal(size=n)
    samples = pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "aliquot_id": [f"aliquot-{i}" for i in range(n)],
            "case_submitter_id": [f"case-{i}" for i in range(n)],
            "project_id": project,
            "sample_type": "Primary Tumor",
        }
    )
    # Big between-project offsets the probe must not credit to the embedding.
    offset = np.array([0.0, 3.0, -3.0])[np.arange(n) % 3]
    scores = pd.DataFrame(
        {
            "aliquot_id": samples["aliquot_id"],
            "P_A": offset + signal,
            "P_B": offset - 0.5 * signal,
            "P_C": offset + 0.1 * rng.normal(size=n),
        }
    )
    embeddings = np.column_stack([signal + 0.1 * rng.normal(size=n), rng.normal(size=(n, 4))])
    return samples, embeddings, scores


def test_pathway_probe_finds_signal_and_not_noise() -> None:
    samples, embeddings, scores = _cohort()
    sample_index, long, fold = every_fold(samples["sample_index"].to_numpy(), embeddings)
    found = pathway_probe(sample_index, long, samples, scores, fold=fold, n_bootstrap=100)
    assert found["split"].tolist() == ["cv"]
    assert found["n"].item() == len(samples)
    assert (found["pathways"] == 3).all()
    # Two of three pathways are predictable; the third is noise.
    assert (found["r2_pooled"] > 0.9).all()
    assert (found["r2_pathway"] > 0.6).all()
    assert (found["r2_pooled_lo"] <= found["r2_pooled"]).all()

    noise = np.random.default_rng(1).normal(size=long.shape)
    chance = pathway_probe(sample_index, noise, samples, scores, fold=fold, n_bootstrap=0)
    assert (chance["r2_pooled"] < 0.05).all()


def test_samples_without_scores_are_left_out() -> None:
    samples, embeddings, scores = _cohort()
    sample_index, long, fold = every_fold(samples["sample_index"].to_numpy(), embeddings)
    everyone = pathway_probe(sample_index, long, samples, scores, fold=fold, n_bootstrap=0)
    half = pathway_probe(sample_index, long, samples, scores.iloc[::2], fold=fold, n_bootstrap=0)
    assert (half["n"] < everyone["n"]).all()


def test_pathway_probe_runs_on_the_fake_dataset(cv_pca) -> None:
    sample_index, embeddings, fold = cv_pca
    result = pathway_probe(
        sample_index,
        embeddings,
        hub.load_samples(),
        labels.load_ssgsea(),
        fold=fold,
        n_bootstrap=50,
    )
    assert result["split"].tolist() == ["cv"]
    assert (result["pathways"] == 6).all()
    assert {"r2_pooled", "r2_pathway", "r2_pooled_lo"} <= set(result.columns)
