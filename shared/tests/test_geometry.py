import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from reimp_shared import hub
from reimp_shared.eval import (
    clustering_agreement,
    geometry_probe,
    neighbour_precision,
    spectrum,
)


def test_spectrum_of_isotropic_and_one_dimensional_embeddings() -> None:
    rng = np.random.default_rng(0)
    rank, top = spectrum(rng.normal(size=(5000, 8)))
    assert rank == pytest.approx(8, rel=0.02)
    assert top == pytest.approx(1 / 8, rel=0.1)
    rank, top = spectrum(np.outer(rng.normal(size=500), rng.normal(size=8)))
    assert rank == pytest.approx(1, abs=1e-6)
    assert top == pytest.approx(1)


def test_neighbour_precision_by_hand() -> None:
    reference = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array(["a", "a", "b", "b"])
    precision = neighbour_precision(
        np.array([[1.0, 0.05]]), reference, np.array(["a"]), labels, (1, 2, 4)
    )
    assert precision.tolist() == [[1.0, 1.0, 0.5]]


def test_clustering_agreement_matches_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 300)
    clusters = np.where(rng.random(300) < 0.7, labels, rng.integers(0, 5, 300))
    scores = clustering_agreement(labels, clusters)
    assert scores["ari"] == pytest.approx(adjusted_rand_score(labels, clusters))
    assert scores["nmi"] == pytest.approx(normalized_mutual_info_score(labels, clusters))

    # Integer weights are duplicated rows.
    weights = rng.integers(0, 3, 300).astype(float)
    repeated = np.repeat(np.arange(300), weights.astype(int))
    weighted = clustering_agreement(labels, clusters, weights[None, :])
    assert weighted["ari"][0] == pytest.approx(
        adjusted_rand_score(labels[repeated], clusters[repeated])
    )
    assert weighted["nmi"][0] == pytest.approx(
        normalized_mutual_info_score(labels[repeated], clusters[repeated])
    )


def test_geometry_probe_on_the_fake_dataset(cv_pca) -> None:
    sample_index, embeddings, fold = cv_pca
    samples = hub.load_samples()
    scores = geometry_probe(sample_index, embeddings, samples, fold=fold, n_bootstrap=50)
    assert scores["split"].tolist() == ["cv"]
    tumours = (samples["sample_type"] == "Primary Tumor").sum()
    assert scores["n"].item() == tumours
    assert scores["precision_at_1"].item() > 0.8
    assert scores["precision_at_1_lo"].item() <= scores["precision_at_1"].item()
    for column in [
        "effective_rank",
        "top_eigenvalue_share",
        "precision_at_10",
        "nmi",
        "ari",
        "silhouette",
    ]:
        assert np.isfinite(scores[column].item())
