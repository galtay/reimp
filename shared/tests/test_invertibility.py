import importlib

import numpy as np
import pytest

from reimp_shared.eval import invertibility, invertibility_target, reconstruction_scores
from reimp_shared.eval.invertibility import ReconstructionSums
from reimp_shared.testing import PROJECTS, every_fold


def _profiles(n: int = 40, g: int = 6, seed: int = 0) -> np.ndarray:
    """Samples whose genes differ in mean far more than the samples differ."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 5.0, g) + rng.normal(0.0, 1.0, (n, g))


# ---------- reconstruction_scores ----------


def test_perfect_prediction_scores_one() -> None:
    true = _profiles()
    scores = reconstruction_scores(true, true, true.mean(axis=0))
    for name in ["r2_pooled", "r2_gene", "pearson_sample"]:
        assert scores[name] == pytest.approx(1.0)


def test_predicting_the_gene_means_scores_zero_r2_but_high_pearson() -> None:
    """Why per-sample Pearson needs its baseline column."""
    true = _profiles()
    means = true.mean(axis=0)
    scores = reconstruction_scores(true, np.broadcast_to(means, true.shape), means)
    assert scores["r2_pooled"] == pytest.approx(0.0, abs=1e-9)
    assert scores["r2_gene"] == pytest.approx(0.0, abs=1e-9)
    assert scores["pearson_sample"] == pytest.approx(scores["pearson_sample_baseline"])
    assert scores["pearson_sample_baseline"] > 0.9


def test_r2_pooled_weights_genes_by_variance() -> None:
    rng = np.random.default_rng(0)
    true = np.column_stack([rng.normal(0, 10, 50), rng.normal(0, 1, 50)])
    # Exact on the high-variance gene, the gene mean on the low-variance one.
    pred = np.column_stack([true[:, 0], np.full(50, true[:, 1].mean())])
    scores = reconstruction_scores(true, pred, true.mean(axis=0))
    assert scores["r2_gene"] == pytest.approx(0.5)
    assert scores["r2_pooled"] > 0.95


def test_constant_genes_are_left_out_of_r2_gene() -> None:
    true = _profiles()
    true[:, 0] = 3.0
    scores = reconstruction_scores(true, true, true.mean(axis=0))
    assert scores["r2_gene"] == pytest.approx(1.0)


def test_integer_weights_are_duplicated_samples() -> None:
    rng = np.random.default_rng(0)
    true = _profiles()
    pred = true + rng.normal(0.0, 0.5, true.shape)
    baseline = true.mean(axis=0)
    weights = rng.integers(0, 3, len(true)).astype(float)
    repeated = np.repeat(np.arange(len(true)), weights.astype(int))
    weighted = reconstruction_scores(true, pred, baseline, weights[None, :])
    direct = reconstruction_scores(true[repeated], pred[repeated], baseline)
    assert {name: v[0] for name, v in weighted.items()} == pytest.approx(direct)


def test_sums_over_pieces_score_as_one_set(monkeypatch) -> None:
    """Folds arriving one at a time, each with its own baseline, in blocks of rows."""
    # The package re-exports the function under the module's name, so reach
    # the module itself.
    monkeypatch.setattr(importlib.import_module("reimp_shared.eval.invertibility"), "BLOCK_ROWS", 7)
    rng = np.random.default_rng(0)
    true = _profiles(n=60)
    pred = true + rng.normal(0.0, 0.5, true.shape)
    baselines = [true[:25].mean(axis=0), true[25:].mean(axis=0) + 1.0]
    weights = rng.integers(0, 3, (4, 60)).astype(float)
    sums = ReconstructionSums(4, np.zeros(true.shape[1]) + 3.0)
    sums.add(true[:25], pred[:25], baselines[0], weights[:, :25])
    sums.add(true[25:], pred[25:], baselines[1], weights[:, 25:])
    pieces = sums.scores()
    whole = reconstruction_scores(true, pred, baselines[0], weights)
    for name in ["r2_pooled", "r2_gene", "pearson_sample"]:
        np.testing.assert_allclose(pieces[name], whole[name])
    # The baseline Pearson is each row's against its own piece's baseline.
    for r in range(4):
        direct = [
            reconstruction_scores(true[rows], pred[rows], base, weights[r : r + 1, rows])
            for rows, base in [(slice(0, 25), baselines[0]), (slice(25, 60), baselines[1])]
        ]
        counts = [weights[r, :25].sum(), weights[r, 25:].sum()]
        expected = sum(
            c * d["pearson_sample_baseline"][0] for c, d in zip(counts, direct, strict=True)
        )
        assert pieces["pearson_sample_baseline"][r] == pytest.approx(expected / sum(counts))


# ---------- invertibility ----------


def test_invertibility_separates_informative_from_random_embeddings(fake_dataset) -> None:
    target = invertibility_target()
    sample_index, values, fold = every_fold(
        target.samples["sample_index"].to_numpy(), target.values
    )
    exact = invertibility(sample_index, values, target, fold=fold, n_bootstrap=100)
    noise = np.random.default_rng(0).normal(size=(len(sample_index), 8))
    random = invertibility(sample_index, noise, target, fold=fold, n_bootstrap=100)
    assert exact["split"].tolist() == ["cv"]
    assert exact["n"].item() == len(target.samples)
    assert (exact["r2_pooled"] > 0.99).all()
    assert (random["r2_pooled"] < 0.2).all()
    assert (exact["pearson_sample"] > random["pearson_sample"]).all()
    assert (random["r2_pooled_lo"] <= random["r2_pooled"]).all()
    assert (random["r2_pooled"] <= random["r2_pooled_hi"]).all()
    assert "r2_gene_lo" not in random  # biased under resampling; see invertibility()


def test_invertibility_scores_whichever_samples_were_embedded(fake_dataset) -> None:
    target = invertibility_target()
    keep = (target.samples["project_id"] != PROJECTS[0]).to_numpy()
    sample_index, embeddings, fold = every_fold(
        target.samples["sample_index"].to_numpy()[keep], target.values[keep]
    )
    scores = invertibility(sample_index, embeddings, target, fold=fold, n_bootstrap=0)
    assert scores["n"].item() == keep.sum()


def test_invertibility_rejects_samples_missing_from_the_target(fake_dataset) -> None:
    target = invertibility_target()
    with pytest.raises(ValueError, match="not in the target"):
        invertibility(np.array([10_000]), np.zeros((1, 3)), target, fold=np.array([0]))
