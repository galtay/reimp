import numpy as np
import pytest

from reimp_shared.ranking import GeneRanker, inverse_document_frequency


def _values(seed: int = 0) -> np.ndarray:
    """Dense, skewed, positive — like expression."""
    return np.random.default_rng(seed).gamma(0.5, 10.0, size=(40, 12))


def test_unknown_names_raise() -> None:
    with pytest.raises(ValueError, match="unknown score"):
        GeneRanker("rank")
    with pytest.raises(ValueError, match="unknown idf scheme"):
        GeneRanker("tfidf", idf_scheme="bm25")


def test_scoring_needs_a_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        GeneRanker().scores(np.ones((1, 3)))


def test_expression_orders_by_value_with_ties_in_gene_order() -> None:
    values = np.array([[1.0, 5.0, 5.0, 0.0]])
    assert GeneRanker("expression").fit(values).order(values).tolist() == [[1, 2, 0, 3]]


def test_order_is_a_ranking_of_each_sample() -> None:
    values = _values()
    ranker = GeneRanker().fit(values)
    order = ranker.order(values)
    assert (np.sort(order, axis=1) == np.arange(values.shape[1])).all()
    ranked = np.take_along_axis(ranker.scores(values), order, axis=1)
    assert (np.diff(ranked, axis=1) <= 0).all()


def test_statistics_come_from_the_fitting_samples_only() -> None:
    train, test = _values(0), _values(1) * 100
    ranker = GeneRanker("z").fit(train)
    np.testing.assert_allclose(ranker.scores(train).mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(ranker.scores(train).std(axis=0), 1.0)
    everything = GeneRanker("z").fit(np.vstack([train, test]))
    assert not np.allclose(ranker.scores(test), everything.scores(test))


def test_z_scores_constant_genes_zero() -> None:
    values = _values()
    values[:, 3] = 2.0
    assert (GeneRanker("z").fit(values).scores(values)[:, 3] == 0).all()


def test_count_idf_gives_genes_detected_everywhere_no_weight() -> None:
    values = np.array([[1.0, 0, 3], [2, 0, 0], [5, 4, 0], [1, 0, 0]])
    idf = inverse_document_frequency(values, "count")
    np.testing.assert_allclose(idf, [0.0, np.log(4), np.log(4)])


def test_smooth_idf_never_reaches_zero() -> None:
    values = np.array([[1.0, 0, 3], [2, 0, 0], [5, 4, 0], [1, 0, 0]])
    idf = inverse_document_frequency(values, "smooth")
    np.testing.assert_allclose(idf, [1.0, np.log(5 / 2) + 1, np.log(5 / 2) + 1])


def test_entropy_idf_matches_count_for_uniform_genes_and_needs_no_threshold() -> None:
    values = np.array(
        [
            [2.0, 0, 7, 1],
            [2.0, 0, 0, 3],
            [2.0, 5, 0, 2],
            [2.0, 5, 0, 9],
        ]
    )
    entropy = inverse_document_frequency(values, "entropy")
    count = inverse_document_frequency(values, "count")
    # Genes 0-2 are uniform on the samples that express them: the forms agree.
    np.testing.assert_allclose(entropy[:3], count[:3], atol=1e-12)
    # Gene 3 is detected everywhere, but unevenly: only entropy sees it.
    assert count[3] == 0.0
    assert entropy[3] > 0.0


def test_genes_never_expressed_get_no_weight() -> None:
    values = _values()
    values[:, 0] = 0.0
    for scheme in ["count", "smooth", "entropy"]:
        assert inverse_document_frequency(values, scheme)[0] == 0.0


def test_cohort_l2_is_unchanged_by_rescaling_a_gene() -> None:
    """Scaling one gene in every sample moves it in an expression ranking, not in cohort_l2."""
    values = _values()
    scaled = values.copy()
    scaled[:, 5] *= 50

    def order(score: str, v: np.ndarray) -> np.ndarray:
        return GeneRanker(score).fit(v).order(v)

    np.testing.assert_array_equal(order("cohort_l2", values), order("cohort_l2", scaled))
    assert not np.array_equal(order("expression", values), order("expression", scaled))
