import numpy as np
import pytest

from reimp_vae.scaling import GeneScaler, median_absolute_deviation, top_genes_by_mad


def test_mad_is_the_median_not_the_mean_absolute_deviation() -> None:
    """Gene 0 is constant but for two outliers: large mean deviation, no median deviation."""
    values = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [100, 5], [100, 6]], dtype=float)
    np.testing.assert_array_equal(median_absolute_deviation(values), [0.0, 2.0])
    assert top_genes_by_mad(values, 1).tolist() == [1]
    # Tybalt's code (pandas' mean absolute deviation) would have kept gene 0.
    assert np.abs(values - values.mean(axis=0)).mean(axis=0).argmax() == 0


def test_top_genes_keep_column_order_and_break_ties_to_the_earlier_column() -> None:
    base = np.array([-1.0, 0.0, 1.0])  # MAD 1
    values = base[:, None] * np.array([1, 3, 3, 0, 2])  # MADs 1, 3, 3, 0, 2
    assert top_genes_by_mad(values, 1).tolist() == [1]
    assert top_genes_by_mad(values, 3).tolist() == [1, 2, 4]
    # Exactly as many as asked, even genes with no deviation at all.
    assert top_genes_by_mad(values, 5).tolist() == [0, 1, 2, 3, 4]


def test_minmax_maps_training_rows_onto_the_unit_interval_and_clips_the_rest() -> None:
    train = np.array([[0.0, 10, 5], [2, 20, 5], [4, 30, 5]])
    scaler = GeneScaler("minmax").fit(train)
    np.testing.assert_allclose(
        scaler.transform(train), [[0, 0, 0], [0.5, 0.5, 0], [1, 1, 0]], atol=1e-7
    )
    # Outside the training range: clipped, since BCE and a sigmoid cannot represent it.
    # The constant gene has scale 1, so 7 maps to 2 before clipping.
    np.testing.assert_allclose(scaler.transform(np.array([[-2.0, 40, 7]])), [[0, 1, 1]])
    assert scaler.transform(train).dtype == np.float32


def test_zscore_uses_the_training_mean_and_sd_and_does_not_clip() -> None:
    train = np.random.default_rng(0).normal(3.0, 2.0, (500, 4))
    scaler = GeneScaler("zscore").fit(train)
    z = scaler.transform(train).astype(np.float64)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-5)
    far = train.mean(axis=0, keepdims=True) + 10 * train.std(axis=0)
    np.testing.assert_allclose(scaler.transform(far), 10.0, rtol=1e-5)


def test_genes_and_statistics_come_from_the_rows_fit() -> None:
    values = np.random.default_rng(0).normal(0.0, [1.0, 5.0, 0.1, 3.0], (100, 4))
    train = values[:60]
    scaler = GeneScaler("minmax", n_genes=2).fit(train)
    assert scaler.genes_.tolist() == [1, 3]
    np.testing.assert_array_equal(scaler.offset_, train[:, [1, 3]].min(axis=0))
    np.testing.assert_array_equal(scaler.scale_, np.ptp(train[:, [1, 3]], axis=0))
    assert scaler.transform(values).shape == (100, 2)


def test_none_keeps_values_and_can_still_select() -> None:
    values = np.random.default_rng(0).normal(0.0, [1.0, 5.0, 0.1], (50, 3))
    scaler = GeneScaler("none", n_genes=2).fit(values)
    np.testing.assert_allclose(scaler.transform(values), values[:, [0, 1]], rtol=1e-6)


def test_state_dict_round_trip() -> None:
    values = np.random.default_rng(0).normal(0.0, [1.0, 5.0, 0.1, 3.0], (40, 4))
    fitted = GeneScaler("zscore", n_genes=3).fit(values)
    restored = GeneScaler("zscore", n_genes=3)
    restored.load_state_dict(fitted.state_dict())
    np.testing.assert_array_equal(restored.transform(values), fitted.transform(values))
    assert GeneScaler().state_dict() == {}


def test_misuse_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        GeneScaler().transform(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unknown scaling"):
        GeneScaler("robust")
    with pytest.raises(ValueError, match="positive"):
        GeneScaler(n_genes=0)
