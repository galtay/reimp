import numpy as np

from reimp_plier.scaler import GeneScaler


def test_gene_scaler_fits_training_statistics_and_clips_to_their_range() -> None:
    train = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [2.0, 5.0, 9.0]], dtype=np.float32)
    s = GeneScaler.fit(train)
    np.testing.assert_allclose(s.mean, [2.0, 5.0, 5.0])
    np.testing.assert_allclose(s.sd, train.std(axis=0, ddof=1))
    np.testing.assert_array_equal(s.low, [1.0, 5.0, 2.0])
    np.testing.assert_array_equal(s.high, [3.0, 5.0, 9.0])
    # The constant gene is dropped.
    assert s.varying().tolist() == [0, 2]
    kept = s.subset(s.varying())
    np.testing.assert_allclose(kept.transform(train[:, [0, 2]]).mean(axis=0), 0, atol=1e-12)
    # A held-out value outside the training range is taken at its end.
    wild = np.array([[100.0, -100.0]], dtype=np.float32)
    expected = (np.array([3.0, 2.0]) - kept.mean) / kept.sd
    np.testing.assert_allclose(kept.transform(wild)[0], expected)
    assert kept.transform(wild).dtype == np.float64
