import numpy as np
import pytest

from reimp_shared.eval.bootstrap import bootstrap_weights, summarize, weighted_mean


def test_rows_of_a_case_are_drawn_together() -> None:
    cases = np.array(["a", "a", "b", "c", "c", "c"])
    weights = bootstrap_weights(cases, n_bootstrap=200, seed=0)
    assert weights.shape == (201, 6)
    assert (weights[0] == 1).all()
    replicates = weights[1:]
    assert (replicates[:, 0] == replicates[:, 1]).all()
    assert (replicates[:, 3] == replicates[:, 5]).all()
    # Each replicate draws as many cases as there are.
    assert (replicates[:, [0, 2, 3]].sum(axis=1) == 3).all()


def test_seeded_replicates_repeat() -> None:
    cases = np.arange(50)
    np.testing.assert_array_equal(bootstrap_weights(cases, 10, 3), bootstrap_weights(cases, 10, 3))


def test_no_replicates_leaves_only_the_estimate() -> None:
    weights = bootstrap_weights(np.array(["a", "b"]), n_bootstrap=0)
    assert weights.shape == (1, 2)
    assert summarize({"score": weights @ np.array([0.2, 0.4]) / 2}) == {"score": pytest.approx(0.3)}


def test_weighted_mean_skips_nan_rows() -> None:
    weights = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
    np.testing.assert_allclose(weighted_mean(weights, np.array([1.0, np.nan, 4.0])), [2.5, 2.0])


def test_summarize_reports_a_percentile_interval() -> None:
    values = np.concatenate([[0.5], np.linspace(0.0, 1.0, 1001)])
    summary = summarize({"m": values})
    assert summary["m"] == 0.5
    assert summary["m_lo"] == pytest.approx(0.025)
    assert summary["m_hi"] == pytest.approx(0.975)
    # The replicates are kept for paired comparisons.
    np.testing.assert_array_equal(summary["m_boot"], values[1:])
