import numpy as np
import pandas as pd
import pytest

from reimp_shared.eval.protocol import CV, plan
from reimp_shared.splits import N_FOLDS, sample_folds
from reimp_shared.testing import every_fold


def _samples(n: int = 300) -> pd.DataFrame:
    """Two aliquots per case, cases spread over three projects."""
    case = np.arange(n) // 2
    return pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "case_submitter_id": [f"case-{c}" for c in case],
            "project_id": [f"P{c % 3}" for c in case],
        }
    )


def _long(samples: pd.DataFrame, folds=range(N_FOLDS)) -> tuple[np.ndarray, np.ndarray]:
    """`sample_index` and `fold` for every sample embedded by each of `folds`' models."""
    sample_index, _, fold = every_fold(
        samples["sample_index"].to_numpy(), np.zeros((len(samples), 1))
    )
    keep = np.isin(fold, list(folds))
    return sample_index[keep], fold[keep]


def test_each_fold_fits_on_its_train_and_val_and_tests_its_own_cases() -> None:
    samples = _samples()
    sample_index, fold = _long(samples)
    cv = plan(samples, sample_index, fold)
    assert [f.fold for f in cv.fits] == list(range(N_FOLDS))
    test_fold = sample_folds(samples).to_numpy()[sample_index]
    for f in cv.fits:
        assert (fold[f.rows] == f.fold).all()
        np.testing.assert_array_equal(f.test, test_fold[f.rows] == f.fold)
        np.testing.assert_array_equal(f.fit, ~f.test)


def test_rows_follow_the_embeddings() -> None:
    samples = _samples()
    cv = plan(samples, np.array([5, 0, 5]), np.array([0, 0, 1]))
    assert cv.rows["sample_index"].tolist() == [5, 0, 5]
    assert cv.rows["case_submitter_id"].tolist() == ["case-2", "case-0", "case-2"]


def test_all_five_folds_score_every_sample_once_as_cv() -> None:
    samples = _samples()
    sample_index, fold = _long(samples)
    cv = plan(samples, sample_index, fold)
    assert cv.name == CV
    assert sorted(sample_index[cv.tested].tolist()) == samples["sample_index"].tolist()


@pytest.mark.parametrize(("folds", "name"), [([2], "fold2"), ([0, 3], "fold0+3")])
def test_fewer_folds_score_their_own_test_cases(folds, name) -> None:
    samples = _samples()
    sample_index, fold = _long(samples, folds)
    cv = plan(samples, sample_index, fold)
    assert cv.name == name
    tested = sample_index[cv.tested]
    expected = samples["sample_index"][np.isin(sample_folds(samples), folds)]
    assert sorted(tested.tolist()) == sorted(expected.tolist())


def test_folds_come_from_the_cohort_whatever_is_embedded() -> None:
    """Embedding a subset moves nobody: ranks are taken over the whole cohort."""
    samples = _samples()
    sample_index, fold = _long(samples.iloc[::3])
    cv = plan(samples, sample_index, fold)
    cohort_fold = sample_folds(samples).to_numpy()
    np.testing.assert_array_equal(cv.tested, cohort_fold[sample_index] == fold)


def test_unknown_folds_raise() -> None:
    samples = _samples()
    sample_index, fold = _long(samples, [0])
    with pytest.raises(ValueError, match="folds must be among"):
        plan(samples, sample_index, fold + N_FOLDS)


def test_a_sample_embedded_twice_in_one_fold_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        plan(_samples(), np.array([0, 0]), np.array([1, 1]))


def test_every_embedding_needs_a_fold() -> None:
    with pytest.raises(ValueError, match="entries for"):
        plan(_samples(), np.arange(4), np.zeros(3, dtype=int))
