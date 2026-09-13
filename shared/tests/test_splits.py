import numpy as np
import pandas as pd
import pytest

from reimp_shared import hub
from reimp_shared.splits import (
    FOLD_WIDTH,
    N_BUCKETS,
    N_FOLDS,
    SPLITS,
    VAL_BUCKETS,
    case_buckets,
    case_hash,
    fold_layout,
    rank_block,
    sample_buckets,
    sample_folds,
    split_indices,
    split_samples,
    summarize,
    summarize_folds,
)

# Real TCGA cases.
CASES = ["TCGA-OR-A5JP", "TCGA-OR-A5KX", "TCGA-05-4244", "TCGA-A7-A0CE", "TCGA-ZX-AA5X"]


def _cohort(sizes: dict[str, int], aliquots: int = 1) -> pd.DataFrame:
    """Projects of the given sizes, every case with `aliquots` samples."""
    rows = [
        (f"{project}-{i:05d}", project)
        for project, n in sizes.items()
        for i in range(n)
        for _ in range(aliquots)
    ]
    samples = pd.DataFrame(rows, columns=["case_submitter_id", "project_id"])
    samples.insert(0, "sample_index", np.arange(len(samples)))
    return samples


def _roles(buckets) -> np.ndarray:
    """A bucket's role in every fold, as one number: its test fold and whether it validates."""
    buckets = np.asarray(buckets)
    return 2 * (buckets // FOLD_WIDTH) + (buckets % FOLD_WIDTH >= FOLD_WIDTH - VAL_BUCKETS)


# ---------- step 1: patient to bucket ----------


def test_pinned_hashes() -> None:
    """Changing the hash or the salt handling would silently reshuffle every fold."""
    assert [case_hash(case) for case in CASES] == [
        9474772820511054697,
        826251701267784183,
        13214882527886800590,
        14860863071082929464,
        602088037059829958,
    ]


@pytest.mark.network
def test_pinned_buckets_on_the_published_dataset() -> None:
    """The real cohort's buckets for real cases, which also pins the ranking and spreading."""
    buckets = case_buckets(hub.load_samples())
    assert [int(buckets[case]) for case in CASES] == [57, 3, 70, 64, 18]


def test_rank_block_deals_ranks_evenly() -> None:
    assert rank_block(np.arange(5), 5).tolist() == [0, 1, 2, 3, 4]
    assert rank_block(np.arange(7), 7).tolist() == [0, 1, 1, 2, 3, 3, 4]
    assert rank_block(np.arange(20), 20).tolist() == [r // 4 for r in range(20)]


@pytest.mark.parametrize("n", [7, 20, 36, 99, 250, 1231])
def test_every_project_splits_evenly_into_test_folds(n) -> None:
    samples = _cohort({"P": n, "Q": 60})
    in_p = (samples["project_id"] == "P").to_numpy()
    counts = np.bincount(sample_folds(samples)[in_p], minlength=N_FOLDS)
    assert counts.max() - counts.min() <= 1


def test_the_slot_within_a_block_is_the_hash_alone() -> None:
    samples = _cohort({"P": 300, "Q": 50})
    buckets = case_buckets(samples)
    slots = [case_hash(case) % FOLD_WIDTH for case in buckets.index]
    np.testing.assert_array_equal(buckets.to_numpy() % FOLD_WIDTH, slots)


def test_split_sizes_follow_the_layout() -> None:
    shares = split_samples(_cohort({"P": 5000, "Q": 5000}), 1).value_counts(normalize=True)
    assert shares["test"] == pytest.approx(0.2, abs=1e-3)
    # Val is 10% of the non-test blocks by the hash: 8% on average.
    assert shares["val"] == pytest.approx(0.08, abs=0.01)
    assert shares["train"] == pytest.approx(0.72, abs=0.01)


def test_small_projects_still_get_validation_patients() -> None:
    """Twenty-patient projects: evenly spaced ranks never alias away the val buckets."""
    samples = _cohort({f"P{i}": 20 for i in range(10)})
    for k in range(N_FOLDS):
        assert (split_samples(samples, k) == "val").mean() == pytest.approx(0.08, abs=0.04)


def test_buckets_ignore_row_order_and_aliquots() -> None:
    samples = _cohort({"P": 120, "Q": 45}, aliquots=2)
    buckets = case_buckets(samples).sort_index()
    shuffled = case_buckets(samples.sample(frac=1, random_state=0)).sort_index()
    one_each = case_buckets(samples.drop_duplicates("case_submitter_id")).sort_index()
    assert buckets.equals(shuffled) and buckets.equals(one_each)
    per_sample = sample_buckets(samples)
    assert per_sample.index.equals(samples.index)
    assert (per_sample.groupby(samples["case_submitter_id"]).nunique() == 1).all()


def test_other_projects_do_not_move_a_project() -> None:
    alone = case_buckets(_cohort({"P": 80}))
    together = case_buckets(_cohort({"P": 80, "Q": 500}))
    assert alone.equals(together.loc[alone.index])


def test_dropping_a_patient_moves_few_others() -> None:
    """Only patients whose rank crosses a block boundary change role, one per boundary."""
    samples = _cohort({"P": 500})
    before = case_buckets(samples)
    for dropped in samples["case_submitter_id"].iloc[::25]:
        after = case_buckets(samples[samples["case_submitter_id"] != dropped])
        moved = (_roles(after) != _roles(before.loc[after.index])).sum()
        assert moved <= N_FOLDS - 1


def test_salt_draws_independent_folds() -> None:
    samples = _cohort({"P": 5000})
    agree = (sample_folds(samples) == sample_folds(samples, salt="other")).mean()
    assert agree == pytest.approx(1 / N_FOLDS, abs=0.03)


def test_a_case_in_two_projects_raises() -> None:
    samples = _cohort({"P": 3}, aliquots=2)
    samples.loc[0, "project_id"] = "Q"
    with pytest.raises(ValueError, match="more than one project"):
        case_buckets(samples)


# ---------- step 2: bucket to fold ----------


@pytest.mark.parametrize("fold", range(N_FOLDS))
def test_fold_layout_is_72_8_20(fold) -> None:
    splits = pd.Series(fold_layout(fold))
    assert splits.value_counts().to_dict() == {"train": 72, "val": 8, "test": 20}
    buckets = np.arange(N_BUCKETS)
    np.testing.assert_array_equal(buckets[splits == "test"] // FOLD_WIDTH, fold)
    # Validation draws the same share from each of the other folds.
    val_folds = buckets[splits == "val"] // FOLD_WIDTH
    assert sorted(set(val_folds.tolist())) == sorted(set(range(N_FOLDS)) - {fold})
    assert np.bincount(val_folds, minlength=N_FOLDS)[val_folds].tolist() == [VAL_BUCKETS] * 8


def test_every_case_is_a_test_case_in_exactly_one_fold() -> None:
    samples = _cohort({"P": 300, "Q": 77})
    tests = np.stack([split_samples(samples, k).to_numpy() == "test" for k in range(N_FOLDS)])
    assert (tests.sum(axis=0) == 1).all()
    np.testing.assert_array_equal(tests.argmax(axis=0), sample_folds(samples).to_numpy())


@pytest.mark.parametrize("fold", [-1, N_FOLDS, 1.5])
def test_bad_folds_raise(fold) -> None:
    with pytest.raises(ValueError, match="fold must be"):
        split_samples(_cohort({"P": 10}), fold)


# ---------- on the fake dataset ----------


def test_split_indices_partition_the_samples(fake_dataset) -> None:
    samples = hub.load_samples()
    indices = split_indices(samples, 2)
    assert list(indices) == list(SPLITS)
    everything = np.concatenate(list(indices.values()))
    assert sorted(everything.tolist()) == samples["sample_index"].tolist()


def test_summaries_count_every_sample(fake_dataset) -> None:
    samples = hub.load_samples()
    table = summarize(samples, 3)
    assert list(table.columns) == [*SPLITS, "total"]
    assert table["total"].sum() == len(samples)
    assert (table[list(SPLITS)].sum(axis=1) == table["total"]).all()
    folds = summarize_folds(samples)
    assert list(folds.columns) == [*(f"fold{k}" for k in range(N_FOLDS)), "total"]
    assert folds["total"].sum() == len(samples)
    assert (table["test"] == folds["fold3"]).all()
