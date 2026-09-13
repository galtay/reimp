"""Patient-level 5-fold cross-validation, stratified by project.

A case can contribute several aliquots — a tumour and its matched normal, a
metastasis, a re-sequenced library — and those are not independent samples.
Splitting by aliquot would put one patient on both sides of a split, so every
assignment here is made per case and shared by all of its aliquots.

Two steps take a patient to a fold:

1. Patient to bucket, one of 100 in 5 blocks of 20. Each project's patients
   are ranked by a salted blake2b hash of their case ID, and the patient
   ranked r of n goes to block floor(5 · (r + ½) / n): every project fills
   the blocks evenly, to within one patient. The hash then picks one of the
   block's 20 buckets: bucket = 20 · block + hash mod 20.
2. Bucket to fold. Bucket b is a test bucket of fold b // 20. In fold k,
   fold k's buckets are test, and of every other fold the last 2 buckets
   are val and the rest train: 72 train, 8 val, 20 test. Every patient is a
   test patient in exactly one fold; one fold on its own is an ordinary
   train / val / test split. Val is held out from fitting for choices such
   as when to stop, and is drawn from every training fold alike (a whole
   fold as val would leave 60% to train on).

The test folds are stratified by project exactly; val, 10% of each block by
the hash, only in expectation. There is no split file: the same dataset
gives the same folds on every machine, and a different salt draws
independent ones. A patient's block depends on the other patients in its
project, so buckets are always computed over the whole cohort — the
dataset's full `samples` table — never over a filtered selection. A release
of the dataset that adds or drops a patient moves only the few others whose
rank crosses a block boundary, and nobody's place within a block.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

N_BUCKETS = 100
DEFAULT_SALT = "reimp-v1"
CASE_KEY = "case_submitter_id"
PROJECT_KEY = "project_id"

N_FOLDS = 5
FOLD_WIDTH = N_BUCKETS // N_FOLDS
# Of each fold's buckets, the last VAL_BUCKETS validate whenever the fold is
# not the test fold: 8% of patients, while training keeps 72%.
VAL_BUCKETS = 2
SPLITS = ("train", "val", "test")


def check_fold(fold: int) -> None:
    """Raise unless `fold` is one of the folds."""
    if fold not in range(N_FOLDS):
        raise ValueError(f"fold must be one of {list(range(N_FOLDS))}, got {fold!r}")


def case_hash(case_id: str, salt: str = DEFAULT_SALT) -> int:
    """The case's salted 64-bit hash: it ranks the case in its project and picks its slot."""
    digest = hashlib.blake2b(f"{salt}:{case_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def rank_block(rank, n):
    """The block (test fold) of the patient ranked `rank` (from 0) of `n` in its project.

    floor(N_FOLDS · (rank + ½) / n), in integers; elementwise on arrays.
    """
    return N_FOLDS * (2 * np.asarray(rank) + 1) // (2 * np.asarray(n))


def fold_layout(fold: int) -> tuple[str, ...]:
    """Step 2: the split of each bucket in fold `fold`."""
    check_fold(fold)

    def split(bucket: int) -> str:
        if bucket // FOLD_WIDTH == fold:
            return "test"
        return "val" if bucket % FOLD_WIDTH >= FOLD_WIDTH - VAL_BUCKETS else "train"

    return tuple(split(bucket) for bucket in range(N_BUCKETS))


def case_buckets(samples: pd.DataFrame, salt: str = DEFAULT_SALT) -> pd.Series:
    """Step 1: every case's bucket, indexed by case. `samples` must be the whole cohort."""
    cases = samples[[CASE_KEY, PROJECT_KEY]].drop_duplicates()
    straddling = cases[CASE_KEY][cases[CASE_KEY].duplicated()]
    if len(straddling):
        raise ValueError(f"cases in more than one project, e.g. {straddling.iloc[:5].tolist()}")
    hashes = np.array([case_hash(case, salt) for case in cases[CASE_KEY]], dtype=np.uint64)
    # Case ID breaks the (vanishingly unlikely) tie between equal hashes.
    cases = cases.assign(hash=hashes).sort_values([PROJECT_KEY, "hash", CASE_KEY])
    projects = cases.groupby(PROJECT_KEY, sort=False)
    block = rank_block(
        projects.cumcount().to_numpy(), projects[CASE_KEY].transform("size").to_numpy()
    )
    slot = (cases["hash"].to_numpy() % np.uint64(FOLD_WIDTH)).astype(np.int64)
    return pd.Series(FOLD_WIDTH * block + slot, index=cases[CASE_KEY].to_numpy(), name="bucket")


def sample_buckets(samples: pd.DataFrame, salt: str = DEFAULT_SALT) -> pd.Series:
    """Every sample's bucket, via its case, aligned to `samples.index`."""
    buckets = samples[CASE_KEY].map(case_buckets(samples, salt))
    return buckets.astype(np.int64).rename("bucket")


def sample_folds(samples: pd.DataFrame, salt: str = DEFAULT_SALT) -> pd.Series:
    """The fold in which every sample is a test sample, aligned to `samples.index`."""
    return (sample_buckets(samples, salt) // FOLD_WIDTH).rename("fold")


def split_samples(samples: pd.DataFrame, fold: int, salt: str = DEFAULT_SALT) -> pd.Series:
    """Every sample's split in fold `fold`, aligned to `samples.index`."""
    layout = np.array(fold_layout(fold))
    buckets = sample_buckets(samples, salt).to_numpy()
    return pd.Series(layout[buckets], index=samples.index, name="split")


def split_indices(
    samples: pd.DataFrame, fold: int, salt: str = DEFAULT_SALT
) -> dict[str, np.ndarray]:
    """`sample_index` values per split of fold `fold` — row positions in every value config."""
    split = split_samples(samples, fold, salt)
    return {name: samples.loc[split == name, "sample_index"].to_numpy() for name in SPLITS}


def summarize(samples: pd.DataFrame, fold: int, salt: str = DEFAULT_SALT) -> pd.DataFrame:
    """Samples per project in each split of fold `fold`, with the project total."""
    split = split_samples(samples, fold, salt)
    table = pd.crosstab(samples[PROJECT_KEY], split).reindex(columns=list(SPLITS), fill_value=0)
    table["total"] = table.sum(axis=1)
    return table


def summarize_folds(samples: pd.DataFrame, salt: str = DEFAULT_SALT) -> pd.DataFrame:
    """Samples per project in each fold's test set, with the project total."""
    folds = sample_folds(samples, salt)
    table = pd.crosstab(samples[PROJECT_KEY], folds).reindex(columns=range(N_FOLDS), fill_value=0)
    table.columns = [f"fold{k}" for k in range(N_FOLDS)]
    table["total"] = table.sum(axis=1)
    return table
