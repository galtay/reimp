"""Which rows a probe fits on and which it scores, fold by fold.

Probes take embeddings in long form — `sample_index`, `embeddings` and
`fold` — where row r is sample `sample_index[r]` as embedded by the model
trained for fold `fold[r]` (see `read_embeddings`), together with the
dataset's `samples` table, the cohort the folds are drawn over. Each fold
present is one fit: on the fold's train and val samples, in that fold's
embedding space, predicting the fold's test samples. The folds' test
predictions are pooled and scored once, so every scored sample is scored by
an embedding model and a probe that never saw its patient. With all five
folds that is every sample, named `cv`; fewer folds — one, for a quick
train / val / test report — are named for the folds pooled, `fold2` or
`fold0+3`.

Val is for the embedding model's early stopping. A probe picks its own
hyperparameters by cross-validation within its fitting rows, so it fits on
train and val together.

Folds' models embed into different spaces, so a score that compares two
samples' predictions — a C-index's pairs — compares them within a fold
only, and one that describes a whole space — a spectrum, a clustering — is
taken per fold and averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reimp_shared.splits import DEFAULT_SALT, FOLD_WIDTH, N_FOLDS, sample_buckets

CV = "cv"


@dataclass(frozen=True)
class Fit:
    """One fold's fit over the long rows `rows` (positions), all in that fold's space.

    `fit` and `test` are boolean masks aligned to `rows`: the fold's train
    and val samples, and its test samples.
    """

    fold: int
    rows: np.ndarray
    fit: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Plan:
    """A probe's long rows, its fits (one per fold), and `tested`: the rows some fold tests.

    `rows` holds each embedding's row of `samples`, in embedding order.
    """

    rows: pd.DataFrame
    fits: list[Fit]
    tested: np.ndarray

    @property
    def name(self) -> str:
        """`cv` for all five folds, else the folds pooled: `fold2`, `fold0+3`."""
        folds = [f.fold for f in self.fits]
        return CV if folds == list(range(N_FOLDS)) else "fold" + "+".join(map(str, folds))


def _check_unique(sample_index: pd.Series, where: str) -> None:
    repeated = sample_index[sample_index.duplicated()]
    if len(repeated):
        raise ValueError(
            f"{where} embed a sample more than once, e.g. sample_index {repeated.iloc[:5].tolist()}"
        )


def plan(
    samples: pd.DataFrame,
    sample_index: np.ndarray,
    fold: np.ndarray,
    salt: str = DEFAULT_SALT,
) -> Plan:
    """The fits a probe makes for embeddings of `sample_index` from each fold's model.

    `samples` is the whole cohort, which the folds are drawn over. Each fold
    present is fit; each may embed a sample at most once.
    """
    rows = samples.set_index("sample_index").loc[sample_index].reset_index()
    fold = np.asarray(fold)
    if len(fold) != len(rows):
        raise ValueError(f"fold has {len(fold)} entries for {len(rows)} embeddings")
    present = np.unique(fold).tolist()
    if not present or not set(present) <= set(range(N_FOLDS)):
        raise ValueError(f"folds must be among {list(range(N_FOLDS))}, got {present}")
    buckets = pd.Series(sample_buckets(samples, salt).to_numpy(), index=samples["sample_index"])
    test_fold = buckets.loc[sample_index].to_numpy() // FOLD_WIDTH
    fits = []
    tested = np.zeros(len(rows), bool)
    for k in present:
        positions = np.flatnonzero(fold == k)
        _check_unique(rows["sample_index"].iloc[positions], f"fold {k}'s embeddings")
        test = test_fold[positions] == k
        fits.append(Fit(int(k), positions, ~test, test))
        tested[positions[test]] = True
    return Plan(rows, fits, tested)
