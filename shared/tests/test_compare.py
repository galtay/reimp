import numpy as np
import pandas as pd
import pytest

from reimp_shared.eval.bootstrap import bootstrap_weights, summarize
from reimp_shared.eval.compare import paired_differences


def _row(embeddings: str, task: str, n: int, replicates: np.ndarray) -> dict:
    return {"embeddings": embeddings, "split": "cv", "task": task, "n": n, **summarize(replicates)}


def test_shared_errors_cancel_in_the_paired_interval() -> None:
    """Wrong on the same patients: separate intervals overlap, the paired one excludes 0."""
    rng = np.random.default_rng(0)
    cases = np.arange(400)
    weights = bootstrap_weights(cases, n_bootstrap=500)
    hard = rng.random(400)  # per-patient difficulty, shared by both
    good = (hard < 0.80).astype(float)
    better = good.copy()
    better[np.flatnonzero(hard >= 0.80)[:12]] = 1.0  # right on 12 more patients
    table = pd.DataFrame(
        [
            _row("a", "t", 400, {"accuracy": weights @ better / 400}),
            _row("b", "t", 400, {"accuracy": weights @ good / 400}),
        ]
    )
    (row,) = paired_differences(table, "a", "b").to_dict(orient="records")
    assert row["delta"] == pytest.approx(12 / 400)
    assert 0 < row["delta_lo"] <= row["delta"] <= row["delta_hi"]
    assert row["share_above"] == 1.0
    a, b = table.iloc[0], table.iloc[1]
    assert a["accuracy_lo"] < b["accuracy_hi"]  # separately they overlap


def test_rows_match_on_their_keys_and_skip_different_patients() -> None:
    rep = {"accuracy": np.array([0.5, 0.4, 0.6])}
    table = pd.DataFrame(
        [
            _row("a", "t1", 10, rep),
            _row("b", "t1", 10, rep),
            _row("a", "t2", 10, rep),
            _row("b", "t2", 9, rep),  # scored on other patients: not paired
            _row("a", "t3", 10, rep),  # no reference row
        ]
    )
    diffs = paired_differences(table, "a", "b")
    assert diffs["task"].tolist() == ["t1"]
    assert diffs["delta"].item() == 0.0
