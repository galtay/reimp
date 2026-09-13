"""Differences between two embeddings' scores, with paired intervals.

Every probe resamples patients with the same seed, so two embeddings scored
on the same patients share their bootstrap draws: replicate r of each
resamples the same patients. Differencing their scores replicate by
replicate gives a paired interval, which is much narrower than the two
separate intervals suggest whenever the embeddings do well and badly on the
same patients — and it is the interval that says whether one is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reimp_shared.eval.bootstrap import BOOT, LEVEL, interval

# Columns that tell a probe's rows apart within one embeddings and split.
ROW_KEYS = ("task", "endpoint", "project")
# Columns counting who was scored; paired rows must agree on them.
COUNTS = ("n", "patients")


def _replicates(value) -> np.ndarray | None:
    return np.asarray(value, dtype=np.float64) if isinstance(value, list | np.ndarray) else None


def paired_differences(
    table: pd.DataFrame, embeddings: str, reference: str, level: float = LEVEL
) -> pd.DataFrame:
    """Each score of `embeddings` minus `reference`'s, with a paired bootstrap interval.

    `table` holds one probe's rows for both, `_boot` columns included, as
    the probes return them. Rows are matched on the split and the probe's
    key columns. Draws are only paired when both were scored on the same
    patients, so rows whose counts differ are skipped. One row per matched
    row and score: its keys, `score`, `delta`, `delta_lo`, `delta_hi`, and
    `share_above`, the share of replicates in which `embeddings` scores
    higher.
    """
    keys = ["split", *(k for k in ROW_KEYS if k in table)]
    counts = [c for c in COUNTS if c in table]
    scores = [c.removesuffix(BOOT) for c in table.columns if c.endswith(BOOT)]
    theirs = table[table["embeddings"] == reference]
    rows = []
    for _, mine in table[table["embeddings"] == embeddings].iterrows():
        match = theirs
        for key in keys:
            match = match[match[key] == mine[key]]
        if len(match) != 1:
            continue
        other = match.iloc[0]
        if any(mine[c] != other[c] for c in counts):
            continue
        for score in scores:
            a, b = _replicates(mine[f"{score}{BOOT}"]), _replicates(other[f"{score}{BOOT}"])
            if a is None or b is None or a.shape != b.shape:
                continue
            delta = a - b
            valid = ~np.isnan(delta)
            if not valid.any():
                continue
            low, high = interval(delta[valid], level)
            rows.append(
                {
                    **{key: mine[key] for key in keys},
                    "score": score,
                    "delta": float(mine[score] - other[score]),
                    "delta_lo": low,
                    "delta_hi": high,
                    "share_above": float((delta[valid] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)
