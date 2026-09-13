"""The embeddings file every model writes and every evaluation reads.

A model is trained for one cross-validation fold and writes its
embeddings of every sample to one file, each row marked with that fold. A
directory of a model's fold files reads back as one table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reimp_shared.splits import check_fold


def write_embeddings(
    path: Path | str, sample_index: np.ndarray, embeddings: np.ndarray, fold: int
) -> Path:
    """Write `sample_index`, a fixed-length float32 `embedding` and `fold` per sample.

    `fold` is the cross-validation fold the embedding model was trained for.
    """
    check_fold(fold)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(sample_index):
        raise ValueError(
            f"embeddings must be (n, d) with n = len(sample_index); got {embeddings.shape} "
            f"for {len(sample_index)} samples"
        )
    flat = pa.array(embeddings.ravel(), type=pa.float32())
    table = pa.table(
        {
            "sample_index": pa.array(sample_index, type=pa.int32()),
            "embedding": pa.FixedSizeListArray.from_arrays(flat, embeddings.shape[1]),
            "fold": pa.array(np.full(len(sample_index), fold), type=pa.int8()),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_embeddings(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(sample_index, embeddings, fold)` from an embeddings parquet or a directory of them.

    A directory reads its `*.parquet` files as one table; anything else in it
    is ignored.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise ValueError(f"{path}: no .parquet files")
        table = pa.concat_tables([pq.read_table(f) for f in files])
    else:
        table = pq.read_table(path)
    if "fold" not in table.column_names:
        raise ValueError(f"{path}: no `fold` column; write embeddings with their model's fold")
    embedding = table.column("embedding").combine_chunks()
    values = embedding.flatten().to_numpy().reshape(len(embedding), -1)
    return table.column("sample_index").to_numpy(), values, table.column("fold").to_numpy()
