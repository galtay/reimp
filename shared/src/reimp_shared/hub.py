"""Read the `tcga-gene-expression-quantification-open` dataset from the HF Hub.

The dataset splits an AnnData-style matrix across configs: `samples` is the
row axis, `genes` the column axis, and each quantification (`unstranded`,
`tpm_unstranded`, ...) is its own config with one row per sample whose
`values` list runs in `genes` order. Row `j` of a value config is `samples`
row `j`, so the three align by position with no join.

Files come through `hf_hub_download` and land in the standard HF cache. A
full value config decodes to numpy in ~2 s once downloaded, so that cached
parquet is the only cache. Pass a commit hash as `revision` to pin a run to
one build of the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "gabrielaltay/tcga-gene-expression-quantification-open"

# One value config per column of GDC's STAR-counts TSV, GDC's names kept.
# The first three are read counts (int32); the rest are per-library
# normalized measures (float32).
COUNTS = ("unstranded", "stranded_first", "stranded_second")
NORMALIZED = ("tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded")
QUANTIFICATIONS = COUNTS + NORMALIZED


def parquet_path(config: str, revision: str | None = None) -> Path:
    """Local path to one config's parquet, downloaded on first use."""
    path = hf_hub_download(
        REPO_ID, f"{config}/data.parquet", repo_type="dataset", revision=revision
    )
    return Path(path)


def commit(revision: str | None = None) -> str:
    """The dataset commit `revision` resolves to, as the HF cache records it."""
    # .../snapshots/<commit>/samples/data.parquet
    return parquet_path("samples", revision).parts[-3]


def load_samples(revision: str | None = None) -> pd.DataFrame:
    """The row axis: one aliquot per row, in value-config row order."""
    return pq.read_table(parquet_path("samples", revision)).to_pandas()


def load_genes(revision: str | None = None) -> pd.DataFrame:
    """The column axis: one GENCODE v36 gene per row, in `values` order."""
    return pq.read_table(parquet_path("genes", revision)).to_pandas()


def load_values(quantification: str = "unstranded", revision: str | None = None) -> np.ndarray:
    """The full `(n_samples, n_genes)` matrix of one quantification.

    Dtype is as stored: int32 for counts, float32 for normalized measures.
    """
    if quantification not in QUANTIFICATIONS:
        raise ValueError(
            f"unknown quantification {quantification!r}; expected one of {QUANTIFICATIONS}"
        )
    path = parquet_path(quantification, revision)
    table = pq.read_table(path, columns=["sample_index", "values"])
    sample_index = table.column("sample_index").to_numpy()
    if not np.array_equal(sample_index, np.arange(len(sample_index))):
        raise ValueError(f"{quantification}: rows are not in sample_index order")
    values = table.column("values").combine_chunks()
    lengths = np.diff(values.offsets.to_numpy())
    if len(lengths) and (lengths != lengths[0]).any():
        raise ValueError(f"{quantification}: rows have differing numbers of genes")
    return values.flatten().to_numpy().reshape(len(values), -1)
