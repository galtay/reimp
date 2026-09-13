"""Gene selection and value transforms shared by every model and baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch

DEFAULT_LIBRARY_SIZE = 1e5
DEFAULT_GENE_TYPES: tuple[str, ...] = ("protein_coding",)

Transform = Literal["none", "log1p", "lognorm"]
TRANSFORMS: tuple[str, ...] = ("none", "log1p", "lognorm")


def log_normalize(counts, library_size: float = DEFAULT_LIBRARY_SIZE):
    """Scale each row to sum to `library_size`, then take log1p.

    TxFM's preprocessing (paper Eqs. 3-4) and the (Lib+Log)Norm baseline the
    paper benchmarks against. The library is the row sum over the columns
    passed in, so normalizing after gene selection gives every sample a
    library of `library_size` over the selected genes.

    Takes a numpy array or a torch tensor and returns the same kind, float32.
    A row that sums to zero stays zero rather than becoming NaN.
    """
    if isinstance(counts, torch.Tensor):
        x = counts.to(torch.float32)
        lib = x.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.log1p(x * (library_size / lib))
    x = np.asarray(counts, dtype=np.float32)
    lib = np.maximum(x.sum(axis=-1, keepdims=True, dtype=np.float64), 1.0)
    return np.log1p(x * (library_size / lib).astype(np.float32))


def transform_values(
    values: np.ndarray,
    transform: Transform = "none",
    library_size: float = DEFAULT_LIBRARY_SIZE,
) -> np.ndarray:
    """Apply one of `TRANSFORMS` to a samples x genes matrix.

    - `none`     values as stored (int32 counts or float32 TPM / FPKM)
    - `log1p`    log(1 + x), float32 — e.g. log TPM
    - `lognorm`  `log_normalize` with `library_size`, float32 — e.g. TxFM
    """
    if transform == "none":
        return values
    if transform == "log1p":
        return np.log1p(values.astype(np.float32, copy=False))
    if transform == "lognorm":
        return log_normalize(values, library_size)
    raise ValueError(f"unknown transform {transform!r}; expected one of {TRANSFORMS}")


def select_genes(
    genes: pd.DataFrame,
    gene_types: Sequence[str] | None = DEFAULT_GENE_TYPES,
    gene_ids: Sequence[str] | None = None,
    drop_par_y: bool = True,
) -> np.ndarray:
    """Column positions (`gene_index`) of the genes to keep.

    - `gene_types` keeps GENCODE biotypes, in array order: protein-coding
      genes by default, every gene with None.
    - `gene_ids`, when given, overrides `gene_types` and keeps exactly these
      genes in the order given, so a model with a fixed vocabulary gets its
      columns in vocabulary order. A versioned ID (`ENSG00000141510.18`)
      must match exactly; an unversioned one (`ENSG00000141510`) matches
      that gene whatever its version.

    `drop_par_y` removes the `_PAR_Y` rows from a biotype selection — the
    Y-chromosome copies of pseudoautosomal genes, which GDC's pipeline
    quantifies as zero in every sample. An unversioned ID never matches one.
    """
    gene_id = genes["gene_id"]
    par_y = gene_id.str.endswith("_PAR_Y").to_numpy()
    if gene_ids is not None:
        return _by_id(genes, list(gene_ids), par_y)

    if isinstance(gene_types, str):
        gene_types = [gene_types]
    keep = np.ones(len(genes), dtype=bool)
    if gene_types is not None:
        keep &= genes["gene_type"].isin(list(gene_types)).to_numpy()
    if drop_par_y:
        keep &= ~par_y
    if not keep.any():
        raise ValueError(f"no genes left after selecting gene_types={gene_types}")
    return genes["gene_index"].to_numpy()[keep]


def _by_id(genes: pd.DataFrame, ids: list[str], par_y: np.ndarray) -> np.ndarray:
    if len(set(ids)) != len(ids):
        raise ValueError("gene_ids contains duplicates")
    index = genes["gene_index"].to_numpy()
    exact = dict(zip(genes["gene_id"], index, strict=True))
    unversioned = genes["gene_id"][~par_y].str.split(".").str[0]
    base = dict(zip(unversioned, index[~par_y], strict=True))
    lookup = [exact.get(g) if "." in g else base.get(g) for g in ids]
    missing = [g for g, i in zip(ids, lookup, strict=True) if i is None]
    if missing:
        raise ValueError(f"{len(missing)} gene_ids not in the dataset, e.g. {missing[:5]}")
    return np.array(lookup, dtype=index.dtype)


def read_gene_ids(path: str) -> list[str]:
    """Gene IDs from a text file: one per line, `#` comments and blanks ignored."""
    with open(path) as f:
        lines = (line.split("#", 1)[0].strip() for line in f)
        return [line for line in lines if line]
