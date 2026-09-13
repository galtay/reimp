"""Labels for evaluation: patient endpoints, pathway scores, technical covariates.

Patient-level labels are keyed by `case_submitter_id`, the key the
expression dataset's samples carry and the shared split is drawn on;
pathway scores are keyed by `aliquot_id`. Both come from
`gabrielaltay/tcga-patients-open`, which holds the full GDC case tree for
every patient. Its rows also carry every molecular modality (17.7 GB over
33 projects), so only the label columns are read, remotely, and the result
is cached under `$REIMP_CACHE` (default `~/.cache/reimp`) per dataset
commit. The first read takes minutes; later ones are instant.

That dataset is built from an earlier GDC release than the expression
dataset, so a few expression cases have no labels; evaluations inner-join
and report how many they scored. Once a label is validated here, the plan
is to publish it with the expression dataset and read it from there
instead.

Technical covariates are keyed by `sample_index` and derived from the
expression dataset itself: its barcodes and STAR's read tallies.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from reimp_shared import hub

PATIENTS_REPO = "gabrielaltay/tcga-patients-open"
SURVIVAL_ENDPOINTS = ("os", "dss", "pfi", "dfi")
SSGSEA_COLLECTIONS = ("hallmark", "reactome", "pid", "oncogenic", "cancer_cell_atlas")
UNASSIGNED = ("n_unmapped", "n_multimapping", "n_nofeature", "n_ambiguous")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def cache_dir() -> Path:
    return Path(os.environ.get("REIMP_CACHE", Path.home() / ".cache" / "reimp"))


def commit(revision: str | None = None) -> str:
    """The `tcga-patients-open` commit the labels are read at."""
    return _resolve(revision)


def _resolve(revision: str | None) -> str:
    """The commit a revision points at. A full commit hash needs no network."""
    if revision is not None and _COMMIT.fullmatch(revision):
        return revision
    from huggingface_hub import HfApi

    return HfApi().dataset_info(PATIENTS_REPO, revision=revision).sha


def _patient_parquets(commit: str) -> tuple[list[str], object]:
    """Every project's parquet at `commit`, and the filesystem to read them from."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return sorted(fs.glob(f"datasets/{PATIENTS_REPO}@{commit}/TCGA-*/data.parquet")), fs


def read_patient_columns(columns: list[str], revision: str | None = None) -> pa.Table:
    """Top-level `columns` for every patient in every project.

    Column projection fetches only those columns' bytes. Nested fields come
    back as structs and lists; `Table.flatten` expands the structs.
    """
    paths, fs = _patient_parquets(_resolve(revision))

    def read(path: str) -> pa.Table:
        return pq.read_table(path, columns=columns, filesystem=fs)

    with ThreadPoolExecutor(max_workers=8) as pool:
        tables = list(pool.map(read, paths))
    return pa.concat_tables(tables, promote_options="default")


def _cached(name: str, revision: str | None, build) -> pd.DataFrame:
    """A label table from the cache, building and caching it on first use."""
    commit = _resolve(revision)
    path = cache_dir() / PATIENTS_REPO.replace("/", "--") / commit / f"{name}.parquet"
    if not path.exists():
        table = build(commit)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        pq.write_table(table, partial)
        partial.replace(path)
    return pd.read_parquet(path)


def load_survival(revision: str | None = None) -> pd.DataFrame:
    """Survival endpoints per patient.

    Columns: `case_submitter_id`, `project_id`, and for each of OS, DSS, PFI
    and DFI an `<endpoint>_event` (1 event, 0 censored) and `<endpoint>_time`
    (days from diagnosis). These are tcga2hf's re-derivation of the Liu et
    al. 2018 TCGA-CDR endpoints from current GDC data. An endpoint is null
    where it is undefined — DFI throughout SKCM, THYM, UVM and LAML.
    """

    def build(commit: str) -> pa.Table:
        columns = ["case_submitter_id", "project_id", "survival_derived"]
        table = read_patient_columns(columns, commit).flatten()
        return table.rename_columns(
            [name.removeprefix("survival_derived.") for name in table.column_names]
        )

    return _cached("survival", revision, build)


def load_ssgsea(collection: str = "hallmark", revision: str | None = None) -> pd.DataFrame:
    """ssGSEA pathway scores per RNA-seq aliquot: `aliquot_id`, then one column per gene set.

    tcga2hf scores each aliquot's `tpm_unstranded` with Barbie et al.'s
    ssGSEA as GSVA implements it, on MSigDB 2026.1 gene sets. These are its
    `score_raw`: a property of the sample alone, independent of which other
    samples or gene sets were scored alongside, so nothing leaks across the
    split. (GSVA's normalized score divides by the range over the whole
    cohort, which is why tcga2hf does not publish it.)
    """
    if collection not in SSGSEA_COLLECTIONS:
        raise ValueError(f"unknown collection {collection!r}; expected one of {SSGSEA_COLLECTIONS}")

    def build(commit: str) -> pa.Table:
        column = f"samples_ssgsea_{collection}"
        records = pc.list_flatten(read_patient_columns([column], commit).column(column))
        records = records.combine_chunks()
        n = len(records)
        names = records.field("pathway")[0].as_py()
        pathways = records.field("pathway").flatten().to_numpy(zero_copy_only=False)
        if len(pathways) != n * len(names) or (pathways.reshape(n, -1) != names).any():
            raise ValueError(f"{collection}: aliquots were scored on different pathway lists")
        scores = records.field("score_raw").flatten().to_numpy().reshape(n, -1)
        table = pa.table({"aliquot_id": records.field("aliquot_id")})
        for j, name in enumerate(names):
            table = table.append_column(name, pa.array(scores[:, j]))
        return table

    scores = _cached(f"ssgsea_{collection}", revision, build)
    return scores.drop_duplicates("aliquot_id").reset_index(drop=True)


def technical_covariates(revision: str | None = None) -> pd.DataFrame:
    """How each sample was collected and sequenced, one row per `sample_index`.

    - `tss`: tissue source site, the second field of the case barcode
      (`TCGA-OR-A5JP` → `OR`).
    - `plate`: the aliquot's plate, the sixth field of the aliquot barcode
      (`TCGA-OR-A5JP-01A-11R-A29S-07` → `A29S`).
    - `log_reads`: log10 of the library's total reads — those STAR assigned
      to genes plus its four unassigned tallies.
    - `assigned_fraction`: the share of those reads assigned to genes.
    - `strand_balance`: as published in `samples`.
    """
    samples = hub.load_samples(revision)
    assigned = hub.load_values("unstranded", revision).sum(axis=1, dtype=np.int64)
    total = assigned + samples[list(UNASSIGNED)].sum(axis=1).to_numpy()
    return pd.DataFrame(
        {
            "sample_index": samples["sample_index"],
            "tss": samples["case_submitter_id"].str.split("-").str[1],
            "plate": samples["aliquot_submitter_id"].str.split("-").str[5],
            "log_reads": np.log10(total),
            "assigned_fraction": assigned / total,
            "strand_balance": samples["strand_balance"].astype(float),
        }
    )
