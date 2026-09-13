"""A miniature copy of the datasets, for offline tests in any package.

`write_fake_dataset` writes the expression dataset's on-disk layout —
`samples`, `genes`, and one value config per quantification, with the real
schemas — plus a `tcga-patients-open`-style parquet per project carrying
`survival_derived` and `samples_ssgsea_hallmark`, all at toy size.
`use_fake_dataset` points `hub` and `labels` at it, so tests exercise the
real loading code without the network:

    @pytest.fixture
    def fake_dataset(tmp_path, monkeypatch):
        root = write_fake_dataset(tmp_path / "dataset")
        use_fake_dataset(monkeypatch, root)
        return root

Projects have distinct expression profiles, normals are shifted from
tumours, each case has a latent risk that both shifts its tumour's
expression and shortens its survival, and each sequencing plate adds a
small batch effect — so every probe has signal to find. Pathway scores
are a function of each sample's own expression, as ssGSEA's are. Barcodes
follow TCGA's layout, with five tissue source sites and a plate per eight
cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reimp_shared import hub, labels
from reimp_shared.splits import N_FOLDS

PROJECTS = ("TCGA-AAA", "TCGA-BBB", "TCGA-CCC")
GENE_TYPES = ("protein_coding", "lncRNA", "miRNA")
TUMOR, NORMAL = "Primary Tumor", "Solid Tissue Normal"
N_PATHWAYS, PATHWAY_SIZE = 6, 7
SURVIVAL_TYPE = pa.struct(
    [
        pa.field(f"{endpoint}_{part}", pa.int64() if part == "event" else pa.float64())
        for endpoint in labels.SURVIVAL_ENDPOINTS
        for part in ("event", "time")
    ]
)
SSGSEA_TYPE = pa.struct(
    [
        pa.field("sample_id", pa.string()),
        pa.field("aliquot_id", pa.string()),
        pa.field("source_file_id", pa.string()),
        pa.field("pathway", pa.list_(pa.string())),
        pa.field("pathway_url", pa.list_(pa.string())),
        pa.field("matched_gene_count", pa.list_(pa.int32())),
        pa.field("original_gene_count", pa.list_(pa.int32())),
        pa.field("score_raw", pa.list_(pa.float64())),
    ]
)


def write_fake_dataset(root: Path, n_cases: int = 60, n_genes: int = 48, seed: int = 0) -> Path:
    """Write the fake datasets under `root` and return `root`.

    Every case has a primary tumour and every third case a matched normal.
    Genes cycle through three biotypes; the last is a `_PAR_Y` copy of the
    first, all zero as in the real data. As between the real datasets, a
    few expression cases have no labels in the patients dataset.
    """
    rng = np.random.default_rng(seed)
    gene_ids = [f"ENSG{i:011d}.{1 + i % 4}" for i in range(n_genes - 1)]
    gene_ids.append(f"{gene_ids[0]}_PAR_Y")
    gene_types = [GENE_TYPES[i % len(GENE_TYPES)] for i in range(n_genes - 1)] + [GENE_TYPES[0]]
    genes = pa.table(
        {
            "gene_index": pa.array(range(n_genes), pa.int32()),
            "gene_id": gene_ids,
            "gene_name": [f"GENE{i}" for i in range(n_genes)],
            "gene_type": gene_types,
            "chromosome": ["chr1"] * (n_genes - 1) + ["chrY"],
            "start": pa.array([1000 * i for i in range(n_genes)], pa.int64()),
            "end": pa.array([1000 * i + 500 for i in range(n_genes)], pa.int64()),
        }
    )

    case_ids = [f"TCGA-T{c % 5}-{c:04d}" for c in range(n_cases)]
    case_project = [PROJECTS[c % len(PROJECTS)] for c in range(n_cases)]
    case_plate = [c // 8 for c in range(n_cases)]
    rows = []
    for c in range(n_cases):
        for sample_type in [TUMOR] + ([NORMAL] if c % 3 == 0 else []):
            rows.append((c, case_project[c], sample_type))
    n = len(rows)
    samples = pa.table(
        {
            "sample_index": pa.array(range(n), pa.int32()),
            "aliquot_id": [f"aliquot-{i}" for i in range(n)],
            "aliquot_submitter_id": [
                f"{case_ids[c]}-{'11A' if t == NORMAL else '01A'}-01R-A{case_plate[c]:03d}-07"
                for c, _, t in rows
            ],
            "case_id": [f"case-{c}" for c, _, _ in rows],
            "case_submitter_id": [case_ids[c] for c, _, _ in rows],
            "project_id": [p for _, p, _ in rows],
            "sample_type": [t for _, _, t in rows],
            "source_file_id": [f"file-{i}" for i in range(n)],
            "strand_balance": pa.array(rng.uniform(0.45, 0.55, n), pa.float32()),
            **{
                name: pa.array(rng.integers(10**5, 10**6, n), pa.int64())
                for name in labels.UNASSIGNED
            },
        }
    )

    risk = rng.normal(0.0, 1.0, n_cases)
    risk_genes = rng.normal(0.0, 0.5, n_genes)
    profile = rng.normal(0.0, 1.5, (len(PROJECTS), n_genes))
    normal_shift = rng.normal(0.0, 1.0, n_genes)
    plate_shift = rng.normal(0.0, 0.3, (max(case_plate) + 1, n_genes))
    logits = np.stack(
        [
            profile[PROJECTS.index(p)]
            + plate_shift[case_plate[c]]
            + (normal_shift if t == NORMAL else risk[c] * risk_genes)
            for c, p, t in rows
        ]
    )
    proportions = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    library = np.exp(rng.uniform(13, 15, n))
    counts = rng.poisson(library[:, None] * proportions).astype(np.int32)
    counts[:, -1] = 0
    first = rng.binomial(counts, 0.5).astype(np.int32)
    length_kb = rng.uniform(0.5, 5.0, n_genes)
    rate = counts / length_kb
    tpm = rate / rate.sum(axis=1, keepdims=True) * 1e6
    fpkm = rate / counts.sum(axis=1, keepdims=True) * 1e6
    upper_quartile = np.percentile(np.where(counts > 0, counts, np.nan), 75, axis=1)
    values = {
        "unstranded": counts,
        "stranded_first": first,
        "stranded_second": counts - first,
        "tpm_unstranded": tpm,
        "fpkm_unstranded": fpkm,
        "fpkm_uq_unstranded": rate / upper_quartile[:, None] * 1e6,
    }

    _write(root / "genes", genes)
    _write(root / "samples", samples)
    inline = samples.select(
        ["sample_index", "aliquot_id", "case_submitter_id", "project_id", "sample_type"]
    )
    for name, matrix in values.items():
        dtype = pa.int32() if name in hub.COUNTS else pa.float32()
        flat = pa.array(matrix.astype(dtype.to_pandas_dtype()).ravel(), dtype)
        offsets = pa.array(np.arange(0, len(flat) + 1, n_genes), pa.int32())
        _write(root / name, inline.append_column("values", pa.ListArray.from_arrays(offsets, flat)))

    ssgsea = _pathway_records(samples, np.log1p(tpm))
    survival = [_survival(rng, risk[c], case_project[c]) for c in range(n_cases)]
    survival[1] = None  # labels undefined for this case
    for project in PROJECTS:
        # The last case is missing entirely, like cases newer than the
        # patients dataset's GDC release.
        members = [c for c in range(n_cases - 1) if case_project[c] == project]
        table = pa.table(
            {
                "case_submitter_id": [case_ids[c] for c in members],
                "project_id": [project] * len(members),
                "survival_derived": pa.array([survival[c] for c in members], SURVIVAL_TYPE),
                "samples_ssgsea_hallmark": pa.array(
                    [ssgsea[case_ids[c]] for c in members], pa.list_(SSGSEA_TYPE)
                ),
            }
        )
        _write(root / "patients" / project, table)
    return root


def _pathway_records(samples: pa.Table, log_tpm: np.ndarray) -> dict[str, list[dict]]:
    """Per case, one ssGSEA-shaped record per aliquot.

    Pathway j is genes j·7 .. j·7+6; its score is their mean log TPM
    relative to the sample's overall mean — a function of the sample alone.
    """
    names = [f"HALLMARK_FAKE_{j}" for j in range(N_PATHWAYS)]
    members = [np.arange(j * PATHWAY_SIZE, (j + 1) * PATHWAY_SIZE) for j in range(N_PATHWAYS)]
    scores = np.stack([log_tpm[:, m].mean(axis=1) for m in members], axis=1)
    scores -= log_tpm.mean(axis=1, keepdims=True)
    records: dict[str, list[dict]] = {}
    for i, (case, aliquot) in enumerate(
        zip(
            samples["case_submitter_id"].to_pylist(), samples["aliquot_id"].to_pylist(), strict=True
        )
    ):
        records.setdefault(case, []).append(
            {
                "sample_id": f"sample-{i}",
                "aliquot_id": aliquot,
                "source_file_id": f"file-{i}",
                "pathway": names,
                "pathway_url": [f"https://example.org/{name}" for name in names],
                "matched_gene_count": [PATHWAY_SIZE] * N_PATHWAYS,
                "original_gene_count": [PATHWAY_SIZE] * N_PATHWAYS,
                "score_raw": scores[i].tolist(),
            }
        )
    return records


def _survival(rng: np.random.Generator, risk: float, project: str) -> dict:
    """Endpoints for one case: higher risk, earlier events. No DFI in the last project."""
    record = {}
    for endpoint in labels.SURVIVAL_ENDPOINTS:
        event_time = rng.exponential(1000 * np.exp(-risk))
        censor_time = rng.uniform(100, 3000)
        record[f"{endpoint}_event"] = int(event_time <= censor_time)
        record[f"{endpoint}_time"] = float(min(event_time, censor_time))
    if project == PROJECTS[-1]:
        record["dfi_event"] = record["dfi_time"] = None
    return record


def _write(directory: Path, table: pa.Table) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "data.parquet", row_group_size=16)


def every_fold(
    sample_index: np.ndarray, embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long-form cross-validation embeddings in which every fold's model embeds alike.

    Returns `(sample_index, embeddings, fold)`, as `read_embeddings` would
    from a directory of one identical file per fold.
    """
    n = len(sample_index)
    return (
        np.tile(sample_index, N_FOLDS),
        np.tile(embeddings, (N_FOLDS, 1)),
        np.repeat(np.arange(N_FOLDS), n),
    )


def use_fake_dataset(monkeypatch, root: Path) -> None:
    """Make `hub` and `labels` read from `root` for the rest of the test."""
    monkeypatch.setattr(
        hub, "parquet_path", lambda config, revision=None: root / config / "data.parquet"
    )
    patients = sorted(str(p) for p in (root / "patients").glob("TCGA-*/data.parquet"))
    monkeypatch.setattr(labels, "_resolve", lambda revision: "fake")
    monkeypatch.setattr(labels, "_patient_parquets", lambda commit: (patients, None))
    monkeypatch.setenv("REIMP_CACHE", str(root / "cache"))
