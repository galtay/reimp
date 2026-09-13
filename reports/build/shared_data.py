"""Build `reports/shared_data.html`: the shared cohort, its genes and its folds.

    uv run python reports/build/shared_data.py [--out PATH] [--revision COMMIT]

Every number on the page comes from the expression dataset through
`reimp_shared`, so the page is rebuilt, never edited by hand. The template
next to this script holds the prose and the drawing code; this script fills
the prose's `{{name}}` placeholders and embeds the chart data as JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from reportkit import render
from scipy.stats import chi2_contingency

from reimp_shared import hub
from reimp_shared.labels import UNASSIGNED
from reimp_shared.preprocess import DEFAULT_LIBRARY_SIZE, select_genes
from reimp_shared.splits import (
    CASE_KEY,
    DEFAULT_SALT,
    FOLD_WIDTH,
    N_BUCKETS,
    N_FOLDS,
    SPLITS,
    VAL_BUCKETS,
    case_buckets,
    case_hash,
    fold_layout,
    rank_block,
    sample_folds,
    split_samples,
)

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "shared_data.template.html"
OUT = HERE.parent / "shared_data.html"

NORMAL = "Solid Tissue Normal"
# Real cases whose assignments shared/tests/test_splits.py pins.
EXAMPLE_CASES = ("TCGA-OR-A5JP", "TCGA-OR-A5KX", "TCGA-05-4244", "TCGA-A7-A0CE", "TCGA-ZX-AA5X")
TOP_BIOTYPES = 9
# A biotype outside the largest few is called out if it carries this share of reads.
NOTABLE_READ_SHARE = 0.005
MEASURES = {
    "unstranded": "reads assigned to the gene, on either strand",
    "stranded_first": "reads assigned on the first strand",
    "stranded_second": "reads assigned on the second strand",
    "tpm_unstranded": "transcripts per million: counts over gene length, scaled to a million",
    "fpkm_unstranded": "fragments per kilobase of gene per million mapped",
    "fpkm_uq_unstranded": "FPKM with each library scaled by its upper-quartile gene",
}
PROJECT_NAMES = {
    "ACC": "Adrenocortical carcinoma",
    "BLCA": "Bladder urothelial carcinoma",
    "BRCA": "Breast invasive carcinoma",
    "CESC": "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon adenocarcinoma",
    "DLBC": "Diffuse large B-cell lymphoma",
    "ESCA": "Esophageal carcinoma",
    "GBM": "Glioblastoma multiforme",
    "HNSC": "Head and neck squamous cell carcinoma",
    "KICH": "Kidney chromophobe",
    "KIRC": "Kidney renal clear cell carcinoma",
    "KIRP": "Kidney renal papillary cell carcinoma",
    "LAML": "Acute myeloid leukemia",
    "LGG": "Brain lower grade glioma",
    "LIHC": "Liver hepatocellular carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian serous cystadenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma",
    "PCPG": "Pheochromocytoma and paraganglioma",
    "PRAD": "Prostate adenocarcinoma",
    "READ": "Rectum adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin cutaneous melanoma",
    "STAD": "Stomach adenocarcinoma",
    "TGCT": "Testicular germ cell tumors",
    "THCA": "Thyroid carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine corpus endometrial carcinoma",
    "UCS": "Uterine carcinosarcoma",
    "UVM": "Uveal melanoma",
}


def _int(n) -> str:
    return f"{int(round(n)):,}"


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _millions(n: float) -> str:
    return f"{n / 1e6:.1f}M"


def _code(project: str) -> str:
    return project.removeprefix("TCGA-")


def _bins(values: np.ndarray, step: float) -> np.ndarray:
    """Histogram edges `step` apart, aligned to multiples of `step`, covering `values`."""
    lo = np.floor(values.min() / step) * step
    hi = max(np.ceil(values.max() / step) * step, lo + step)
    return np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)


def _histogram(values: np.ndarray, edges: np.ndarray) -> dict:
    counts, edges = np.histogram(values, bins=edges)
    return {
        "edges": [round(float(e), 6) for e in edges],
        "counts": counts.tolist(),
        "median": float(np.median(values)),
    }


def _chi2(table: np.ndarray) -> dict | None:
    """χ² test of independence, or None where the table cannot support one."""
    if min(table.shape) < 2:
        return None
    result = chi2_contingency(table)
    return {"stat": float(result.statistic), "dof": int(result.dof), "p": float(result.pvalue)}


def _moves_on_drop(n: int) -> np.ndarray:
    """For a project of `n` patients: how many others change test fold as each is dropped.

    Dropping a patient shifts the ranks after it down by one and shrinks the
    project to n - 1. A patient's slot within its block comes from its hash
    alone, so its test fold is the only thing that can move.
    """
    ranks = np.arange(n)
    before = rank_block(ranks, n)
    moves = np.empty(n, dtype=int)
    for dropped in range(n):
        rest = ranks[ranks != dropped]
        after = rank_block(rest - (rest > dropped), n - 1)
        moves[dropped] = int((after != before[rest]).sum())
    return moves


def collect(revision: str | None = None, salt: str = DEFAULT_SALT) -> tuple[dict, dict[str, str]]:
    """The chart data, and the prose's numbers as display strings."""
    samples = hub.load_samples(revision)
    genes = hub.load_genes(revision)
    counts = hub.load_values("unstranded", revision)
    commit = hub.commit(revision)
    protein = select_genes(genes)
    cases = samples.drop_duplicates(CASE_KEY)
    normal = samples["sample_type"].eq(NORMAL)

    # ---- cohort ----
    projects = (
        samples.assign(normal=normal)
        .groupby("project_id")
        .agg(samples=(CASE_KEY, "size"), cases=(CASE_KEY, "nunique"), normal=("normal", "sum"))
        .sort_values("samples", ascending=False)
    )
    aliquots = samples.groupby(CASE_KEY).size()
    multi = aliquots[aliquots > 1]
    kinds = samples.groupby(CASE_KEY)["sample_type"].nunique()
    cohort = {
        "projects": [
            {
                "project": project,
                "code": _code(project),
                "name": PROJECT_NAMES.get(_code(project), ""),
                "samples": row.samples,
                "cases": row.cases,
                "normal": row.normal,
            }
            for project, row in projects.iterrows()
        ],
        "sample_types": [
            {"type": kind, "samples": n}
            for kind, n in samples["sample_type"].value_counts().items()
        ],
        "aliquots": [
            {"aliquots": k, "cases": n} for k, n in aliquots.value_counts().sort_index().items()
        ],
    }

    # ---- values ----
    assigned = counts.sum(axis=1, dtype=np.int64)
    total = assigned + samples[list(UNASSIGNED)].sum(axis=1).to_numpy()
    protein_reads = counts[:, protein].sum(axis=1, dtype=np.int64)
    fraction = assigned / total
    protein_share = protein_reads / assigned
    log_total = np.log10(total)
    values = {
        "quantifications": [
            {
                "name": name,
                "kind": "read count" if name in hub.COUNTS else "normalized",
                "dtype": "int32" if name in hub.COUNTS else "float32",
                "measures": MEASURES.get(name, ""),
            }
            for name in hub.QUANTIFICATIONS
        ],
        "reads": _histogram(log_total, _bins(log_total, 0.05)),
        "assigned": _histogram(fraction, _bins(fraction, 0.025)),
        "protein_share": _histogram(protein_share, _bins(protein_share, 0.01)),
    }

    # ---- genes ----
    gene_types = genes["gene_type"]
    reads_per_gene = pd.Series(counts.sum(axis=0, dtype=np.int64), index=gene_types.to_numpy())
    biotypes = pd.DataFrame(
        {"genes": gene_types.value_counts(), "reads": reads_per_gene.groupby(level=0).sum()}
    )
    biotypes["share"] = biotypes["reads"] / biotypes["reads"].sum()
    biotypes = biotypes.sort_values("genes", ascending=False)
    top, rest = biotypes.iloc[:TOP_BIOTYPES], biotypes.iloc[TOP_BIOTYPES:]
    biotype_rows = [
        {
            "biotype": name,
            "genes": row["genes"],
            "read_share": row["share"],
            "selected": name == "protein_coding",
        }
        for name, row in top.iterrows()
    ]
    if len(rest):
        biotype_rows.append(
            {
                "biotype": f"{len(rest)} other biotypes",
                "genes": rest["genes"].sum(),
                "read_share": rest["share"].sum(),
                "selected": False,
            }
        )
    notable = rest[rest["share"] >= NOTABLE_READ_SHARE].sort_values("share", ascending=False)
    detected = counts[:, protein] > 0
    detection = detected.mean(axis=0)
    per_sample = detected.sum(axis=1)
    par_y = genes["gene_id"].str.endswith("_PAR_Y").to_numpy()
    coding = (gene_types == "protein_coding").to_numpy()
    genes_data = {
        "biotypes": biotype_rows,
        "detection": _histogram(detection, np.linspace(0.0, 1.0, 21)),
    }

    # ---- folds ----
    sizes = []
    for k in range(N_FOLDS):
        by_sample = split_samples(samples, k, salt).value_counts()
        by_case = split_samples(cases, k, salt).value_counts()
        sizes.append(
            {
                "fold": k,
                **{f"{s}_samples": int(by_sample.get(s, 0)) for s in SPLITS},
                **{f"{s}_cases": int(by_case.get(s, 0)) for s in SPLITS},
            }
        )
    buckets = case_buckets(samples, salt)
    present = set(cases[CASE_KEY])
    chosen = [c for c in EXAMPLE_CASES if c in present] or cases[CASE_KEY].head(5).tolist()
    examples = []
    for case in chosen:
        first = samples[samples[CASE_KEY] == case].iloc[0]
        bucket = int(buckets[case])
        examples.append(
            {
                "case": case,
                "project": first["project_id"],
                "barcode": first["aliquot_submitter_id"],
                "bucket": bucket,
                "test_fold": bucket // FOLD_WIDTH,
                "roles": [fold_layout(k)[bucket] for k in range(N_FOLDS)],
            }
        )
    example = examples[0]
    folds = {
        "layout": [list(fold_layout(k)) for k in range(N_FOLDS)],
        "fold_width": FOLD_WIDTH,
        "sizes": sizes,
        "examples": examples,
        "example_case": example["case"],
        "example_bucket": example["bucket"],
    }

    # ---- balance ----
    counts_by_fold = (
        pd.crosstab(cases["project_id"], sample_folds(cases, salt).to_numpy())
        .reindex(index=projects.index, columns=range(N_FOLDS), fill_value=0)
        .to_numpy()
    )
    n_cases = counts_by_fold.sum(axis=1, keepdims=True)
    max_off = np.abs(counts_by_fold - n_cases / N_FOLDS).max()
    moves = np.concatenate(
        [_moves_on_drop(int(n)) for n in n_cases[:, 0] if n > 1] or [np.zeros(1, dtype=int)]
    )
    by_type = pd.crosstab(samples["sample_type"], sample_folds(samples, salt).to_numpy())
    # Only types with an expected count of at least 5 in every fold.
    by_type = by_type[by_type.sum(axis=1) >= 5 * N_FOLDS]
    project_test, type_test = _chi2(counts_by_fold), _chi2(by_type.to_numpy())
    balance = {
        "projects": [
            {
                "code": _code(project),
                "name": PROJECT_NAMES.get(_code(project), ""),
                "cases": int(n_cases[i, 0]),
                "folds": counts_by_fold[i].tolist(),
            }
            for i, project in enumerate(projects.index)
        ]
    }

    # ---- the prose's numbers ----
    offset = example["bucket"] % FOLD_WIDTH
    val_start = FOLD_WIDTH - VAL_BUCKETS
    trains = offset < val_start
    notable_text = "; ".join(
        f"{name} ({_int(row['genes'])} genes, {_pct(row['share'])} of reads)"
        for name, row in notable.iterrows()
    )
    members = cases.loc[cases["project_id"] == example["project"], CASE_KEY]
    ranking = sorted(members, key=lambda case: (case_hash(case, salt), case))
    facts = {
        "repo": hub.REPO_ID,
        "commit": commit,
        "commit_short": commit[:7],
        "salt": salt,
        "samples": _int(len(samples)),
        "cases": _int(len(cases)),
        "projects": _int(len(projects)),
        "genes": _int(len(genes)),
        "biotypes": _int(gene_types.nunique()),
        "largest_project": _code(projects.index[0]),
        "largest_samples": _int(projects["samples"].iloc[0]),
        "smallest_project": _code(projects.index[-1]),
        "smallest_samples": _int(projects["samples"].iloc[-1]),
        "project_ratio": f"{projects['samples'].iloc[0] / projects['samples'].iloc[-1]:.0f}",
        "primary": _int((samples["sample_type"] == "Primary Tumor").sum()),
        "normals": _int(normal.sum()),
        "normal_projects": _int(samples.loc[normal, "project_id"].nunique()),
        "multi_cases": _int(len(multi)),
        "multi_samples": _int(multi.sum()),
        "multi_pct": _pct(multi.sum() / len(samples)),
        "mixed_cases": _int((kinds > 1).sum()),
        "reads_median": _millions(np.median(total)),
        "reads_q01": _millions(np.quantile(total, 0.01)),
        "reads_q99": _millions(np.quantile(total, 0.99)),
        "depth_ratio": f"{np.quantile(total, 0.99) / np.quantile(total, 0.01):.0f}",
        "assigned_median": _pct(np.median(fraction), 0),
        "assigned_q01": _pct(np.quantile(fraction, 0.01), 0),
        "protein_share_median": _pct(np.median(protein_share)),
        "protein_share_q01": _pct(np.quantile(protein_share, 0.01), 0),
        "library_size": _int(DEFAULT_LIBRARY_SIZE),
        "protein_coding": _int(coding.sum()),
        "par_y": _int(par_y.sum()),
        "par_y_protein": _int((par_y & coding).sum()),
        "selected": _int(len(protein)),
        "selected_gene_pct": _pct(len(protein) / len(genes)),
        "selected_read_pct": _pct(protein_reads.sum() / assigned.sum()),
        "notable": notable_text or "none",
        "always_detected": _int((detection == 1).sum()),
        "never_detected": _int((detection == 0).sum()),
        "detected_median": _int(np.median(per_sample)),
        "n_folds": str(N_FOLDS),
        "n_buckets": str(N_BUCKETS),
        "fold_width": str(FOLD_WIDTH),
        "val_buckets": str(VAL_BUCKETS),
        "val_start": str(val_start),
        "last_bucket": str(N_BUCKETS - 1),
        "other_folds": str(N_FOLDS - 1),
        "train_pct": str((N_FOLDS - 1) * val_start * 100 // N_BUCKETS),
        "val_pct": str((N_FOLDS - 1) * VAL_BUCKETS * 100 // N_BUCKETS),
        "test_pct": str(FOLD_WIDTH * 100 // N_BUCKETS),
        "whole_fold_train_pct": str((N_FOLDS - 2) * FOLD_WIDTH * 100 // N_BUCKETS),
        "ex_case": example["case"],
        "ex_project": example["project"],
        "ex_n": _int(len(ranking)),
        "ex_rank": str(ranking.index(example["case"])),
        "ex_hash": f"{case_hash(example['case'], salt):016x}",
        "ex_slot": str(case_hash(example["case"], salt) % FOLD_WIDTH),
        "ex_barcode": example["barcode"],
        "ex_bucket": str(example["bucket"]),
        "ex_fold": str(example["test_fold"]),
        "ex_offset": str(offset),
        "ex_offset_rule": (
            f"{offset} is below {val_start}, so the patient trains in the other {N_FOLDS - 1} folds"
            if trains
            else f"{offset} is {val_start} or above, so the patient validates in the other "
            f"{N_FOLDS - 1} folds"
        ),
        "max_off": f"{max_off:.1f}",
        "stability_max": _int(moves.max()),
        "stability_median": _int(np.median(moves)),
        "chi_projects_p": f"{project_test['p']:.2f}" if project_test else "n/a",
        "chi_projects_stat": f"{project_test['stat']:.1f}" if project_test else "n/a",
        "chi_projects_dof": str(project_test["dof"]) if project_test else "n/a",
        "chi_types_p": f"{type_test['p']:.2f}" if type_test else "n/a",
    }
    data = {
        "cohort": cohort,
        "values": values,
        "genes": genes_data,
        "folds": folds,
        "balance": balance,
    }
    return data, facts


def build(out: Path | str = OUT, revision: str | None = None) -> Path:
    """Write the report to `out` from the dataset at `revision`."""
    data, facts = collect(revision)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, facts, TEMPLATE.read_text()))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--revision", default=None, help="dataset commit to pin (default: main)")
    args = parser.parse_args(argv)
    print(f"wrote {build(args.out, args.revision)}")


if __name__ == "__main__":
    main()
