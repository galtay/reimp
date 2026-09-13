"""Build `reports/baselines.html`: every shared probe, run on the baselines.

    uv run python reports/build/baselines.py [--rerun] [--out PATH] [--revision COMMIT]

Two baselines need no model: `reimp-shared baseline-pca` at each of `DIMS`
dimensions, and `reimp-shared baseline-hvg`, the `HVG_GENES` most variable
genes as they are. Each is scored by `reimp-shared probe` pooled over the
five folds; the largest PCA also fold by fold, and again under each of
`SALTS`, which draws the folds anew. Embeddings and score files are kept in
`out/` (`out/pca<d>/`, `out/hvg<n>/`, `out/salts/`, `out/scores/`), and a
rebuild computes only what is missing or older than its inputs; `--rerun`
computes everything again. The page is filled from the score files: this
script fills the template's `{{name}}` placeholders and embeds the chart
data as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from reportkit import render

from reimp_shared import hub
from reimp_shared.cli import main as reimp_shared
from reimp_shared.eval.bootstrap import N_BOOTSTRAP
from reimp_shared.eval.classification import ORGAN_TASKS, TASKS, task_labels
from reimp_shared.eval.compare import paired_differences
from reimp_shared.eval.confounders import DEFAULT_K
from reimp_shared.eval.invertibility import DEFAULT_ALPHAS as RIDGE_ALPHAS
from reimp_shared.eval.survival import DEFAULT_ALPHAS as COX_ALPHAS
from reimp_shared.eval.survival import MACRO_MIN_EVENTS, PRIMARY_TUMOR
from reimp_shared.eval.survival import N_FOLDS as COX_FOLDS
from reimp_shared.preprocess import DEFAULT_LIBRARY_SIZE, select_genes
from reimp_shared.splits import CASE_KEY, FOLD_WIDTH, N_BUCKETS, N_FOLDS, VAL_BUCKETS

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "baselines.template.html"
OUT = HERE.parent / "baselines.html"
WORK = HERE.parents[1] / "out"
DIMS = (4, 16, 64, 256)
HVG_GENES = 5000
# Other salts: the same fold rule, drawing different patients into each fold.
SALTS = ("reimp-alt-1", "reimp-alt-2", "reimp-alt-3", "reimp-alt-4")
# A score within this of the largest baseline's has levelled off.
LEVEL = 0.01
# Score files must agree on these to share a page.
SHARED_META = ("dataset", "labels", "salt", "bootstrap", "endpoint", "pathways")
# Metastases, for saying whom the primary-tumour survival probe leaves out.
METASTATIC = ("Metastatic", "Additional Metastatic")
MEASURES = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
GROUPS = (
    ("classification", "Cancer type", ""),
    ("expression", "Expression", ""),
    ("survival", "Survival", ""),
    ("geometry", "Geometry", ""),
    ("confounders", "Confounders", "lower is more invariant"),
)


@dataclass(frozen=True)
class Metric:
    group: str
    table: str  # the probe's table in a score file
    column: str
    label: str
    task: str | None = None  # the classification task
    on_scorecard: bool = True  # scores in [0, 1] share the scorecard's axis


def _task_label(task: str, samples: pd.DataFrame) -> str:
    if task == "project_id":
        return f"cancer type, {task_labels(samples, task).nunique()} projects"
    if task == "tumor_vs_normal":
        return "tumour vs normal"
    return f"{task}: " + " / ".join(p.removeprefix("TCGA-") for p in ORGAN_TASKS[task])


def metrics(samples: pd.DataFrame) -> list[Metric]:
    """Every score on the page, in page order."""
    return [
        *(
            Metric(
                "classification", "classification", "balanced_accuracy", _task_label(t, samples), t
            )
            for t in TASKS
        ),
        Metric("expression", "invertibility", "r2_pooled", "expression R², pooled over genes"),
        Metric("expression", "invertibility", "r2_gene", "expression R², per gene"),
        Metric("expression", "invertibility", "pearson_sample", "per-sample Pearson"),
        Metric("expression", "pathways", "r2_pooled", "pathway R², pooled"),
        Metric("expression", "pathways", "r2_pathway", "pathway R², per pathway"),
        Metric("survival", "survival", "c_index", "C-index, pairs pooled"),
        Metric("survival", "survival", "c_index_macro", "C-index, mean over projects"),
        Metric("geometry", "geometry", "precision_at_1", "precision@1"),
        Metric("geometry", "geometry", "precision_at_10", "precision@10"),
        Metric("geometry", "geometry", "nmi", "k-means NMI"),
        Metric("geometry", "geometry", "ari", "k-means ARI"),
        Metric("geometry", "geometry", "silhouette", "silhouette"),
        Metric("geometry", "geometry", "effective_rank", "effective rank", on_scorecard=False),
        Metric("confounders", "confounders", "log_reads_r2", "log total reads R²"),
        Metric("confounders", "confounders", "assigned_fraction_r2", "assigned-read fraction R²"),
        Metric("confounders", "confounders", "strand_balance_r2", "strand balance R²"),
        Metric("confounders", "confounders", "plate_enrichment", "plate enrichment"),
        Metric("confounders", "confounders", "tss_enrichment", "tissue source site enrichment"),
    ]


def constant(metric: Metric, samples: pd.DataFrame, pooled: dict) -> tuple[float | None, str]:
    """What the metric scores with no embedding at all, and why; None where it has no such value."""
    column = metric.column
    if metric.table == "classification":
        k = task_labels(samples, metric.task).nunique()
        return 1 / k, f"chance, 1 / {k}: a constant guess recalls one class"
    if column == "pearson_sample":
        return pooled.get("pearson_sample_baseline"), "the training mean profile for every sample"
    if column.startswith("c_index"):
        return 0.5, "within a project, cancer type ranks no one"
    if column.startswith("precision_at"):
        share = task_labels(samples, "project_id").value_counts(normalize=True)
        return float((share**2).sum()), "a random training sample shares the cancer type"
    if column == "ari":
        return 0.0, "clusters that ignore cancer type"
    if column.endswith("_enrichment"):
        return 0.0, "neighbours that ignore the batch"
    if column.startswith("r2") or column.endswith("_r2"):
        if metric.table == "invertibility":
            return 0.0, "each gene's training mean"
        return 0.0, "each project's training mean"
    return None, ""


def _clean(value):
    """A score file's value as plain JSON: NaN and missing become None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


class Scores:
    """The rows of a set of `reimp-shared probe --json` files."""

    def __init__(self, paths: list[Path]) -> None:
        self.meta: dict | None = None
        rows: dict[str, list] = {}
        for path in paths:
            written = json.loads(Path(path).read_text())
            meta = written.pop("meta")
            written.pop("differences", None)
            if self.meta is None:
                self.meta = meta
            differ = [k for k in SHARED_META if meta.get(k) != self.meta.get(k)]
            if differ:
                raise ValueError(f"{path} was scored with a different {differ} than {paths[0]}")
            for table, records in written.items():
                rows.setdefault(table, []).extend(records)
        self.tables = {table: pd.DataFrame(records) for table, records in rows.items()}

    def row(self, table: str, embeddings: str, split: str, task: str | None = None) -> dict | None:
        frame = self.tables.get(table)
        if frame is None or frame.empty:
            return None
        match = (frame["embeddings"] == embeddings) & (frame["split"] == split)
        if task is not None:
            match &= frame["task"] == task
        found = frame[match]
        if len(found) > 1:
            raise ValueError(f"{table}: {len(found)} rows for {embeddings} {split} {task or ''}")
        return None if found.empty else {k: _clean(v) for k, v in found.iloc[0].items()}


def _point(row: dict | None, column: str) -> dict:
    row = row or {}
    return {
        k: row.get(c)
        for k, c in [("value", column), ("lo", f"{column}_lo"), ("hi", f"{column}_hi")]
    }


@dataclass(frozen=True)
class Jobs:
    """The score files: on the default folds, and the largest PCA under each other salt."""

    scores: list[Path]
    redrawn: dict[str, Path]


def run(
    work: Path = WORK,
    dims: tuple[int, ...] = DIMS,
    hvg_genes: int = HVG_GENES,
    salts: tuple[str, ...] = SALTS,
    bootstrap: int = N_BOOTSTRAP,
    rerun: bool = False,
    revision: str | None = None,
) -> Jobs:
    """Fit and score whatever is missing or stale."""
    head = max(dims)

    def cli(salt: str | None, *args: str) -> None:
        options = ["--revision", revision] if revision else []
        reimp_shared([*options, *(["--salt", salt] if salt else []), *args])

    def fit(model: Path, salt: str | None, *args: str) -> tuple[list[Path], float]:
        folds = [model / f"fold{k}.parquet" for k in range(N_FOLDS)]
        if rerun or not all(p.exists() for p in folds):
            cli(salt, *args, "--out", str(model))
        return folds, max(p.stat().st_mtime for p in folds)

    def score(target: Path, inputs: list[Path], fitted: float, salt: str | None) -> Path:
        if rerun or not target.exists() or target.stat().st_mtime < fitted:
            args = ["probe", *map(str, inputs), "--json", str(target)]
            cli(salt, *args, "--bootstrap", str(bootstrap), "--replicates")
        return target

    out = work / "scores"
    paths = []
    for d in dims:
        folds, fitted = fit(work / f"pca{d}", None, "baseline-pca", "--n-components", str(d))
        paths.append(score(out / f"pca{d}.json", [folds[0].parent], fitted, None))
        if d == head:
            paths.append(score(out / f"pca{d}_folds.json", folds, fitted, None))
    hvg = work / f"hvg{hvg_genes}"
    folds, fitted = fit(hvg, None, "baseline-hvg", "--n-genes", str(hvg_genes))
    paths.append(score(out / f"hvg{hvg_genes}.json", [hvg], fitted, None))
    redrawn = {}
    for salt in salts:
        model = work / "salts" / salt / f"pca{head}"
        folds, fitted = fit(model, salt, "baseline-pca", "--n-components", str(head))
        redrawn[salt] = score(out / f"pca{head}_salt-{salt}.json", [model], fitted, salt)
    return Jobs(paths, redrawn)


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(point: dict) -> str:
    return "" if point["lo"] is None else f"[{_f(point['lo'])}, {_f(point['hi'])}]"


def _int(n) -> str:
    return "n/a" if n is None else f"{int(round(n)):,}"


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _listed(items: list[str]) -> str:
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _power(alpha: float) -> str:
    """A penalty as TeX: 1, or a power of ten."""
    exponent = round(math.log10(alpha))
    return "1" if exponent == 0 else f"10^{{{exponent}}}"


def _short(ref: str) -> str:
    """`repo@commit` with the commit cut to 7 characters."""
    repo, _, commit = ref.partition("@")
    return f"{repo}@{commit[:7]}"


def _delta(differences: dict[str, pd.DataFrame], metric: Metric) -> dict | None:
    """The genes-minus-PCA difference of one pooled score, if it has replicates."""
    frame = differences.get(metric.table)
    if frame is None or frame.empty:
        return None
    match = (frame["split"] == "cv") & (frame["score"] == metric.column)
    if metric.task is not None:
        match &= frame["task"] == metric.task
    found = frame[match]
    if len(found) != 1:
        return None
    row = found.iloc[0]
    return {
        "value": float(row["delta"]),
        "lo": float(row["delta_lo"]),
        "hi": float(row["delta_hi"]),
        "share_above": float(row["share_above"]),
    }


def _survival_projects(scores: Scores, names: dict[str, str]) -> list[dict]:
    """Each project's own C-index under each baseline, projects with comparable pairs only."""
    frame = scores.tables.get("survival_projects")
    if frame is None or frame.empty:
        return []
    by_name = {
        key: {
            row["project"]: {k: _clean(v) for k, v in row.items()}
            for _, row in frame[(frame["embeddings"] == name) & (frame["split"] == "cv")].iterrows()
        }
        for key, name in names.items()
    }
    projects = [
        {
            "code": project.removeprefix("TCGA-"),
            "patients": row["patients"],
            "events": row["events"],
            **{key: _point(by_name[key].get(project), "c_index") for key in names},
        }
        for project, row in by_name["pca"].items()
        if row.get("c_index") is not None
    ]
    return sorted(projects, key=lambda p: -p["pca"]["value"])


def collect(
    scores: Scores,
    redrawn: dict[str, Scores],
    samples: pd.DataFrame,
    genes: pd.DataFrame,
    dims: tuple[int, ...] = DIMS,
    hvg_genes: int = HVG_GENES,
) -> tuple[dict, dict[str, str]]:
    """The chart data, and the prose's numbers as display strings."""
    head = max(dims)
    name, hvg = f"pca{head}", f"hvg{hvg_genes}"
    differences = {
        table: paired_differences(frame, hvg, name)
        for table, frame in scores.tables.items()
        if not frame.empty and hvg in set(frame["embeddings"])
    }
    entries = []
    for metric in metrics(samples):
        pooled = scores.row(metric.table, name, "cv", metric.task)
        point = _point(pooled, metric.column)
        if point["value"] is None:
            continue
        value, why = constant(metric, samples, pooled)

        def at(embeddings: str, split: str, table=metric.table, metric=metric) -> dict:
            return _point(scores.row(table, embeddings, split, metric.task), metric.column)

        sweep = [{"dim": d, **at(f"pca{d}", "cv")} for d in dims]
        # The headline itself is in the sweep, so something always qualifies.
        levelled = [
            p["dim"]
            for p in sweep
            if p["value"] is not None and abs(p["value"] - point["value"]) <= LEVEL
        ]
        entries.append(
            {
                "group": metric.group,
                "table": metric.table,
                "column": metric.column,
                "task": metric.task,
                "label": metric.label,
                "scorecard": metric.on_scorecard,
                "constant": value,
                "constant_note": why,
                **point,
                "hvg": at(hvg, "cv"),
                "delta": _delta(differences, metric),
                "levelled_at": min(levelled),
                "sweep": sweep,
                "folds": [
                    {"label": f"fold {k}", "value": at(name, f"fold{k}")["value"]}
                    for k in range(N_FOLDS)
                ],
                "salts": [
                    {
                        "label": f"salt {salt}",
                        "value": _point(
                            other.row(metric.table, name, "cv", metric.task), metric.column
                        )["value"],
                    }
                    for salt, other in redrawn.items()
                ],
            }
        )
    by_key = {(e["table"], e["task"] or e["column"]): e for e in entries}

    def entry(table: str, key: str) -> dict:
        """One pooled score at the largest size; an empty point where it was not scored."""
        return by_key.get(
            (table, key), {"value": None, "lo": None, "hi": None, "sweep": [], "hvg": {}}
        )

    def at_dim(e: dict, d: int) -> float | None:
        return next((p["value"] for p in e["sweep"] if p["dim"] == d), None)

    # ---- classification, every measure ----
    details = []
    for task in TASKS:
        row = scores.row("classification", name, "cv", task)
        if row is None:
            continue
        labels = task_labels(samples, task).dropna()
        details.append(
            {
                "task": task,
                "label": _task_label(task, samples),
                "classes": labels.nunique(),
                "n": row["n"],
                "majority": float(labels.value_counts(normalize=True).iloc[0]),
                **{m: _point(row, m) for m in MEASURES},
            }
        )
    tasks = [e for e in entries if e["table"] == "classification"]
    early_dim = dims[-2] if len(dims) > 1 else head
    early = [e for e in tasks if e["levelled_at"] <= early_dim]
    project = entry("classification", "project_id")
    project_row = scores.row("classification", name, "cv", "project_id") or {}

    # ---- genes against components ----
    card = [e for e in entries if e["scorecard"]]
    paired = [e for e in card if e["delta"] is not None]
    higher = [e["label"] for e in paired if e["delta"]["lo"] > 0]
    lower = [e["label"] for e in paired if e["delta"]["hi"] < 0]

    # ---- one fold alone, and the folds drawn again, against the pooled score ----
    def spread(key: str) -> tuple[tuple, int, int, list[float]]:
        deviations, outside, scored, ratios = [], 0, 0, []
        for e in card:
            values = [f["value"] for f in e[key] if f["value"] is not None]
            for f in e[key]:
                if f["value"] is None:
                    continue
                deviations.append((abs(f["value"] - e["value"]), e["label"], f["label"]))
                if e["lo"] is not None:
                    scored += 1
                    outside += not e["lo"] <= f["value"] <= e["hi"]
            if values and e["lo"] is not None and e["hi"] > e["lo"]:
                ratios.append(
                    float(np.std([e["value"], *values], ddof=1)) / ((e["hi"] - e["lo"]) / 2)
                )
        return max(deviations, default=(None, "n/a", "n/a")), outside, scored, ratios

    worst_fold, fold_outside, fold_scored, _ = spread("folds")
    worst_salt, salt_outside, salt_scored, salt_ratios = spread("salts")

    # ---- survival by project ----
    projects = _survival_projects(scores, {"pca": name, "hvg": hvg})
    skcm = next((p for p in projects if p["code"] == "SKCM"), None)
    # Patients sampled only at metastasis, whom the default survival probe leaves out.
    kinds = samples.groupby(CASE_KEY)["sample_type"].agg(set)
    only_metastatic = kinds.map(lambda k: bool(k & set(METASTATIC)) and not k & set(PRIMARY_TUMOR))
    project_of = samples.drop_duplicates(CASE_KEY).set_index(CASE_KEY)["project_id"]
    only_metastatic_projects = project_of[only_metastatic[only_metastatic].index]
    survival = scores.row("survival", name, "cv") or {}
    geometry = scores.row("geometry", name, "cv") or {}
    pathways = scores.row("pathways", name, "cv") or {}
    invert = scores.row("invertibility", name, "cv") or {}
    narrow = [e for e in card if e["lo"] is not None and e["hi"] - e["lo"] < 2 * LEVEL]
    with_ci = [e for e in card if e["lo"] is not None]
    strand, reads = entry("confounders", "strand_balance_r2"), entry("confounders", "log_reads_r2")
    meta = scores.meta
    train_pct = (N_FOLDS - 1) * (FOLD_WIDTH - VAL_BUCKETS) * 100 // N_BUCKETS

    facts = {
        "dataset": meta["dataset"],
        "dataset_short": _short(meta["dataset"]),
        "labels": meta["labels"],
        "labels_short": _short(meta["labels"]),
        "salt": meta["salt"],
        "bootstrap": _int(meta["bootstrap"]),
        "endpoint": meta["endpoint"].upper(),
        "pathways": meta["pathways"],
        "head": str(head),
        "min_dim": str(min(dims)),
        "dims": _listed([str(d) for d in dims]),
        "other_dims": _listed([str(d) for d in dims if d != head]),
        "hvg_genes": _int(hvg_genes),
        "hvg_n": str(hvg_genes),
        "hvg_ratio": f"{hvg_genes / head:.1f}".rstrip("0").rstrip("."),
        "dims_set": ", ".join(str(d) for d in dims),
        "genes_n": f"{len(select_genes(genes)):,}".replace(",", r"\,"),  # TeX
        "genes_all": _int(len(genes)),
        "lib_exp": str(round(math.log10(DEFAULT_LIBRARY_SIZE))),
        "bootstrap_n": str(meta["bootstrap"]),
        "test_pct": str(FOLD_WIDTH * 100 // N_BUCKETS),
        "ridge_lo": str(round(math.log10(min(RIDGE_ALPHAS)))),
        "ridge_hi": str(round(math.log10(max(RIDGE_ALPHAS)))),
        "cox_alphas": ", ".join(_power(a) for a in sorted(COX_ALPHAS, reverse=True)),
        "cox_folds": str(COX_FOLDS),
        "n_folds": str(N_FOLDS),
        "genes": _int(len(select_genes(genes))),
        "library_size": _int(DEFAULT_LIBRARY_SIZE),
        "train_pct": str(train_pct),
        "val_pct": str((N_FOLDS - 1) * VAL_BUCKETS * 100 // N_BUCKETS),
        "n_samples": _int(invert.get("n")),
        "n_tumour": _int(project_row.get("n")),
        "n_card": str(len(card)),
        "n_narrow": str(len(narrow)),
        "n_with_ci": str(len(with_ci)),
        "n_paired": str(len(paired)),
        "n_hvg_higher": str(len(higher)),
        "n_hvg_lower": str(len(lower)),
        "n_paired_ties": str(len(paired) - len(higher) - len(lower)),
        "hvg_higher_list": _listed(higher),
        "hvg_lower_list": _listed(lower),
        "project_classes": str(task_labels(samples, "project_id").nunique()),
        "project_bacc": _f(project["value"]),
        "project_bacc_ci": _ci(project),
        "project_bacc_min": _f(at_dim(project, min(dims))),
        "project_bacc_hvg": _f(project["hvg"].get("value")),
        "project_acc": _f(project_row.get("accuracy")),
        "project_wf1": _f(project_row.get("weighted_f1")),
        "level": _f(LEVEL, 2),
        "n_tasks": str(len(tasks)),
        "n_early": str(len(early)),
        "early_dim": str(early_dim),
        "colorectal_bacc": _f(entry("classification", "colorectal")["value"]),
        "colorectal_ci": _ci(entry("classification", "colorectal")),
        "inv_r2": _f(invert.get("r2_pooled")),
        "inv_r2_gene": _f(invert.get("r2_gene")),
        "inv_pearson": _f(invert.get("pearson_sample")),
        "pearson_base": _f(invert.get("pearson_sample_baseline")),
        "inv_r2_min": _f(at_dim(entry("invertibility", "r2_pooled"), min(dims))),
        "inv_r2_hvg": _f(entry("invertibility", "r2_pooled")["hvg"].get("value")),
        "n_pathways": _int(pathways.get("pathways")),
        "path_r2": _f(pathways.get("r2_pooled")),
        "path_r2_pathway": _f(pathways.get("r2_pathway")),
        "c_index": _f(survival.get("c_index")),
        "c_index_ci": _ci(entry("survival", "c_index")),
        "c_index_hvg": _f(entry("survival", "c_index")["hvg"].get("value")),
        "c_macro": _f(survival.get("c_index_macro")),
        "c_index_min": _f(at_dim(entry("survival", "c_index"), min(dims))),
        "survival_patients": _int(survival.get("patients")),
        "survival_events": _int(survival.get("events")),
        "survival_projects": _int(survival.get("projects")),
        "macro_min_events": str(MACRO_MIN_EVENTS),
        "n_proj": str(len(projects)),
        "n_proj_above": str(
            sum(p["pca"]["lo"] is not None and p["pca"]["lo"] > 0.5 for p in projects)
        ),
        "metastatic_only": _int(len(only_metastatic_projects)),
        "metastatic_only_where": (
            "all of them in SKCM"
            if (only_metastatic_projects == "TCGA-SKCM").all()
            else f"{_int((only_metastatic_projects == 'TCGA-SKCM').sum())} of them in SKCM"
        ),
        "skcm_patients": _int(skcm["patients"] if skcm else None),
        "skcm_c": _f(skcm["pca"]["value"] if skcm else None),
        "skcm_ci": _ci(skcm["pca"]) if skcm else "",
        "prec1_pct": _pct(geometry.get("precision_at_1"), 0),
        "prec1_pct1": _pct(geometry.get("precision_at_1")),
        "prec_chance_pct": _pct(entry("geometry", "precision_at_1").get("constant")),
        "prec_chance": _f(entry("geometry", "precision_at_1").get("constant")),
        "nmi": _f(geometry.get("nmi")),
        "ari": _f(geometry.get("ari")),
        "eff_rank": _f(geometry.get("effective_rank"), 1),
        "top_share": _pct(geometry.get("top_eigenvalue_share"), 0),
        "strand_min": _f(at_dim(strand, min(dims))),
        "strand_head": _f(strand["value"]),
        "strand_hvg": _f(strand["hvg"].get("value")),
        "reads_min": _f(at_dim(reads, min(dims))),
        "reads_head": _f(reads["value"]),
        "neighbours_k": str(DEFAULT_K),
        "plate_pts": _pct(entry("confounders", "plate_enrichment")["value"]).rstrip("%"),
        "tss_pts": _pct(entry("confounders", "tss_enrichment")["value"]).rstrip("%"),
        "fold_max_dev": _f(worst_fold[0]),
        "fold_max_metric": worst_fold[1],
        "fold_max_fold": worst_fold[2],
        "fold_outside": str(fold_outside),
        "fold_scored": str(fold_scored),
        "n_redraws": str(len(redrawn)),
        "n_assignments": str(len(redrawn) + 1),
        "salt_max_dev": _f(worst_salt[0]),
        "salt_max_metric": worst_salt[1],
        "salt_ratio": _f(float(np.median(salt_ratios)) if salt_ratios else None, 2),
        "salt_outside": str(salt_outside),
        "salt_scored": str(salt_scored),
        "salts": _listed([f"<{s}>" for s in redrawn]).replace("<", "").replace(">", ""),
    }
    data = {
        "dims": list(dims),
        "head": head,
        "hvg_genes": hvg_genes,
        "level": LEVEL,
        "groups": [{"id": g, "title": t, "note": n} for g, t, n in GROUPS],
        "metrics": entries,
        "classification": details,
        "projects": projects,
    }
    return data, facts


def build(
    out: Path | str = OUT,
    work: Path = WORK,
    dims: tuple[int, ...] = DIMS,
    hvg_genes: int = HVG_GENES,
    salts: tuple[str, ...] = SALTS,
    bootstrap: int = N_BOOTSTRAP,
    rerun: bool = False,
    revision: str | None = None,
) -> Path:
    """Compute what is missing under `work`, then write the report to `out`."""
    jobs = run(Path(work), dims, hvg_genes, salts, bootstrap, rerun, revision)
    scores = Scores(jobs.scores)
    redrawn = {salt: Scores([path]) for salt, path in jobs.redrawn.items()}
    samples, genes = hub.load_samples(revision), hub.load_genes(revision)
    data, facts = collect(scores, redrawn, samples, genes, dims, hvg_genes)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, facts, TEMPLATE.read_text()))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--work", type=Path, default=WORK, help="embeddings and score files")
    parser.add_argument("--rerun", action="store_true", help="fit and score every baseline again")
    parser.add_argument("--revision", default=None, help="dataset commit to pin (default: main)")
    args = parser.parse_args(argv)
    out = build(args.out, args.work, rerun=args.rerun, revision=args.revision)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
