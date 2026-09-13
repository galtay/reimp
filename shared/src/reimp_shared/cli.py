"""`reimp-shared` — inspect the folds, build the baselines, evaluate embeddings.

reimp-shared splits [--fold K]
reimp-shared baseline-pca --out out/pca256
reimp-shared baseline-hvg --out out/hvg5000
reimp-shared probe out/pca256 out/txfm_s --against pca256
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from reimp_shared import hub
from reimp_shared.data import load_expression
from reimp_shared.eval import (
    classification_probe,
    confounder_probe,
    geometry_probe,
    hvg_embeddings,
    invertibility,
    invertibility_target,
    paired_differences,
    pathway_probe,
    pca_embeddings,
    read_embeddings,
    survival_predictions,
    survival_scores,
    write_embeddings,
)
from reimp_shared.eval.bootstrap import BOOT, N_BOOTSTRAP
from reimp_shared.labels import (
    PATIENTS_REPO,
    SSGSEA_COLLECTIONS,
    SURVIVAL_ENDPOINTS,
    load_ssgsea,
    load_survival,
    technical_covariates,
)
from reimp_shared.labels import commit as labels_commit
from reimp_shared.preprocess import DEFAULT_LIBRARY_SIZE, TRANSFORMS
from reimp_shared.splits import (
    DEFAULT_SALT,
    FOLD_WIDTH,
    N_FOLDS,
    VAL_BUCKETS,
    sample_folds,
    split_samples,
    summarize,
    summarize_folds,
)

# The baselines and the models write one file per fold, named for it.
FOLD_FILE = re.compile(r"fold\d+")
TABLES = {
    "classification": (
        "classification: logistic regression onto sample labels",
        ["task", "split", "embeddings", "dim", "n"]
        + ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    ),
    "invertibility": (
        "invertibility: ridge regression onto log-normalized protein-coding counts",
        ["split", "embeddings", "dim", "n", "alpha", "r2_pooled", "r2_gene"]
        + ["pearson_sample", "pearson_sample_baseline"],
    ),
    "pathways": (
        "pathways: ridge regression onto ssGSEA scores within projects (R² over project means)",
        ["split", "embeddings", "dim", "n", "pathways", "alpha", "r2_pooled", "r2_pathway"],
    ),
    "survival": (
        "survival: Cox regression stratified by project; C-index within projects (baseline 0.5)",
        ["endpoint", "split", "embeddings", "dim", "patients", "events", "projects"]
        + ["alpha", "c_index", "c_index_macro"],
    ),
    "survival_projects": (
        "survival by project: each project's own C-index, pairs pooled over its folds",
        ["endpoint", "split", "embeddings", "project", "patients", "events", "c_index"],
    ),
    "geometry": (
        "geometry: tumour samples against cancer type",
        ["split", "embeddings", "dim", "n", "effective_rank", "top_eigenvalue_share"]
        + ["precision_at_1", "precision_at_10", "nmi", "ari", "silhouette"],
    ),
    "confounders": (
        "confounders: technical structure within projects (lower is more invariant)",
        ["split", "embeddings", "dim", "n", "log_reads_r2", "assigned_fraction_r2"]
        + ["strand_balance_r2", "plate_enrichment", "tss_enrichment"],
    ),
}
DIFFERENCE_COLUMNS = ["table", "row", "split", "embeddings", "score", "delta", "share_above"]


def _splits(args: argparse.Namespace) -> None:
    samples = hub.load_samples(args.revision)
    if args.fold is None:
        table = summarize_folds(samples, args.salt)
        group = sample_folds(samples, args.salt).map(lambda k: f"fold{k}")
    else:
        table = summarize(samples, args.fold, args.salt)
        group = split_samples(samples, args.fold, args.salt)
    with pd.option_context("display.max_rows", None):
        print(table)
    print()
    for name in table.columns.drop("total"):
        members = samples[group == name]
        n, cases = len(members), members["case_submitter_id"].nunique()
        print(f"{name:<6}{n:>7} samples{cases:>7} cases  ({n / len(samples):.1%} of samples)")
    if args.fold is None:
        train = (N_FOLDS - 1) * (FOLD_WIDTH - VAL_BUCKETS)
        val = (N_FOLDS - 1) * VAL_BUCKETS
        print(f"\neach fold's test set; its model trains on {train}% of cases, validates on {val}%")


def _baseline(
    args: argparse.Namespace, embed: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None:
    """Per fold, `embed(values, train_rows)` of the chosen expression, for every sample."""
    data = load_expression(
        quantification=args.quantification,
        gene_types=args.gene_types or None,
        transform=args.transform,
        library_size=args.library_size,
        split_salt=args.salt,
        revision=args.revision,
    )
    sample_index = data.samples["sample_index"].to_numpy()
    for fold in args.folds:
        split = split_samples(data.samples, fold, args.salt).to_numpy()
        embeddings = embed(data.values, np.flatnonzero(split == "train"))
        out = write_embeddings(args.out / f"fold{fold}.parquet", sample_index, embeddings, fold)
        print(f"wrote {out}  ({embeddings.shape[0]} samples x {embeddings.shape[1]} dims)")


def _baseline_pca(args: argparse.Namespace) -> None:
    _baseline(args, lambda values, train: pca_embeddings(values, train, args.n_components))


def _baseline_hvg(args: argparse.Namespace) -> None:
    _baseline(args, lambda values, train: hvg_embeddings(values, train, args.n_genes))


def _formatted(table: pd.DataFrame, columns: list[str]) -> str:
    """Floats to 3 places, with `[lo, hi]` wherever an interval was computed."""
    shown = pd.DataFrame(index=table.index)
    for column in columns:
        values = table[column] if column in table else pd.Series(float("nan"), index=table.index)
        if f"{column}_lo" in table:
            low, high = table[f"{column}_lo"], table[f"{column}_hi"]
            shown[column] = [
                f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in zip(values, low, high, strict=True)
            ]
        elif pd.api.types.is_float_dtype(values):
            shown[column] = [f"{v:.3f}" for v in values]
        else:
            shown[column] = values
    return shown.to_string(index=False)


def _embeddings_name(path: Path) -> str:
    """A model's label: its directory's name, for the directory or a fold file in it."""
    if path.is_dir():
        return path.name
    if FOLD_FILE.fullmatch(path.stem) and path.parent.name:
        return path.parent.name
    return path.stem


def _differences(tables: dict[str, pd.DataFrame], reference: str) -> pd.DataFrame:
    """Every other embeddings' scores minus `reference`'s, table by table."""
    frames = []
    for name, table in tables.items():
        if table.empty:
            continue
        for other in table["embeddings"].unique():
            if other == reference:
                continue
            found = paired_differences(table, other, reference)
            if found.empty:
                continue
            keys = [k for k in ("task", "project") if k in found]
            row = found[keys].astype(str).agg(" ".join, axis=1) if keys else ""
            frames.append(found.assign(table=name, row=row, embeddings=other, against=reference))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _records(table: pd.DataFrame, replicates: bool) -> list[dict]:
    if not replicates:
        table = table.drop(columns=[c for c in table.columns if c.endswith(BOOT)])
    return table.to_dict(orient="records")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _probe(args: argparse.Namespace) -> None:
    names = [_embeddings_name(Path(path)) for path in args.embeddings]
    if args.against is not None and args.against not in names:
        raise SystemExit(f"--against {args.against!r} is none of the embeddings: {names}")
    samples = hub.load_samples(args.revision)
    target = invertibility_target(args.revision)
    covariates = technical_covariates(args.revision)
    print("loading labels (a first run reads them remotely: minutes)", file=sys.stderr)
    survival = load_survival()
    pathways = load_ssgsea(args.pathways)
    results = {name: [] for name in TABLES}
    for path, name in zip(args.embeddings, names, strict=True):
        sample_index, embeddings, fold = read_embeddings(path)
        tag = {"embeddings": name, "dim": embeddings.shape[1]}
        common = {"fold": fold, "salt": args.salt, "n_bootstrap": args.bootstrap}
        risks = survival_predictions(
            sample_index,
            embeddings,
            samples,
            survival,
            args.endpoint,
            fold=fold,
            salt=args.salt,
        )
        by_patient, by_project = survival_scores(risks, args.bootstrap)
        scores = {
            "classification": classification_probe(sample_index, embeddings, samples, **common),
            "invertibility": invertibility(sample_index, embeddings, target, **common),
            "pathways": pathway_probe(sample_index, embeddings, samples, pathways, **common),
            "survival": by_patient,
            "survival_projects": by_project,
            "geometry": geometry_probe(sample_index, embeddings, samples, **common),
            "confounders": confounder_probe(
                sample_index, embeddings, samples, covariates, **common
            ),
        }
        for table, frame in scores.items():
            results[table].append(frame.assign(**tag))

    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in results.items()}
    for name, (title, columns) in TABLES.items():
        print(f"\n{title}")
        print(_formatted(tables[name], columns))
    differences = pd.DataFrame()
    if args.against is not None:
        differences = _differences(tables, args.against)
        print(f"\ndifferences from {args.against}, with paired 95% intervals")
        print(_formatted(differences, DIFFERENCE_COLUMNS) if not differences.empty else "none")
    if args.json:
        meta = {
            "dataset": f"{hub.REPO_ID}@{hub.commit(args.revision)}",
            "labels": f"{PATIENTS_REPO}@{labels_commit()}",
            "salt": args.salt,
            "bootstrap": args.bootstrap,
            "endpoint": args.endpoint,
            "pathways": args.pathways,
        }
        records = {name: _records(table, args.replicates) for name, table in tables.items()}
        if args.against is not None:
            records["differences"] = differences.to_dict(orient="records")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"meta": meta, **records}, indent=2, default=_jsonable))
        print(f"\nwrote {args.json}")


def _expression_arguments(p: argparse.ArgumentParser) -> None:
    """Where a baseline writes, which folds it fits, and which expression it reads."""
    p.add_argument("--out", type=Path, required=True, help="directory for fold<k>.parquet files")
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        choices=range(N_FOLDS),
        default=list(range(N_FOLDS)),
        help="folds to fit, each on its train split (default: all)",
    )
    p.add_argument("--quantification", default="unstranded", choices=hub.QUANTIFICATIONS)
    p.add_argument(
        "--gene-types",
        nargs="*",
        default=["protein_coding"],
        help="GENCODE biotypes to keep; give the flag with no values to keep all",
    )
    p.add_argument("--transform", default="lognorm", choices=TRANSFORMS)
    p.add_argument("--library-size", type=float, default=DEFAULT_LIBRARY_SIZE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reimp-shared", description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default=None, help="dataset commit to pin (default: main)")
    parser.add_argument("--salt", default=DEFAULT_SALT, help="split salt")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("splits", help="samples per project in each fold's test set")
    p.add_argument(
        "--fold", type=int, choices=range(N_FOLDS), help="one fold's train / val / test instead"
    )
    p.set_defaults(fn=_splits)

    p = sub.add_parser("baseline-pca", help="per fold, PCA embeddings of transformed expression")
    _expression_arguments(p)
    p.add_argument("--n-components", type=int, default=256)
    p.set_defaults(fn=_baseline_pca)

    p = sub.add_parser(
        "baseline-hvg", help="per fold, the most variable genes of transformed expression"
    )
    _expression_arguments(p)
    p.add_argument("--n-genes", type=int, default=5000)
    p.set_defaults(fn=_baseline_hvg)

    p = sub.add_parser("probe", help="every evaluation, on one or more embeddings files")
    p.add_argument(
        "embeddings",
        nargs="+",
        type=Path,
        help="a model's directory of per-fold files (all five score as `cv`), or single files",
    )
    p.add_argument("--endpoint", default="pfi", choices=SURVIVAL_ENDPOINTS)
    p.add_argument("--pathways", default="hallmark", choices=SSGSEA_COLLECTIONS)
    p.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="patient-bootstrap replicates for intervals (0 for none)",
    )
    p.add_argument(
        "--against",
        default=None,
        help="also report every other embeddings' scores minus this one's (a name, e.g. pca256)",
    )
    p.add_argument("--json", type=Path, default=None, help="also write the scores here")
    p.add_argument(
        "--replicates",
        action="store_true",
        help="keep each score's bootstrap replicates in --json, for paired comparisons later",
    )
    p.set_defaults(fn=_probe)

    args = parser.parse_args(argv)
    args.fn(args)
