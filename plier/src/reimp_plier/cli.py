"""`plier fit` — PLIER for one cross-validation fold, from a YAML config.

plier fit --config plier/configs/debug.yaml
plier fit --config plier/configs/tcga.yaml --data.fold 3
plier fit --config plier/configs/tcga.yaml --prior null --out_dir runs/plier_noprior

Each run writes `<out_dir>/fold<k>/`: `model.npz` (the genes, their
training means and SDs, Z, U, B, the λs and the prior), `annotations.tsv`
(held-out-gene AUCs for U's gene sets) and `config.yaml`, from which
`plier-embed` reloads the same data.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from reimp_plier.model import PLIER
from reimp_plier.pipeline import CONFIG_FILE, FoldModel, fit_fold
from reimp_plier.prior import read_gmt
from reimp_shared.data import load_expression
from reimp_shared.preprocess import DEFAULT_GENE_TYPES, DEFAULT_LIBRARY_SIZE, Transform


@dataclass
class DataConfig:
    """What the model sees, as `load_expression` arguments; `fold` is the fold it trains for."""

    quantification: str = "unstranded"
    gene_types: list[str] | None = field(default_factory=lambda: list(DEFAULT_GENE_TYPES))
    transform: Transform = "lognorm"
    library_size: float = DEFAULT_LIBRARY_SIZE
    projects: list[str] | None = None
    fold: int = 0
    revision: str | None = None


def build_parser() -> ArgumentParser:
    fit = ArgumentParser(description="Fit PLIER on one fold's training samples.")
    fit.add_argument("--config", action=ActionConfigFile)
    fit.add_argument(
        "--out_dir", type=str, default="runs/plier", help="the model goes to <out_dir>/fold<k>"
    )
    fit.add_argument(
        "--prior",
        type=str | None,
        default=None,
        help="GMT file of gene sets; null for the no-prior ablation (U = 0)",
    )
    fit.add_argument(
        "--all_genes",
        type=bool,
        default=True,
        help="model every gene that varies over the training samples; false: the prior's only",
    )
    fit.add_class_arguments(DataConfig, "data")
    fit.add_class_arguments(PLIER, "model")
    parser = ArgumentParser(prog="plier", description=__doc__.splitlines()[0])
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("fit", fit)
    return parser


def fit(config: Namespace) -> Path:
    """Load the data, fit the fold's model, and write it with the config that made it."""
    data_config = DataConfig(**config.data.as_dict())
    data = load_expression(**asdict(data_config))
    gene_sets = None if config.prior is None else read_gmt(config.prior)
    fold = fit_fold(data, PLIER(**config.model.as_dict()), gene_sets, config.all_genes)
    out = fold.save(Path(config.out_dir) / f"fold{data_config.fold}")
    record = {
        "out_dir": config.out_dir,
        "prior": config.prior,
        "all_genes": config.all_genes,
        "data": asdict(data_config),
        "model": fold.model.params(),
    }
    (out / CONFIG_FILE).write_text(yaml.safe_dump(record, sort_keys=False))
    _report(fold, gene_sets is not None, out)
    return out


def _report(fold: FoldModel, has_prior: bool, out: Path) -> None:
    m = fold.model
    l3 = "none" if m.l3_ is None else f"{m.l3_:.4g}"
    print(
        f"k = {m.k_}, lambda1 = {m.l1_:.4g}, lambda2 = {m.l2_:.4g}, lambda3 = {l3}; "
        f"{m.n_iter_} iterations; {len(fold.gene_ids)} genes"
    )
    if has_prior:
        with_set = int((m.u_.sum(axis=0) > 0).sum())
        print(
            f"prior: {len(m.names_)} gene sets, {len(fold.unmapped)} symbols unmapped; "
            f"LVs with a gene set {with_set} of {m.k_}, "
            f"annotated (held-out AUC > 0.7, FDR < 0.05) {len(m.annotated())}"
        )
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("reimp_plier").setLevel(logging.INFO)
    config = build_parser().parse_args(argv)
    if config.subcommand == "fit":
        fit(config.fit)
