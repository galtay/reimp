"""PLIER for one cross-validation fold, every statistic from its training samples.

`fit_fold` fits the per-gene means and SDs on the fold's training
samples, drops the genes constant there, maps the prior onto the rest and
fits PLIER — its SVD, k, λ1, λ2 and λ3 — on those samples alone.
`FoldModel.embed` z-scores any sample with the training statistics and
projects it with the trained Z and λ2, so training, validation and test
samples are embedded by one map.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from reimp_plier.model import PLIER, GeneScaler
from reimp_plier.prior import prior_matrix
from reimp_shared.data import ExpressionData
from reimp_shared.genesets import GeneSets

log = logging.getLogger(__name__)

MODEL_FILE, ANNOTATIONS_FILE, CONFIG_FILE = "model.npz", "annotations.tsv", "config.yaml"


@dataclass
class FoldModel:
    """A fitted fold: the genes it models, their training statistics, and PLIER.

    `unmapped` lists the prior's symbols that name none of the modelled genes.
    """

    gene_ids: np.ndarray
    gene_names: np.ndarray
    scaler: GeneScaler
    model: PLIER
    unmapped: list[str] = field(default_factory=list)

    def embed(self, data: ExpressionData) -> np.ndarray:
        """Every sample of `data`, samples x k: B = (ZᵀZ + λ2 I)⁻¹Zᵀy."""
        columns = pd.Index(data.genes["gene_id"]).get_indexer(self.gene_ids)
        if (columns < 0).any():
            raise ValueError(f"{int((columns < 0).sum())} of the model's genes are not in the data")
        y = self.scaler.transform(data.values[:, columns])
        return self.model.project(y).T.astype(np.float32)

    def save(self, directory: Path | str) -> Path:
        """`model.npz` (arrays and hyperparameters) and `annotations.tsv` under `directory`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        m = self.model
        np.savez_compressed(
            directory / MODEL_FILE,
            params=np.array(json.dumps(m.params())),
            gene_id=self.gene_ids.astype(str),
            gene_name=self.gene_names.astype(str),
            unmapped=np.array(self.unmapped, dtype=str),
            mean=self.scaler.mean,
            sd=self.scaler.sd,
            z=m.z_,
            b=m.b_,
            u=m.u_,
            prior=m.prior_.astype(bool),
            prior_cv=m.prior_cv_.astype(bool),
            gene_set=np.array(m.names_, dtype=str),
            singular_values=m.singular_values_,
            bdiff=m.bdiff_,
            k=m.k_,
            l1=m.l1_,
            l2=m.l2_,
            l3=np.nan if m.l3_ is None else m.l3_,
            n_iter=m.n_iter_,
        )
        m.annotations_.to_csv(directory / ANNOTATIONS_FILE, sep="\t", index=False)
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> FoldModel:
        directory = Path(directory)
        with np.load(directory / MODEL_FILE, allow_pickle=False) as f:
            model = PLIER(**json.loads(str(f["params"])))
            model.z_, model.b_, model.u_ = f["z"], f["b"], f["u"]
            model.prior_ = f["prior"].astype(np.float64)
            model.prior_cv_ = f["prior_cv"].astype(np.float64)
            model.names_ = f["gene_set"].tolist()
            model.singular_values_, model.bdiff_ = f["singular_values"], f["bdiff"]
            model.k_, model.n_iter_ = int(f["k"]), int(f["n_iter"])
            model.l1_, model.l2_ = float(f["l1"]), float(f["l2"])
            model.l3_ = None if np.isnan(f["l3"]) else float(f["l3"])
            fold = cls(
                gene_ids=f["gene_id"],
                gene_names=f["gene_name"],
                scaler=GeneScaler(mean=f["mean"], sd=f["sd"]),
                model=model,
                unmapped=f["unmapped"].tolist(),
            )
        model.annotations_ = pd.read_csv(directory / ANNOTATIONS_FILE, sep="\t")
        return fold


def fit_fold(
    data: ExpressionData,
    model: PLIER,
    gene_sets: GeneSets | None = None,
    all_genes: bool = True,
) -> FoldModel:
    """Fit `model` on the training samples of `data`'s fold.

    Only `data.rows("train")` is read. With `all_genes` (reimp's default,
    the package's `allGenes = TRUE`) every gene that varies over the
    training samples is modelled, genes in no set having empty rows of C;
    without it only the prior's genes are. No `gene_sets`: the no-prior
    ablation, U = 0.
    """
    train = data.values[data.rows("train")]
    scaler = GeneScaler.fit(train)
    genes = scaler.varying()
    names = data.genes["gene_name"].to_numpy()
    prior, unmapped = None, []
    if gene_sets is not None:
        mapped = prior_matrix(gene_sets, names[genes])
        prior, unmapped = mapped.matrix, mapped.unmapped
        log.info(
            "prior: %d gene sets, %d symbols, %d name none of the %d varying genes",
            len(gene_sets),
            mapped.symbols,
            len(unmapped),
            len(genes),
        )
        if not all_genes:
            members = prior.sum(axis=1) > 0
            genes, prior = genes[members], prior[members]
    elif not all_genes:
        raise ValueError("all_genes=False keeps the prior's genes, so it needs a prior")
    log.info(
        "%d training samples, %d genes (%d constant over them dropped)",
        len(train),
        len(genes),
        int((scaler.sd == 0).sum()),
    )
    scaler = scaler.subset(genes)
    y = scaler.transform(train[:, genes])
    del train
    model.fit(y, prior, None if gene_sets is None else list(gene_sets.names))
    return FoldModel(
        gene_ids=data.genes["gene_id"].to_numpy()[genes],
        gene_names=names[genes],
        scaler=scaler,
        model=model,
        unmapped=unmapped,
    )
