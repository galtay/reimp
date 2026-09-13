"""Configurable access to the TCGA expression matrix, for any model.

Three choices define what a model sees, and each is an argument here:

  quantification  which GDC measure: read counts (`unstranded`,
                  `stranded_first`, `stranded_second`) or a per-library
                  normalized one (`tpm_unstranded`, `fpkm_unstranded`,
                  `fpkm_uq_unstranded`).
  genes           which columns: GENCODE biotypes (protein-coding by
                  default), an explicit list of Ensembl IDs (a model's fixed
                  vocabulary), or all 60,660.
  transform       what to do to each row: nothing (the default), log1p, or
                  library-size normalization then log1p (`lognorm`).

Samples can also be restricted by project and by sample type. The
train / val / test assignment is one fold of `reimp_shared.splits`' 5-fold
cross-validation — fold 0 unless `fold` says otherwise — drawn over the
whole dataset before any filter, so it is the same under every combination
of the above:
two models trained on different quantifications or gene sets still hold
out the same patients.

`load_expression` returns numpy for any framework; `ExpressionDataset` and
`ExpressionDataModule` wrap it for torch and Lightning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from reimp_shared import hub
from reimp_shared.preprocess import (
    DEFAULT_GENE_TYPES,
    DEFAULT_LIBRARY_SIZE,
    Transform,
    read_gene_ids,
    select_genes,
    transform_values,
)
from reimp_shared.splits import DEFAULT_SALT, SPLITS, check_fold, sample_folds, split_samples


@dataclass
class ExpressionData:
    """A loaded selection: `values[i, j]` is sample `samples.iloc[i]`, gene `genes.iloc[j]`.

    `samples` keeps every column of the dataset's `samples` config —
    `sample_index` being the row's position in the full dataset — plus
    `split`, the sample's split in the fold loaded, and `fold`, the fold in
    which it is a test sample.
    """

    samples: pd.DataFrame
    genes: pd.DataFrame
    values: np.ndarray

    def rows(self, split: str) -> np.ndarray:
        """Row positions in `values` of one split's samples."""
        return np.flatnonzero(self.samples["split"].to_numpy() == split)


def load_expression(
    quantification: str = "unstranded",
    gene_types: Sequence[str] | None = DEFAULT_GENE_TYPES,
    gene_ids: Sequence[str] | None = None,
    drop_par_y: bool = True,
    transform: Transform = "none",
    library_size: float = DEFAULT_LIBRARY_SIZE,
    projects: Sequence[str] | None = None,
    sample_types: Sequence[str] | None = None,
    fold: int = 0,
    split_salt: str = DEFAULT_SALT,
    revision: str | None = None,
) -> ExpressionData:
    """Load one quantification for the chosen genes and samples, transformed.

    Genes are chosen as in `preprocess.select_genes`. `lognorm` normalizes
    over the selected genes, so each sample's library is `library_size`
    across exactly the columns the model sees. `split` is each sample's
    split in fold `fold`.
    """
    samples = hub.load_samples(revision)
    genes = hub.load_genes(revision)
    columns = select_genes(genes, gene_types, gene_ids, drop_par_y)

    keep = np.ones(len(samples), dtype=bool)
    if projects is not None:
        keep &= samples["project_id"].isin(list(projects)).to_numpy()
    if sample_types is not None:
        keep &= samples["sample_type"].isin(list(sample_types)).to_numpy()
    if not keep.any():
        raise ValueError(f"no samples in projects={projects} with sample_types={sample_types}")
    rows = np.flatnonzero(keep)

    values = hub.load_values(quantification, revision)[np.ix_(rows, columns)]
    kept = samples.iloc[rows].reset_index(drop=True)
    # Folds are drawn over the whole cohort, then the selected rows kept.
    kept["split"] = split_samples(samples, fold, split_salt).to_numpy()[rows]
    kept["fold"] = sample_folds(samples, split_salt).to_numpy()[rows]
    return ExpressionData(
        samples=kept,
        genes=genes.iloc[columns].reset_index(drop=True),
        values=transform_values(values, transform, library_size),
    )


class ExpressionDataset(Dataset):
    """Rows `rows` of an expression matrix, as `{"values": (G,), "sample_index": ()}`."""

    def __init__(self, values: Tensor, sample_index: Tensor, rows: np.ndarray) -> None:
        self.values = values
        self.sample_index = sample_index
        self.rows = torch.as_tensor(rows, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        row = self.rows[i]
        return {"values": self.values[row], "sample_index": self.sample_index[row]}


class ExpressionDataModule(L.LightningDataModule):
    """`load_expression` behind train / val / test / predict loaders.

    Takes every `load_expression` argument, plus `gene_ids_path` (a text
    file of IDs, one per line, for vocabularies too long for a config) and
    loader settings; `data.fold: 2` in a config trains the model for fold
    2. The predict loader covers every selected sample, in `sample_index`
    order.

    `n_genes` is set at construction — gene selection needs only the small
    `genes` config — so a CLI can pass it to the model before `setup`
    loads the matrix.
    """

    def __init__(
        self,
        quantification: str = "unstranded",
        gene_types: tuple[str, ...] | None = DEFAULT_GENE_TYPES,
        gene_ids: list[str] | None = None,
        gene_ids_path: str | None = None,
        drop_par_y: bool = True,
        transform: Transform = "none",
        library_size: float = DEFAULT_LIBRARY_SIZE,
        projects: list[str] | None = None,
        sample_types: list[str] | None = None,
        fold: int = 0,
        split_salt: str = DEFAULT_SALT,
        revision: str | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if gene_ids_path is not None:
            if gene_ids is not None:
                raise ValueError("pass gene_ids or gene_ids_path, not both")
            gene_ids = read_gene_ids(gene_ids_path)
        check_fold(fold)
        self.load_kwargs = dict(
            quantification=quantification,
            gene_types=gene_types,
            gene_ids=gene_ids,
            drop_par_y=drop_par_y,
            transform=transform,
            library_size=library_size,
            projects=projects,
            sample_types=sample_types,
            fold=fold,
            split_salt=split_salt,
            revision=revision,
        )
        genes = hub.load_genes(revision)
        self.n_genes = len(select_genes(genes, gene_types, gene_ids, drop_par_y))
        self.data: ExpressionData | None = None
        self.datasets: dict[str, ExpressionDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        if self.data is not None:
            return
        self.data = load_expression(**self.load_kwargs)
        values = torch.from_numpy(self.data.values)
        sample_index = torch.from_numpy(self.data.samples["sample_index"].to_numpy(copy=True))
        for split in SPLITS:
            self.datasets[split] = ExpressionDataset(values, sample_index, self.data.rows(split))
        self.datasets["all"] = ExpressionDataset(values, sample_index, np.arange(len(values)))

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader:
        return self._loader("test")

    def predict_dataloader(self) -> DataLoader:
        return self._loader("all")
