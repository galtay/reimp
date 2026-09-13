"""The shared `ExpressionDataModule`, with gene selection, scaling and labels per fold.

`VAEDataModule` loads a fold exactly as `ExpressionDataModule` does, then
fits a `GeneScaler` — top genes by MAD, then min-max or z-score — on the
fold's training rows and applies it to every row. The fitted scaler is the
DataModule's state: Lightning saves it in every checkpoint and restores it
with `load_from_checkpoint`, so `vae-embed` scales exactly as training did,
without refitting.

With `supervision` other than `none`, every item also carries `label`, the
sample's organ or project (see `organs`): a fixed function of its project,
the same in every fold.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

from reimp_shared.data import (
    ExpressionData,
    ExpressionDataModule,
    ExpressionDataset,
    load_expression,
)
from reimp_shared.preprocess import DEFAULT_GENE_TYPES, DEFAULT_LIBRARY_SIZE, Transform
from reimp_shared.splits import DEFAULT_SALT, SPLITS
from reimp_vae.organs import Supervision, check_supervision, label_classes, sample_labels
from reimp_vae.scaling import GeneScaler, Scaling


class LabelledExpressionDataset(ExpressionDataset):
    """An `ExpressionDataset` whose items also carry `label`, one class index per row."""

    def __init__(
        self, values: Tensor, sample_index: Tensor, rows: np.ndarray, labels: Tensor
    ) -> None:
        super().__init__(values, sample_index, rows)
        self.labels = labels

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        item = super().__getitem__(i)
        item["label"] = self.labels[self.rows[i]]
        return item


class VAEDataModule(ExpressionDataModule):
    """`ExpressionDataModule` plus a per-gene scaler fit on training rows, and labels.

    - `top_genes` keeps the genes with the largest median absolute deviation
      over the training rows (Tybalt: 5,000); None keeps every selected gene.
    - `scaling` is `minmax` (Tybalt; val and test clipped to [0, 1]),
      `zscore` (the MMD-AE; val and test clipped to the training range) or
      `none`.
    - `supervision` adds labels: `organ` or `project` (`organs`), and
      `project_organ` replaces the committed project -> organ mapping.

    `n_genes` (after `top_genes`) and `n_classes` are set at construction,
    so a CLI can pass them to the model before `setup` loads the matrix.
    `data` holds what the model sees: the chosen genes, scaled.
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
        top_genes: int | None = None,
        scaling: Scaling = "none",
        supervision: Supervision = "none",
        project_organ: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            quantification=quantification,
            gene_types=gene_types,
            gene_ids=gene_ids,
            gene_ids_path=gene_ids_path,
            drop_par_y=drop_par_y,
            transform=transform,
            library_size=library_size,
            projects=projects,
            sample_types=sample_types,
            fold=fold,
            split_salt=split_salt,
            revision=revision,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        check_supervision(supervision)
        self.scaler = GeneScaler(scaling, top_genes)
        if top_genes is not None:
            self.n_genes = min(top_genes, self.n_genes)
        self.project_organ: Mapping[str, str] | None = project_organ
        self.classes = label_classes(supervision, project_organ)
        self.n_classes = len(self.classes)

    def setup(self, stage: str | None = None) -> None:
        if self.data is not None:
            return
        loaded = load_expression(**self.load_kwargs)
        if self.scaler.genes_ is None:  # not restored from a checkpoint
            self.scaler.fit(loaded.values[loaded.rows("train")])
        self.data = ExpressionData(
            samples=loaded.samples,
            genes=loaded.genes.iloc[self.scaler.genes_].reset_index(drop=True),
            values=self.scaler.transform(loaded.values),
        )
        values = torch.from_numpy(self.data.values)
        sample_index = torch.from_numpy(self.data.samples["sample_index"].to_numpy(copy=True))
        labels = None
        if self.n_classes:
            labels = torch.from_numpy(
                sample_labels(
                    self.data.samples["project_id"],
                    self.hparams.supervision,
                    self.project_organ,
                )
            )

        def dataset(rows: np.ndarray) -> ExpressionDataset:
            if labels is None:
                return ExpressionDataset(values, sample_index, rows)
            return LabelledExpressionDataset(values, sample_index, rows, labels)

        for split in SPLITS:
            self.datasets[split] = dataset(self.data.rows(split))
        self.datasets["all"] = dataset(np.arange(len(values)))

    def state_dict(self) -> dict[str, Tensor]:
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.scaler.load_state_dict(state_dict)
