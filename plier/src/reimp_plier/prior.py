"""The gene-set prior: GMT files, mapped onto the genes the model sees.

PLIER's C is a binary genes x gene-sets matrix. The default prior,
`plier/priors/recommended.gmt` (exported by `plier/scripts/export_prior.R`),
holds the PLIER package's cell-type markers and canonical pathways keyed
by HGNC symbol; symbols are matched exactly to GENCODE v36 `gene_name`s,
and those that match no gene are counted, not guessed at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GeneSets:
    """Named gene sets, as a GMT file holds them: name, description, members."""

    names: list[str]
    descriptions: list[str]
    members: list[list[str]]

    def __len__(self) -> int:
        return len(self.names)


def read_gmt(path: Path | str) -> GeneSets:
    """Gene sets from a GMT file: one per line, tab-separated `name`, `description`, genes."""
    names, descriptions, members = [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name, description, *genes = line.rstrip("\n").split("\t")
        names.append(name)
        descriptions.append(description)
        members.append(list(dict.fromkeys(g for g in genes if g)))
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate gene-set names")
    return GeneSets(names, descriptions, members)


def write_gmt(path: Path | str, sets: GeneSets) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\t".join([name, description, *genes])
        for name, description, genes in zip(
            sets.names, sets.descriptions, sets.members, strict=True
        )
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass
class Prior:
    """C over the model's genes: `matrix[g, s]` is 1 when gene g is in set `names[s]`.

    `symbols` counts the distinct member symbols in the file; `unmapped`
    lists those that name none of the model's genes.
    """

    matrix: np.ndarray
    names: list[str]
    symbols: int
    unmapped: list[str]


def prior_matrix(sets: GeneSets, gene_names: Sequence[str]) -> Prior:
    """Map gene sets onto genes by name; a symbol shared by several genes marks them all."""
    rows: dict[str, list[int]] = {}
    for i, name in enumerate(gene_names):
        rows.setdefault(name, []).append(i)
    matrix = np.zeros((len(gene_names), len(sets)), dtype=np.float64)
    symbols: set[str] = set()
    for s, genes in enumerate(sets.members):
        symbols.update(genes)
        for gene in genes:
            matrix[rows.get(gene, []), s] = 1.0
    unmapped = sorted(g for g in symbols if g not in rows)
    return Prior(matrix=matrix, names=list(sets.names), symbols=len(symbols), unmapped=unmapped)
