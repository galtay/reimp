"""The gene-set prior, C, mapped onto the genes the model sees.

PLIER's C is a binary genes x gene-sets matrix. The sets come from
`reimp_shared.genesets.load_gene_sets` — MSigDB collections by name, GMT
files by path — keyed by HGNC symbol. Symbols are matched exactly to
GENCODE v36 `gene_name`s, and those that match no gene are counted, not
guessed at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from reimp_shared.genesets import GeneSets


@dataclass
class Prior:
    """C over the model's genes: `matrix[g, s]` is 1 when gene g is in set `names[s]`.

    `symbols` counts the distinct member symbols in the sets; `unmapped`
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
