"""COMPASS's fixed gene -> gene set -> concept hierarchy.

The 132 literature gene sets and the 43 concepts they feed are COMPASS's
`compass/tokenizer/conception_processed.tsv`, packaged unchanged as
`reimp_compass/data/conception_processed.tsv` (provenance and licence in
`data/README.md`). Each row is one set: its member genes by HGNC symbol,
colon-separated in `Genes`, and the concept it feeds in
`BroadCelltypePathway`. Sets are ordered by `GeneSet_index` and concepts by
`Concept_index` with `Reference` (housekeeping genes) last — COMPASS's own
order.

A membership is one (set, gene) pair. The file has 1,283 memberships over
916 genes; a gene may sit in several sets, and two sets list one gene
twice, which COMPASS keeps (the gene gets two logits in that set's softmax).

Genes are addressed by symbol, so `Hierarchy.member_positions` maps each
membership onto a gene selection's `gene_name` column. A symbol the
selection lacks drops out of its set, as COMPASS's tokenizer drops genes
outside its vocabulary. Run the module to print the Ensembl IDs of the
concept genes in the protein-coding default, the list
`configs/concept_genes.txt` holds:

  uv run python -m reimp_compass.hierarchy > compass/configs/concept_genes.txt
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).parent / "data" / "conception_processed.tsv"
REFERENCE = "Reference"
COLUMNS = ("GeneSet", "BroadCelltypePathway", "GeneSet_index", "Concept_index", "Genes")


@dataclass(frozen=True)
class Hierarchy:
    """Gene sets, their member symbols, and the concept each set feeds.

    `set_concept[s]` is the position in `concepts` of set `s`'s concept.
    """

    sets: tuple[str, ...]
    set_genes: tuple[tuple[str, ...], ...]
    set_concept: tuple[int, ...]
    concepts: tuple[str, ...]

    @property
    def genes(self) -> tuple[str, ...]:
        """Every member symbol once, in order of first appearance."""
        return tuple(dict.fromkeys(g for genes in self.set_genes for g in genes))

    @property
    def member_set(self) -> np.ndarray:
        """Set position of each membership, sets in order: (n_memberships,) int64."""
        sizes = [len(genes) for genes in self.set_genes]
        return np.repeat(np.arange(len(self.sets)), sizes)

    def member_positions(self, gene_names: Sequence[str]) -> np.ndarray:
        """Position in `gene_names` of each membership's gene, or -1 where it is absent.

        A symbol that occurs more than once in `gene_names` (possible when
        selecting beyond the protein-coding default) maps to its first
        occurrence.
        """
        first: dict[str, int] = {}
        for i, name in enumerate(gene_names):
            first.setdefault(name, i)
        return np.array(
            [first.get(g, -1) for genes in self.set_genes for g in genes], dtype=np.int64
        )

    def missing(self, gene_names: Sequence[str]) -> list[str]:
        """Member symbols not in `gene_names`."""
        present = set(gene_names)
        return [g for g in self.genes if g not in present]


def load_hierarchy(path: str | Path | None = None) -> Hierarchy:
    """Read a hierarchy table: COMPASS's by default, or one with its columns.

    Needs `GeneSet`, `BroadCelltypePathway`, `GeneSet_index`,
    `Concept_index` and `Genes`; every set of a concept must carry the same
    `Concept_index`.
    """
    table = pd.read_csv(DEFAULT_PATH if path is None else path, sep="\t")
    absent = [c for c in COLUMNS if c not in table.columns]
    if absent:
        raise ValueError(f"hierarchy table lacks columns {absent}")
    table = table.sort_values("GeneSet_index", kind="stable")
    index = table.groupby("BroadCelltypePathway")["Concept_index"].unique()
    if (index.map(len) > 1).any():
        raise ValueError("a concept's sets disagree on its Concept_index")
    concepts = sorted(index.index, key=lambda c: (c == REFERENCE, index[c][0]))
    position = {c: i for i, c in enumerate(concepts)}
    set_genes = tuple(tuple(genes.split(":")) for genes in table["Genes"])
    if any(not all(genes) for genes in set_genes):
        raise ValueError("a gene set has an empty member")
    return Hierarchy(
        sets=tuple(table["GeneSet"]),
        set_genes=set_genes,
        set_concept=tuple(position[c] for c in table["BroadCelltypePathway"]),
        concepts=tuple(concepts),
    )


def concept_gene_ids(genes: pd.DataFrame, hierarchy: Hierarchy | None = None) -> list[str]:
    """Ensembl IDs of the concept genes among the protein-coding default, in column order."""
    from reimp_shared.preprocess import select_genes

    hierarchy = load_hierarchy() if hierarchy is None else hierarchy
    selected = genes.iloc[select_genes(genes)]
    return selected.loc[selected["gene_name"].isin(hierarchy.genes), "gene_id"].tolist()


def _main() -> None:
    from reimp_shared import hub

    hierarchy = load_hierarchy()
    ids = concept_gene_ids(hub.load_genes(), hierarchy)
    print("# COMPASS's concept genes among the protein-coding default, as Ensembl IDs:")
    print(f"# {len(ids)} of the hierarchy's {len(hierarchy.genes)} symbols.")
    print("# Generated by `uv run python -m reimp_compass.hierarchy`.")
    print("\n".join(ids))


if __name__ == "__main__":
    _main()
