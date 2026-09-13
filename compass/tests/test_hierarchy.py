from pathlib import Path
from statistics import median

import pytest

from reimp_compass.hierarchy import concept_gene_ids, load_hierarchy
from reimp_shared import hub
from reimp_shared.preprocess import read_gene_ids

CONFIGS = Path(__file__).parents[1] / "configs"
FAKE_PROTEIN_CODING = [f"GENE{i}" for i in range(0, 46, 3)]


def test_packaged_hierarchy_is_compass_s() -> None:
    """The counts paper.md and the paper's projector size (1,514 parameters) imply."""
    h = load_hierarchy()
    assert len(h.sets) == 132
    assert len(h.concepts) == 43
    assert len(h.genes) == 916
    # One logit per membership: 1,283 + 132 set logits + three Linear(32 -> 1)
    # scorers (sets, cancer, patient) is the released projector's 1,514.
    assert len(h.member_set) == 1283
    assert 1283 + 132 + 3 * 33 == 1514
    sizes = [len(genes) for genes in h.set_genes]
    assert (min(sizes), median(sizes), max(sizes)) == (1, 7, 51)
    assert h.concepts[0] == "Bcell_general"
    assert h.concepts[-1] == "Reference"
    reference = [s for s, c in zip(h.sets, h.set_concept, strict=True) if c == 42]
    assert reference == ["Ubiquitous_immune_sc", "Ubiquitous_sc", "Reference_NanoString09"]
    assert sorted(set(h.set_concept)) == list(range(43))


def test_sets_follow_the_index_and_reference_goes_last(hierarchy_path) -> None:
    h = load_hierarchy(hierarchy_path)
    assert h.sets == ("Ref_set", "A_one", "A_two", "B_one", "B_two", "B_ghost")
    assert h.concepts == ("Concept_A", "Concept_B", "Reference")
    assert h.set_concept == (2, 0, 0, 1, 1, 1)
    assert h.member_set.tolist() == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5]


def test_members_map_by_symbol_and_mark_missing_genes(hierarchy_path) -> None:
    h = load_hierarchy(hierarchy_path)
    positions = h.member_positions(FAKE_PROTEIN_CODING)
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 10, 11, 10, -1]
    assert positions.tolist() == expected
    assert h.missing(FAKE_PROTEIN_CODING) == ["GENE1", "NOTAGENE"]


def test_a_repeated_symbol_maps_to_its_first_column(hierarchy_path) -> None:
    h = load_hierarchy(hierarchy_path)
    assert h.member_positions(["GENE3", "GENE0", "GENE0"])[:2].tolist() == [1, 0]


def test_a_concept_with_two_indexes_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text(
        "GeneSet\tBroadCelltypePathway\tGeneSet_index\tConcept_index\tGenes\n"
        "s0\tC\t0\t0\tA:B\n"
        "s1\tC\t1\t1\tC\n"
    )
    with pytest.raises(ValueError, match="Concept_index"):
        load_hierarchy(path)


def test_missing_columns_are_rejected(tmp_path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("GeneSet\tGenes\ns0\tA\n")
    with pytest.raises(ValueError, match="lacks columns"):
        load_hierarchy(path)


@pytest.mark.network
def test_every_concept_gene_is_in_the_protein_coding_default() -> None:
    ids = concept_gene_ids(hub.load_genes())
    assert len(ids) == 916
    assert ids == read_gene_ids(str(CONFIGS / "concept_genes.txt"))
