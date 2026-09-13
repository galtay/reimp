import pytest

from reimp_shared.testing import use_fake_dataset, write_fake_dataset

# A hierarchy over the fake dataset's genes. Its protein-coding genes are
# GENE0, GENE3, ..., GENE45; GENE1 is a lncRNA, so the default selection
# lacks it, and NOTAGENE is in no selection. Rows are out of index order,
# the index has a gap, Reference comes first by index but is moved last,
# GENE30 is listed twice in one set, and B_ghost has no gene in the data.
HIERARCHY = [
    ("GeneSet", "BroadCelltypePathway", "GeneSet_index", "Concept_index", "Genes"),
    ("Ref_set", "Reference", 0, 0, "GENE0:GENE3"),
    ("A_two", "Concept_A", 3, 1, "GENE15:GENE18"),
    ("A_one", "Concept_A", 1, 1, "GENE6:GENE9:GENE12"),
    ("B_one", "Concept_B", 4, 2, "GENE21:GENE24:GENE27:GENE1"),
    ("B_two", "Concept_B", 5, 2, "GENE30:GENE33:GENE30"),
    ("B_ghost", "Concept_B", 6, 2, "NOTAGENE"),
]


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """The miniature dataset, with `hub` reading from it."""
    root = write_fake_dataset(tmp_path / "dataset")
    use_fake_dataset(monkeypatch, root)
    return root


@pytest.fixture
def hierarchy_path(tmp_path) -> str:
    """A six-set, three-concept hierarchy table over the fake dataset's genes."""
    path = tmp_path / "hierarchy.tsv"
    path.write_text("".join("\t".join(map(str, row)) + "\n" for row in HIERARCHY))
    return str(path)
