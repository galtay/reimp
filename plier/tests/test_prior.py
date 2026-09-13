from collections import Counter
from pathlib import Path

import pytest

from reimp_plier.prior import GeneSets, prior_matrix, read_gmt, write_gmt

PRIORS = Path(__file__).parents[1] / "priors"


def test_gmt_round_trip(tmp_path) -> None:
    sets = GeneSets(["A", "B cells"], ["one", "two"], [["X", "Y"], ["Z"]])
    assert read_gmt(write_gmt(tmp_path / "sets.gmt", sets)) == sets


def test_read_gmt_drops_repeated_members_and_rejects_repeated_names(tmp_path) -> None:
    path = tmp_path / "sets.gmt"
    path.write_text("A\td\tX\tY\tX\n\nB\td\tZ\n")
    assert read_gmt(path).members == [["X", "Y"], ["Z"]]
    path.write_text("A\td\tX\nA\td\tY\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_gmt(path)


def test_prior_matrix_maps_symbols_to_genes_and_counts_the_unmapped() -> None:
    sets = GeneSets(["A", "B"], ["", ""], [["X", "Y", "Q"], ["Y", "Z"]])
    # "Y" names two genes, as a few GENCODE gene_names do: both are members.
    prior = prior_matrix(sets, ["X", "Y", "Z", "Y", "W"])
    assert prior.matrix.tolist() == [[1, 0], [1, 1], [0, 1], [1, 1], [0, 0]]
    assert prior.names == ["A", "B"]
    assert prior.symbols == 4
    assert prior.unmapped == ["Q"]


def test_recommended_prior_is_the_papers_selection() -> None:
    """IRIS/DMAP + LM22 + canonicalPathways without REACTOME and PID (paper.md, For reimp)."""
    sets = read_gmt(PRIORS / "recommended.gmt")
    sources = Counter(description.split("@")[0] for description in sets.descriptions)
    assert sources == {
        "PLIER::bloodCellMarkersIRISDMAP": 61,
        "PLIER::svmMarkers": 22,
        "PLIER::canonicalPathways": 177,
    }
    assert not any(name.startswith(("REACTOME_", "PID_")) for name in sets.names)
    assert len({gene for members in sets.members for gene in members}) == 4932
    assert sum(len(members) >= 10 for members in sets.members) == 257
