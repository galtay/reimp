from reimp_plier.prior import prior_matrix
from reimp_shared.genesets import GeneSets


def test_prior_matrix_maps_symbols_to_genes_and_counts_the_unmapped() -> None:
    sets = GeneSets(["A", "B"], ["", ""], [["X", "Y", "Q"], ["Y", "Z"]])
    # "Y" names two genes, as a few GENCODE gene_names do: both are members.
    prior = prior_matrix(sets, ["X", "Y", "Z", "Y", "W"])
    assert prior.matrix.tolist() == [[1, 0], [1, 1], [0, 1], [1, 1], [0, 0]]
    assert prior.names == ["A", "B"]
    assert prior.symbols == 4
    assert prior.unmapped == ["Q"]
