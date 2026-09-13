import numpy as np
import pytest
import torch

from reimp_shared.ranking import GeneRanker
from reimp_tifbert.sequences import all_windows, n_windows, rank_genes, sample_windows

PAD = 10_000


def _values(seed: int = 0, n: int = 6, g: int = 30) -> np.ndarray:
    """Skewed, positive, with a fifth of entries unexpressed — like TPM."""
    rng = np.random.default_rng(seed)
    values = rng.gamma(0.5, 10.0, size=(n, g))
    values[rng.random((n, g)) < 0.2] = 0.0
    return values


def _sentences(lengths: list[int], width: int) -> torch.Tensor:
    """Row i holds tokens 1000·i + 0, 1, ... for its length, then padding."""
    genes = torch.full((len(lengths), width), PAD)
    for i, n in enumerate(lengths):
        genes[i, :n] = 1000 * i + torch.arange(n)
    return genes


# ---------- ranking ----------


def test_a_sentence_is_the_rankers_order_of_the_expressed_genes() -> None:
    values = _values()
    ranker = GeneRanker().fit(values)
    assert (ranker.weight_ > 0).all()
    genes, lengths = rank_genes(ranker, values, None, PAD)
    order = ranker.order(values)
    assert genes.dtype == np.int64 and genes.shape == values.shape
    for row, n in enumerate(lengths):
        expressed = values[row] > 0
        assert n == expressed.sum()
        assert genes[row, :n].tolist() == [g for g in order[row] if expressed[g]]
        assert (genes[row, n:] == PAD).all()


def test_genes_never_expressed_in_training_are_left_out() -> None:
    train, test = _values(0), _values(1)
    train[:, 4] = 0.0
    test[:, 4] = 50.0
    ranker = GeneRanker().fit(train)
    genes, lengths = rank_genes(ranker, test, None, PAD)
    assert not (genes == 4).any()
    assert (lengths == ((test > 0).sum(axis=1) - 1)).all()


def test_scores_that_go_negative_still_keep_only_expressed_genes_in_order() -> None:
    values = _values()
    ranker = GeneRanker("z").fit(values)
    genes, lengths = rank_genes(ranker, values, None, PAD)
    for row, n in enumerate(lengths):
        kept = genes[row, :n]
        assert n == (values[row] > 0).sum()
        assert (values[row, kept] > 0).all()
        assert (np.diff(ranker.scores(values[row : row + 1])[0, kept]) <= 0).all()


def test_max_genes_keeps_the_top_ranked() -> None:
    values = _values()
    ranker = GeneRanker().fit(values)
    full, full_lengths = rank_genes(ranker, values, None, PAD)
    top, top_lengths = rank_genes(ranker, values, 5, PAD)
    assert top.shape == (len(values), 5)
    np.testing.assert_array_equal(top, full[:, :5])
    np.testing.assert_array_equal(top_lengths, np.minimum(full_lengths, 5))


# ---------- windows ----------


@pytest.mark.parametrize(
    ("n", "expected"), [(0, 0), (1, 1), (512, 1), (513, 2), (768, 2), (769, 3), (10_000, 39)]
)
def test_window_count_reaches_the_end_of_the_sentence(n, expected) -> None:
    """The paper's ~10,000 genes at 512 / 256 make 39 windows."""
    assert n_windows(torch.tensor([n]), 512, 256).item() == expected


def test_all_windows_tile_every_sentence_from_the_start() -> None:
    lengths = [0, 3, 8, 11]
    genes = _sentences(lengths, 12)
    tokens, rows = all_windows(genes, torch.tensor(lengths), window=4, stride=2, pad_id=PAD)
    assert rows.tolist() == [1, 2, 2, 2, 3, 3, 3, 3, 3]
    starts = [0, 0, 2, 4, 0, 2, 4, 6, 8]
    padded = torch.nn.functional.pad(genes, (0, 4), value=PAD)
    for window, row, start in zip(tokens, rows, starts, strict=True):
        assert torch.equal(window, padded[row, start : start + 4])
    for row, n in enumerate(lengths):
        seen = set(tokens[rows == row].flatten().tolist()) - {PAD}
        assert seen == set(genes[row, :n].tolist())


def test_sample_windows_draws_each_window_uniformly_and_skips_empty_sentences() -> None:
    lengths = torch.tensor([11] * 2000 + [0])
    genes = _sentences([11], 12).repeat(2001, 1)
    tokens, rows = sample_windows(genes, lengths, window=4, stride=2, pad_id=PAD)
    assert rows.tolist() == list(range(2000))
    starts = tokens[:, 0]  # row 0's tokens are their own positions
    assert set(starts.tolist()) == {0, 2, 4, 6, 8}
    shares = torch.bincount(starts)[[0, 2, 4, 6, 8]] / 2000
    assert ((shares - 0.2).abs() < 0.04).all()


def test_sample_windows_is_reproducible_with_a_generator() -> None:
    genes, lengths = _sentences([11, 7, 30], 30), torch.tensor([11, 7, 30])

    def draw() -> torch.Tensor:
        generator = torch.Generator().manual_seed(0)
        return sample_windows(genes, lengths, 4, 2, PAD, generator)[0]

    assert torch.equal(draw(), draw())
