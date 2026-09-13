import numpy as np
import pytest
import torch

from reimp_bulkformer.graph import (
    coexpression_graph,
    gcn_normalize,
    shuffle_genes,
    sparse_adjacency,
)

N_BLOCKS, BLOCK = 3, 4


def _blocks(n: int = 200, seed: int = 0) -> np.ndarray:
    """Genes in blocks of four sharing a latent factor (within-block r ≈ 0.9), then a constant."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, N_BLOCKS))
    x = np.repeat(factors, BLOCK, axis=1) + 0.3 * rng.normal(size=(n, N_BLOCKS * BLOCK))
    return np.concatenate([x, np.full((n, 1), 5.0)], axis=1)


def _dense(index: torch.Tensor, weight: torch.Tensor, n: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(index, weight, (n, n)).to_dense()


def test_edges_join_correlated_genes_with_weight_abs_r() -> None:
    x = _blocks()
    index, weight = coexpression_graph(x, k=BLOCK, threshold=0.4)
    a = _dense(index, weight, x.shape[1])
    block = np.arange(N_BLOCKS * BLOCK) // BLOCK
    for i in range(len(block)):
        assert set(np.flatnonzero(a[i]).tolist()) == set(np.flatnonzero(block == block[i]).tolist())
    r = np.abs(np.corrcoef(x[:, :-1], rowvar=False))
    src, dst = index.numpy()
    off = src != dst
    np.testing.assert_allclose(weight.numpy()[off], r[src[off], dst[off]], rtol=1e-4)


def test_every_gene_has_a_self_edge_and_a_constant_gene_only_that() -> None:
    x = _blocks()
    n = x.shape[1]
    index, weight = coexpression_graph(x, k=BLOCK)
    a = _dense(index, weight, n)
    torch.testing.assert_close(a.diagonal(), torch.ones(n))
    assert np.flatnonzero(a[-1]).tolist() == [n - 1]


def test_graph_is_symmetric_and_sorted() -> None:
    index, weight = coexpression_graph(_blocks(), k=3)
    a = _dense(index, weight, 13)
    torch.testing.assert_close(a, a.T)
    keys = index[0] * 13 + index[1]
    assert torch.equal(keys, keys.sort().values)


def test_negative_correlation_is_an_edge() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    x = np.stack([a, -a + 0.1 * rng.normal(size=100), rng.normal(size=100)], axis=1)
    index, weight = coexpression_graph(x, k=2)
    edges = dict(zip(map(tuple, index.T.tolist()), weight.tolist(), strict=True))
    assert edges[(0, 1)] > 0.9 and edges[(1, 0)] > 0.9
    assert (0, 2) not in edges and (1, 2) not in edges


def test_k_counts_the_self_edge_and_caps_each_genes_own_choices() -> None:
    """One block of ten correlated genes: each keeps itself and k − 1 others."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 1)) + 0.5 * rng.normal(size=(300, 10))
    index, _ = coexpression_graph(x, k=3)
    degree = torch.bincount(index[0], minlength=10)
    # Own choices (k − 1 = 2) plus self, plus genes that chose this one.
    assert (degree >= 3).all()
    assert len(index[0]) <= 10 * 1 + 2 * 10 * 2


def test_threshold_drops_weak_edges() -> None:
    index, _ = coexpression_graph(_blocks(), k=BLOCK, threshold=0.99)
    assert torch.equal(index[0], index[1])  # within-block r ≈ 0.92: self-edges only


def test_chunking_does_not_change_the_graph() -> None:
    x = _blocks()
    whole = coexpression_graph(x, k=BLOCK, chunk_size=1024)
    chunked = coexpression_graph(x, k=BLOCK, chunk_size=5)
    assert torch.equal(whole[0], chunked[0])
    torch.testing.assert_close(whole[1], chunked[1])


def test_more_neighbours_than_genes() -> None:
    index, _ = coexpression_graph(_blocks(), k=50)
    assert index.max() < 13


def test_gcn_normalize_is_symmetric_degree_normalization() -> None:
    x = _blocks()
    n = x.shape[1]
    index, weight = coexpression_graph(x, k=BLOCK)
    a = _dense(index, weight, n)
    d = a.sum(dim=1).rsqrt()
    expected = d[:, None] * a * d[None, :]
    torch.testing.assert_close(_dense(index, gcn_normalize(index, weight, n), n), expected)


def test_sparse_adjacency_multiplies_like_its_dense_form() -> None:
    index, weight = coexpression_graph(_blocks(), k=BLOCK)
    adjacency = sparse_adjacency(index, weight, 13)
    x = torch.randn(13, 5)
    torch.testing.assert_close(torch.sparse.mm(adjacency, x), _dense(index, weight, 13) @ x)


def test_shuffled_genes_keep_the_graphs_shape_but_not_its_genes() -> None:
    index, weight = coexpression_graph(_blocks(), k=BLOCK)
    shuffled = shuffle_genes(index, 13, torch.Generator().manual_seed(1))
    degrees = torch.bincount(index[0], minlength=13).sort().values
    assert torch.equal(torch.bincount(shuffled[0], minlength=13).sort().values, degrees)
    assert torch.equal(shuffled[0] == shuffled[1], index[0] == index[1])
    assert not torch.equal(shuffled, index)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least 2 samples"):
        coexpression_graph(np.ones((1, 4)))
    with pytest.raises(ValueError, match="self-edge"):
        coexpression_graph(_blocks(), k=0)
