import numpy as np
import pytest
import torch

from reimp_bulkformer.graph import coexpression_graph, gcn_normalize, sparse_adjacency
from reimp_bulkformer.model import (
    MASK_VALUE,
    BulkFormer,
    ExpressionEmbedding,
    graph_conv,
    mask_genes,
    masked_mse,
)

N_GENES = 30
TINY = dict(d_model=16, n_blocks=1, n_layers=1, n_heads=2, dropout=0.0)


def _values(n: int = 4, seed: int = 0) -> torch.Tensor:
    """log1p-TPM-like values with correlated gene blocks."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 3))
    x = np.repeat(factors, N_GENES // 3, axis=1) + 0.3 * rng.normal(size=(n, N_GENES))
    return torch.from_numpy(np.log1p(np.exp(2 + x))).float()


def _model(use_graph: bool = True, seed: int = 0) -> BulkFormer:
    torch.manual_seed(seed)
    model = BulkFormer(N_GENES, use_graph=use_graph, **TINY)
    if use_graph:
        model.set_graph(*coexpression_graph(_values(64).expm1(), k=5))
    return model.eval()


# ---------- masking and loss ----------


def test_mask_genes_masks_the_same_count_in_every_row() -> None:
    mask = mask_genes(8, 200, 0.15)
    assert mask.shape == (8, 200) and mask.dtype == torch.bool
    assert (mask.sum(dim=1) == 30).all()
    assert mask_genes(2, 5, 0.01).sum(dim=1).tolist() == [1, 1]  # at least one


def test_mask_genes_is_reproducible_with_a_generator() -> None:
    a = mask_genes(4, 100, 0.15, generator=torch.Generator().manual_seed(0))
    b = mask_genes(4, 100, 0.15, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


def test_masked_mse_counts_only_masked_positions() -> None:
    target = torch.zeros(2, 4)
    pred = torch.tensor([[1.0, 100.0, 100.0, 100.0], [3.0, 100.0, 100.0, 100.0]])
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 0] = True
    assert masked_mse(pred, target, mask).item() == pytest.approx((1 + 9) / 2)


# ---------- embeddings ----------


def test_expression_embedding_is_a_fixed_sinusoid() -> None:
    embedding = ExpressionEmbedding(8)
    assert not list(embedding.parameters())
    theta = 100.0 ** (-2 * torch.arange(4) / 8)
    x = torch.tensor([[0.0, 2.5]])
    out = embedding(x)
    assert out.shape == (1, 2, 8)
    torch.testing.assert_close(out[0, 1], torch.cat([(2.5 * theta).sin(), (2.5 * theta).cos()]))
    torch.testing.assert_close(out[0, 0], torch.cat([torch.zeros(4), torch.ones(4)]))


def test_expression_embedding_needs_an_even_width() -> None:
    with pytest.raises(ValueError, match="even"):
        ExpressionEmbedding(7)


def test_masked_values_do_not_reach_the_output() -> None:
    """Masked genes are a placeholder everywhere: the true value cannot leak."""
    model = _model()
    values = _values()
    mask = mask_genes(*values.shape, 0.2, generator=torch.Generator().manual_seed(0))
    scrambled = values.clone()
    scrambled[mask] = torch.rand(int(mask.sum())) * 10
    with torch.no_grad():
        torch.testing.assert_close(model(values, mask), model(scrambled, mask))
        # ... while an unmasked change does.
        assert not torch.allclose(model(values, mask), model(values + 0.5, mask))


def test_masked_genes_are_the_placeholder_in_the_sample_context() -> None:
    model = _model()
    values = _values()
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[:, :3] = True
    seen = []
    model.sample_mlp.register_forward_hook(lambda m, args, out: seen.append(args[0]))
    with torch.no_grad():
        model(values, mask)
    assert (seen[0][:, :3] == MASK_VALUE).all()
    torch.testing.assert_close(seen[0][:, 3:], values[:, 3:])


def test_sample_context_width_defaults_to_d_model() -> None:
    model = BulkFormer(N_GENES, **TINY)
    assert model.sample_mlp[0].out_features == TINY["d_model"]
    assert BulkFormer(N_GENES, sample_hidden=40, **TINY).sample_mlp[0].out_features == 40


# ---------- graph ----------


def test_graph_conv_is_a_per_sample_adjacency_product() -> None:
    index, weight = coexpression_graph(_values(64).expm1(), k=5)
    norm = gcn_normalize(index, weight, N_GENES)
    adjacency = sparse_adjacency(index, norm, N_GENES)
    x = torch.randn(3, N_GENES, 4)
    dense = adjacency.to_dense()
    torch.testing.assert_close(graph_conv(adjacency, x), torch.stack([dense @ s for s in x]))


def test_tokens_depend_on_the_graph() -> None:
    model, values = _model(), _values()
    with torch.no_grad():
        before = model.encode(values)
        index, weight = model.graph_index, model.graph_weight
        model.set_graph(index[:, index[0] == index[1]], weight[index[0] == index[1]])
        assert not torch.allclose(model.encode(values), before)


def test_initial_predictions_are_not_all_clamped() -> None:
    """The head starts above ReLU's flat zero, so every gene gets a gradient."""
    for seed in range(5):
        with torch.no_grad():
            assert (_model(seed=seed)(_values()) > 0).float().mean() > 0.9


def test_a_graph_model_needs_its_graph() -> None:
    model = BulkFormer(N_GENES, **TINY)
    assert not model.has_graph
    with pytest.raises(RuntimeError, match="set_graph"):
        model(_values())
    with pytest.raises(ValueError, match="n_genes"):
        model.set_graph(torch.tensor([[0], [N_GENES]]), torch.ones(1))


def test_performer_only_model_runs_without_a_graph() -> None:
    model = _model(use_graph=False)
    assert model.blocks[0].gcn is None
    with torch.no_grad():
        assert model(_values()).shape == (4, N_GENES)


def test_graph_buffers_are_saved_and_the_adjacency_is_not() -> None:
    state = _model().state_dict()
    assert state["graph_index"].shape[0] == 2 and state["graph_index"].numel() > 0
    assert "adjacency" not in state


# ---------- outputs ----------


def test_predictions_are_non_negative_per_gene() -> None:
    with torch.no_grad():
        out = _model()(_values(), mask_genes(4, N_GENES, 0.15))
    assert out.shape == (4, N_GENES)
    assert (out >= 0).all()


def test_embedding_pools_final_gene_tokens() -> None:
    model, values = _model(), _values()
    with torch.no_grad():
        tokens = model.encode(values)
        top, mean = model.embed(values, "max"), model.embed(values, "mean")
    assert tokens.shape == (4, N_GENES, TINY["d_model"])
    torch.testing.assert_close(top, tokens.amax(dim=1))
    torch.testing.assert_close(mean, tokens.mean(dim=1))
    with pytest.raises(ValueError, match="pooling"):
        model.embed(values, "median")


def test_activation_checkpointing_gives_the_same_gradients() -> None:
    values = _values()
    mask = mask_genes(*values.shape, 0.2, generator=torch.Generator().manual_seed(0))
    grads = []
    for flag in (False, True):
        model = _model().train()
        for block in model.blocks:
            block.activation_checkpointing = flag
        masked_mse(model(values, mask), values, mask).backward()
        grads.append(model.gene_embedding.weight.grad)
    torch.testing.assert_close(grads[0], grads[1])
