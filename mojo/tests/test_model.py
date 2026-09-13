import pytest
import torch
from torch.nn import functional as F

from reimp_mojo.model import (
    MOJO,
    Block,
    DownBlock,
    UpBlock,
    channel_schedule,
    masked_token_loss,
    padded_length,
    rotary,
)
from reimp_shared.tokens import IGNORE_INDEX

N_BINS = 16


def _model(n_genes: int = 37, n_down: int = 3) -> MOJO:
    torch.manual_seed(0)
    model = MOJO(
        n_genes,
        n_bins=N_BINS,
        embed_dim=16,
        conv_channels=16,
        d_model=32,
        n_down=n_down,
        n_layers=2,
        n_heads=4,
        dim_ff=64,
        stem_kernel=5,
        kernel_size=3,
    )
    return model.eval()


def _tokens(n: int = 3, g: int = 37, seed: int = 0) -> torch.Tensor:
    return torch.randint(N_BINS, (n, g), generator=torch.Generator().manual_seed(seed))


# ---------- length bookkeeping ----------


@pytest.mark.parametrize(
    ("n_genes", "n_down", "expected"),
    [(37, 3, 40), (40, 3, 40), (1, 3, 8), (19_944, 8, 19_968), (17_116, 8, 17_152)],
)
def test_padded_length_is_the_next_multiple_of_2_to_the_halvings(n_genes, n_down, expected):
    assert padded_length(n_genes, n_down) == expected


def test_papers_gene_count_pools_to_its_67_positions() -> None:
    assert padded_length(17_116, 8) // 2**8 == 67


def test_channel_schedule_is_geometric_between_the_ends() -> None:
    assert channel_schedule(32, 32, 4) == [32] * 5
    assert channel_schedule(16, 256, 4) == [16, 32, 64, 128, 256]
    assert channel_schedule(128, 256, 8)[-1] == 256


def test_down_block_halves_the_length_and_keeps_the_skip() -> None:
    pooled, skip = DownBlock(8, 12, 3)(torch.randn(2, 20, 8))
    assert pooled.shape == (2, 10, 12)
    assert skip.shape == (2, 20, 12)
    # Average pooling: each pooled position is the mean of two skip positions.
    torch.testing.assert_close(pooled, skip.view(2, 10, 2, 12).mean(dim=2))


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 15])
def test_up_block_doubles_the_length_for_any_odd_kernel(kernel_size) -> None:
    up = UpBlock(12, 8, kernel_size)
    assert up(torch.randn(2, 10, 12), torch.randn(2, 20, 12)).shape == (2, 20, 8)


def test_even_kernels_are_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        MOJO(40, kernel_size=4)


def test_encoder_pools_the_padded_genes_to_length_over_2_to_the_halvings() -> None:
    model = _model(n_genes=37, n_down=3)
    assert (model.length, model.n_pooled) == (40, 5)
    pooled, skips = model.encode(_tokens())
    assert pooled.shape == (3, 5, 32)
    # Stem, then one skip per halving block, at 40, 40, 20, 10 positions.
    assert [s.shape[1] for s in skips] == [40, 40, 20, 10]
    assert [s.shape[2] for s in skips] == channel_schedule(16, 32, 3)


def test_logits_come_back_at_one_position_per_gene() -> None:
    model = _model(n_genes=37, n_down=3)
    assert model(_tokens()).shape == (3, 37, N_BINS)


def test_wrong_gene_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="37 gene tokens"):
        _model(n_genes=37)(_tokens(g=36))


def test_padding_uses_the_pad_token_and_no_gene_embedding() -> None:
    model = _model(n_genes=37, n_down=3)
    embedded = model._embed_tokens(_tokens())
    assert embedded.shape == (3, 40, 16)
    pad = model.token_embedding.weight[model.pad_id]
    torch.testing.assert_close(embedded[:, 37:], pad.expand(3, 3, 16))


# ---------- embedding ----------


def test_embedding_is_the_mean_over_pooled_positions_of_the_last_layer() -> None:
    model = _model()
    tokens = _tokens()
    with torch.no_grad():
        embedding = model.embed(tokens)
        assert embedding.shape == (3, 32)
        torch.testing.assert_close(embedding, model.encode(tokens)[0].mean(dim=1))


def test_embedding_depends_on_expression_and_on_gene_identity() -> None:
    model = _model()
    tokens = _tokens()
    with torch.no_grad():
        base = model.embed(tokens)
        assert not torch.allclose(base, model.embed((tokens + 1) % N_BINS), atol=1e-4)
        # The same values at other genes: gene embeddings make order matter.
        assert not torch.allclose(base, model.embed(tokens.flip(1)), atol=1e-4)


def test_every_parameter_gets_a_gradient_from_the_logits() -> None:
    model = _model().train()
    model(_tokens()).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


# ---------- transformer ----------


def test_rotary_scores_depend_only_on_the_offset() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 8).expand(1, 1, 12, 8)
    k = torch.randn(1, 1, 1, 8).expand(1, 1, 12, 8)
    scores = (rotary(q) @ rotary(k).transpose(-1, -2))[0, 0]
    for offset in (-3, 0, 5):
        diagonal = scores.diagonal(offset)
        torch.testing.assert_close(diagonal, diagonal[:1].expand_as(diagonal))
    # ... and do change with it.
    assert not torch.allclose(scores.diagonal(0)[0], scores.diagonal(5)[0])


def test_rotary_preserves_norms_and_leaves_position_zero_alone() -> None:
    x = torch.randn(2, 3, 7, 8)
    out = rotary(x)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(out[:, :, 0], x[:, :, 0])


def test_transformer_layer_is_order_sensitive() -> None:
    """Unlike a set encoder, rotary positions make pooled order matter."""
    torch.manual_seed(0)
    block = Block(16, 2, 32, dropout=0.0).eval()
    x = torch.randn(2, 6, 16)
    perm = torch.tensor([3, 1, 5, 0, 2, 4])
    with torch.no_grad():
        assert not torch.allclose(block(x)[:, perm], block(x[:, perm]), atol=1e-4)


def test_block_rejects_odd_head_dims_and_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Block(30, 4, 64, dropout=0.0)
    with pytest.raises(ValueError, match="even head"):
        Block(12, 4, 64, dropout=0.0)


# ---------- loss ----------


def test_masked_token_loss_counts_only_selected_positions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, N_BINS)
    labels = torch.full((2, 5), IGNORE_INDEX)
    labels[0, 1], labels[1, 3] = 4, 7
    loss, accuracy = masked_token_loss(logits, labels)
    expected = F.cross_entropy(logits[[0, 1], [1, 3]], torch.tensor([4, 7]))
    torch.testing.assert_close(loss, expected)
    # Changing the logits of unselected positions changes nothing.
    other = logits.clone()
    other[0, 0] += 10.0
    torch.testing.assert_close(masked_token_loss(other, labels)[0], loss)
    # Accuracy is over the two selected positions.
    logits[0, 1, 4] = logits[1, 3, 7] = 100.0
    assert masked_token_loss(logits, labels)[1].item() == 1.0
