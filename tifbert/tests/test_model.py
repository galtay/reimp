import pytest
import torch
from torch.nn import functional as F

from reimp_shared.tokens import IGNORE_INDEX
from reimp_tifbert.model import TifBERT, mask_genes, mean_pool, mlm_loss

N_GENES = 50


def _model(**kwargs) -> TifBERT:
    torch.manual_seed(0)
    config = dict(window=16, d_model=32, n_layers=2, n_heads=4, dim_ff=64, dropout=0.0)
    return TifBERT(N_GENES, **{**config, **kwargs}).eval()


# ---------- masking ----------


def test_mask_genes_masks_15_percent_of_genes_and_never_padding() -> None:
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(N_GENES, (400, 64), generator=generator)
    tokens[:, 48:] = N_GENES  # padding
    inputs, labels = mask_genes(tokens, N_GENES, generator=generator)
    pad = tokens == N_GENES
    assert torch.equal(inputs[pad], tokens[pad])
    assert (labels[pad] == IGNORE_INDEX).all()
    selected = labels != IGNORE_INDEX
    assert selected[~pad].float().mean().item() == pytest.approx(0.15, abs=0.01)
    assert torch.equal(labels[selected], tokens[selected])
    masked = inputs == N_GENES + 1
    assert (masked <= selected).all()
    assert (masked.sum() / selected.sum()).item() == pytest.approx(0.8, abs=0.02)
    # Random replacements are genes, never a special token.
    assert inputs[selected & ~masked].max().item() < N_GENES


def test_mask_genes_is_reproducible_with_a_generator() -> None:
    tokens = torch.randint(N_GENES, (4, 32))
    a = mask_genes(tokens, N_GENES, generator=torch.Generator().manual_seed(3))
    b = mask_genes(tokens, N_GENES, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------- loss and pooling ----------


def test_mlm_loss_is_cross_entropy_over_the_masked_positions() -> None:
    logits, labels = torch.randn(7, N_GENES + 2), torch.randint(N_GENES, (7,))
    torch.testing.assert_close(mlm_loss(logits, labels), F.cross_entropy(logits, labels))


def test_mlm_loss_with_nothing_masked_is_zero_and_differentiable() -> None:
    logits = torch.randn(0, N_GENES + 2, requires_grad=True)
    loss = mlm_loss(logits, torch.zeros(0, dtype=torch.long))
    assert loss.item() == 0.0
    loss.backward()


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.arange(12.0).view(1, 4, 3)
    tokens = torch.tensor([[3, 7, N_GENES, N_GENES]])
    torch.testing.assert_close(mean_pool(hidden, tokens, N_GENES), hidden[:, :2].mean(dim=1))


# ---------- architecture ----------


def test_padding_does_not_change_the_genes_hidden_states() -> None:
    model = _model()
    tokens = torch.tensor([[4, 9, 1, 30, 22]])
    padded = F.pad(tokens, (0, 6), value=N_GENES)
    with torch.no_grad():
        torch.testing.assert_close(model(padded)[:, :5], model(tokens), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(model.embed(padded), model.embed(tokens), atol=1e-5, rtol=1e-5)


def test_the_encoder_reads_rank_order() -> None:
    """A gene's position is its rank: the same genes in another order are another sample."""
    model = _model()
    tokens = torch.arange(10, 22).unsqueeze(0)
    with torch.no_grad():
        assert not torch.allclose(model.embed(tokens), model.embed(tokens.flip(1)), atol=1e-3)


def test_logits_cover_the_vocabulary_through_the_tied_embedding() -> None:
    model = _model()
    logits = model.logits(model(torch.randint(N_GENES, (2, 8))))
    assert logits.shape == (2, 8, N_GENES + 2)
    logits.sum().backward()
    # The decoder is the token embedding: no separate output matrix.
    assert model.token_embedding.weight.grad is not None
    assert sum(p.numel() for p in model.parameters() if p.shape[:1] == (N_GENES + 2,)) == (
        model.token_embedding.weight.numel() + model.head_bias.numel()
    )


def test_bert_initialization() -> None:
    model = TifBERT(N_GENES, window=16, d_model=256, n_layers=1, n_heads=4, dim_ff=512)
    assert (model.token_embedding.weight[N_GENES] == 0).all()
    assert model.encoder.layers[0].self_attn.in_proj_weight.std().item() == pytest.approx(
        0.02, rel=0.05
    )
    assert model.head[0].weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_heads_must_divide_the_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TifBERT(N_GENES, d_model=30, n_heads=4)
