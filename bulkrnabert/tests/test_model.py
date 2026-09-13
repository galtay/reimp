import pytest
import torch
from torch.nn import functional as F

from reimp_bulkrnabert.model import Block, BulkRNABert, mlm_accuracy, mlm_loss
from reimp_shared.tokens import IGNORE_INDEX

N_BINS = 64


def _model(n_genes: int = 30) -> BulkRNABert:
    torch.manual_seed(0)
    return BulkRNABert(n_genes, n_bins=N_BINS, d_model=32, n_layers=2, n_heads=4, dim_ff=64).eval()


# ---------- loss ----------


def test_mlm_loss_is_cross_entropy_over_selected_positions_only() -> None:
    logits = torch.randn(2, 5, N_BINS)
    labels = torch.full((2, 5), IGNORE_INDEX)
    labels[0, 1], labels[1, 3] = 7, 0
    expected = F.cross_entropy(logits[[0, 1], [1, 3]], torch.tensor([7, 0]))
    torch.testing.assert_close(mlm_loss(logits, labels), expected)
    # Logits where nothing was selected do not enter the loss.
    changed = logits.clone()
    changed[0, 0] += 100.0
    torch.testing.assert_close(mlm_loss(changed, labels), expected)


def test_mlm_accuracy_counts_selected_positions_only() -> None:
    logits = torch.zeros(1, 4, N_BINS)
    logits[0, torch.arange(4), torch.tensor([5, 6, 7, 8])] = 1.0  # predicts 5, 6, 7, 8
    labels = torch.tensor([[5, 0, IGNORE_INDEX, IGNORE_INDEX]])
    assert mlm_accuracy(logits, labels).item() == 0.5


# ---------- architecture ----------


def test_shapes_and_mask_token() -> None:
    model = _model()
    tokens = torch.randint(0, N_BINS + 1, (3, 30))  # the mask token, N_BINS, included
    assert model.mask_id == N_BINS
    assert model.token_embedding.num_embeddings == N_BINS + 1
    # The head predicts a bin, never the mask token.
    assert model(tokens).shape == (3, 30, N_BINS)
    assert model.encode(tokens).shape == (3, 30, 32)


def test_embedding_is_the_mean_over_genes_of_the_final_hidden_states() -> None:
    model = _model()
    tokens = torch.randint(0, N_BINS, (3, 30))
    with torch.no_grad():
        torch.testing.assert_close(model.embed(tokens), model.encode(tokens).mean(dim=1))


def test_no_positional_encoding_beyond_the_gene_embedding() -> None:
    """Reorder the genes and their embeddings together: the sample embedding is unchanged."""
    model = _model()
    tokens = torch.randint(0, N_BINS, (3, 30))
    perm = torch.randperm(30)
    with torch.no_grad():
        before = model.embed(tokens)
        model.gene_embedding.weight.copy_(model.gene_embedding.weight[perm])
        after = model.embed(tokens[:, perm])
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_gene_identity_distinguishes_equal_tokens() -> None:
    """Every gene at the same bin still gets its own hidden state, from its gene embedding."""
    model = _model()
    with torch.no_grad():
        hidden = model.encode(torch.full((1, 30), 5))
    assert not torch.allclose(hidden[0, 0], hidden[0, 1], atol=1e-3)


def test_encoder_rejects_the_wrong_number_of_genes() -> None:
    with pytest.raises(ValueError, match="30 gene tokens"):
        _model().encode(torch.zeros(2, 29, dtype=torch.long))


def test_block_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Block(30, 4, 64, dropout=0.0)


@pytest.mark.parametrize("chunk", [1, 7, 30, 100])
def test_chunked_attention_is_the_same_attention(chunk) -> None:
    """Chunking the queries changes the kernels' sizes, not the result or its gradients."""
    whole = _model()
    chunked = _model()
    for block in chunked.blocks:
        block.attn_chunk = chunk
    tokens = torch.randint(0, N_BINS, (3, 30))
    torch.testing.assert_close(chunked(tokens), whole(tokens), atol=1e-5, rtol=1e-5)
    whole(tokens).sum().backward()
    chunked(tokens).sum().backward()
    torch.testing.assert_close(
        chunked.gene_embedding.weight.grad, whole.gene_embedding.weight.grad, atol=1e-5, rtol=1e-4
    )


def test_block_rejects_an_empty_chunk() -> None:
    with pytest.raises(ValueError, match="attn_chunk"):
        Block(32, 4, 64, dropout=0.0, attn_chunk=0)
