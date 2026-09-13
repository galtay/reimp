import math

import pytest
import torch

from reimp_bulkformer.favor import (
    FavorAttention,
    default_n_features,
    linear_attention,
    orthogonal_features,
    softmax_features,
)


def _softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return scores.softmax(dim=-1) @ v


def _qkv(seed: int = 0, n: int = 24, d: int = 8) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(2, n, d, generator=g) for _ in range(3))


def _favor(q, k, v, n_features: int, seed: int = 0) -> torch.Tensor:
    projection = orthogonal_features(n_features, q.shape[-1], torch.Generator().manual_seed(seed))
    return linear_attention(
        softmax_features(q, projection, is_query=True),
        softmax_features(k, projection, is_query=False),
        v,
    )


def test_orthogonal_features_are_orthogonal_within_a_block() -> None:
    w = orthogonal_features(20, 8, torch.Generator().manual_seed(0))
    assert w.shape == (20, 8)
    for block in (w[:8], w[8:16], w[16:]):
        gram = block @ block.T
        torch.testing.assert_close(
            gram - gram.diag().diag(), torch.zeros_like(gram), atol=1e-4, rtol=0
        )


def test_default_n_features_is_d_ln_d() -> None:
    assert default_n_features(80) == int(80 * math.log(80))
    assert default_n_features(1) == 1


def test_linear_attention_is_normalized_kernel_attention() -> None:
    g = torch.Generator().manual_seed(0)
    q_prime, k_prime = torch.rand(2, 10, 6, generator=g), torch.rand(2, 10, 6, generator=g)
    v = torch.randn(2, 10, 3, generator=g)
    kernel = q_prime @ k_prime.transpose(-2, -1)
    expected = (kernel / kernel.sum(dim=-1, keepdim=True)) @ v
    torch.testing.assert_close(linear_attention(q_prime, k_prime, v), expected)


# The model's regime: thousands of gene tokens, 32-d heads (tcga.yaml: d 256,
# 8 heads) and the default feature count, d_head · ln d_head = 110.
N_GENES, DIM_HEAD = 4096, 32
DEFAULT_M = default_n_features(DIM_HEAD)


def _relative_error(scale: float, n_features: int, eps: float | None = None) -> float:
    """FAVOR+'s mean absolute error on softmax attention, over uniform attention's.

    Queries and keys are Gaussian times `scale`: the estimate's variance
    grows like exp(|q + k|² / √d), so they are kept small, as trained ones
    are.
    """
    q, k, v = _qkv(n=N_GENES, d=DIM_HEAD)
    q, k = q * scale, k * scale
    exact = _softmax_attention(q, k, v)
    projection = orthogonal_features(n_features, DIM_HEAD, torch.Generator().manual_seed(0))
    kwargs = {} if eps is None else {"eps": eps}
    approx = linear_attention(
        softmax_features(q, projection, is_query=True, **kwargs),
        softmax_features(k, projection, is_query=False, **kwargs),
        v,
    )
    uniform = v.mean(dim=1, keepdim=True).expand_as(exact)
    return ((approx - exact).abs().mean() / (uniform - exact).abs().mean()).item()


def test_favor_approximates_softmax_attention_over_thousands_of_genes() -> None:
    """The estimate converges on softmax attention as features grow from the default.

    At the default count alone, with random rather than trained queries
    and keys, it is about as far from softmax attention as uniform
    attention is; 16 times as many features halve that.
    """
    errors = [_relative_error(0.5, m) for m in (DEFAULT_M, 4 * DEFAULT_M, 16 * DEFAULT_M)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] < 2.0 and errors[2] < 0.5


@pytest.mark.parametrize("scale", [0.7, 1.0])
def test_performer_eps_beats_a_smaller_one_at_the_default_feature_count(scale) -> None:
    """eps = 1e-4 is at least as close to softmax attention as 1e-6, over thousands of genes.

    Keys are scaled by their largest feature over the sequence, so eps
    is comparable to a typical key feature and shrinks the estimate
    towards uniform attention: less variance, some bias. At 110 features
    the variance is what matters.
    """
    default = _relative_error(scale, DEFAULT_M)
    assert default == _relative_error(scale, DEFAULT_M, eps=1e-4)
    assert default < 1.6 and default < 0.8 * _relative_error(scale, DEFAULT_M, eps=1e-6)


def test_features_are_positive_and_stabilizers_cancel() -> None:
    q, k, v = _qkv(n=6)
    projection = orthogonal_features(64, 8, torch.Generator().manual_seed(0))
    features = softmax_features(q, projection, is_query=True)
    assert (features > 0).all()
    # A stabilizer scales each query's features by a constant: that must
    # not change attention.
    shifted = linear_attention(
        softmax_features(q, projection, is_query=True) * 7.0,
        softmax_features(k, projection, is_query=False),
        v,
    )
    torch.testing.assert_close(shifted, _favor(q, k, v, 64))


def test_attention_is_permutation_equivariant() -> None:
    """No positions: permuting the genes permutes the outputs."""
    torch.manual_seed(0)
    attention = FavorAttention(16, n_heads=2).eval()
    x = torch.randn(2, 30, 16)
    perm = torch.randperm(30)
    with torch.no_grad():
        torch.testing.assert_close(
            attention(x)[:, perm], attention(x[:, perm]), atol=1e-5, rtol=1e-4
        )


def test_redraw_changes_the_projection_and_output() -> None:
    torch.manual_seed(0)
    attention = FavorAttention(16, n_heads=2).eval()
    x = torch.randn(1, 12, 16)
    with torch.no_grad():
        before, projection = attention(x), attention.projection.clone()
        assert torch.equal(attention(x), before)
        attention.redraw_features(torch.Generator().manual_seed(1))
        assert not torch.equal(attention.projection, projection)
        assert not torch.allclose(attention(x), before)


def test_projection_is_saved_with_the_weights() -> None:
    attention = FavorAttention(16, n_heads=2, n_features=10)
    assert attention.state_dict()["projection"].shape == (10, 8)


def test_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        FavorAttention(30, n_heads=4)
