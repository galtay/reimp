"""FAVOR+ attention (Choromanski et al., "Rethinking Attention with Performers", ICLR 2021).

Softmax attention costs O(N²) in the number of tokens, and BulkFormer's
tokens are every gene of a sample, ~20k. FAVOR+ replaces the softmax kernel
exp(q·k / √d) with an inner product of positive random features,
φ(q)·φ(k), unbiased for it:

  φ(x) = exp(W x' − |x'|² / 2) / √m,   x' = x / d^¼,

with W's m rows orthogonal random Gaussian directions. Attention is then
D⁻¹ φ(Q) (φ(K)ᵀ V), with D = diag(φ(Q) φ(K)ᵀ 1): O(N · m · d), never forming
the N × N matrix. Non-causal, as BulkFormer's genes are a set.

A reimplementation of what BulkFormer imports from performer-pytorch 1.1.4
(unmaintained): the same feature map and stabilizers, one random projection
shared across heads, redrawn on demand (`redraw_features`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def default_n_features(dim_head: int) -> int:
    """performer-pytorch's default: d · ln d random features per head."""
    return max(1, int(dim_head * math.log(dim_head))) if dim_head > 1 else 1


def orthogonal_features(
    n_features: int, dim: int, generator: torch.Generator | None = None
) -> Tensor:
    """(n_features, dim) random projection: orthogonal blocks with Gaussian row norms.

    Rows come in blocks of `dim` mutually orthogonal directions (QR of a
    Gaussian matrix); each row is then scaled to the norm of an independent
    Gaussian vector, so every row is marginally N(0, I) — which keeps the
    kernel estimate unbiased — while orthogonality within a block lowers its
    variance. Drawn on the CPU.
    """
    blocks = []
    for _ in range(math.ceil(n_features / dim)):
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
        blocks.append(q.T)
    directions = torch.cat(blocks)[:n_features]
    norms = torch.randn(n_features, dim, generator=generator).norm(dim=1, keepdim=True)
    return directions * norms


def softmax_features(x: Tensor, projection: Tensor, is_query: bool, eps: float = 1e-6) -> Tensor:
    """Positive random features of (..., N, d) queries or keys -> (..., N, m).

    Subtracting a maximum keeps exp from overflowing. It cancels between
    attention's numerator and denominator as long as it is shared by
    everything one normalization divides: per row for queries, over all
    keys of a head for keys. `eps` keeps every feature positive. Keys are
    scaled by their largest feature over the whole sequence, so a typical
    key feature can be ~1e-4 of it; performer-pytorch's eps of 1e-4 then
    pulls attention towards uniform, and the estimate stops improving with
    more features. 1e-6 does not (tests/test_favor.py).
    """
    dim, n_features = x.shape[-1], projection.shape[0]
    x = x * dim**-0.25
    logits = x @ projection.T
    half_sq_norm = x.pow(2).sum(dim=-1, keepdim=True) / 2
    dims = (-1,) if is_query else (-1, -2)
    stabilizer = logits.amax(dim=dims, keepdim=True).detach()
    return (torch.exp(logits - half_sq_norm - stabilizer) + eps) * n_features**-0.5


def linear_attention(q_prime: Tensor, k_prime: Tensor, v: Tensor) -> Tensor:
    """D⁻¹ φ(Q) (φ(K)ᵀ V) from features (..., N, m) and values (..., N, d)."""
    context = k_prime.transpose(-2, -1) @ v
    normalizer = q_prime @ k_prime.sum(dim=-2).unsqueeze(-1)
    return (q_prime @ context) / normalizer


class FavorAttention(nn.Module):
    """Multi-head self-attention with FAVOR+ in place of the softmax."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_features: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.dim_head = d_model // n_heads
        self.n_features = n_features or default_n_features(self.dim_head)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Persistent: a checkpoint keeps the draw its weights were trained with.
        self.register_buffer("projection", orthogonal_features(self.n_features, self.dim_head))

    @torch.no_grad()
    def redraw_features(self, generator: torch.Generator | None = None) -> None:
        fresh = orthogonal_features(self.n_features, self.dim_head, generator)
        self.projection.copy_(fresh.to(self.projection))

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q = softmax_features(qkv[0], self.projection, is_query=True)
        k = softmax_features(qkv[1], self.projection, is_query=False)
        out = linear_attention(q, k, qkv[2])
        return self.dropout(self.proj(out.transpose(1, 2).reshape(b, n, d)))
