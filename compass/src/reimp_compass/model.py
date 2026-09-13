"""COMPASS's encoder and concept bottleneck, without the cancer-type token.

  scaler     per-gene min-max of log TPM, fit on training samples and kept
             as buffers, so a checkpoint carries its statistics.
  tokenizer  FT-Transformer numerical tokens: token_g = ReLU(x_g · W_g + P_g),
             with a per-gene W_g and P_g in R^d (the paper calls P the
             "learnable positional encoding"). No CLS or cancer-type token.
  encoder    post-norm transformer layers shaped as COMPASS's released
             Performer layer — d = 32, 2 heads of width 32, GELU FFN of 64,
             dropout 0.2 — with exact attention (SDPA) in place of
             Performer's linear attention.
  projector  genes -> sets: a softmax over one learned logit per membership,
             independent of the input, weights the members' contextual
             embeddings, and one Linear(d -> 1) shared by every set scores
             the result. Sets -> concepts: a softmax over one learned logit
             per set weights the set scores.
  loss       triplet loss on the concept vector, cosine distance, margin 1.

Logits start at N(0, 1) and gene tokens at U(±1/√d), as in COMPASS's code.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from reimp_compass.hierarchy import Hierarchy


class MinMaxScaler(nn.Module):
    """Per-gene (x − min) / (max − min), with min and max measured by `fit`.

    As scikit-learn's `MinMaxScaler`, which COMPASS uses: a gene constant on
    the fit samples gets scale 1, and values outside the fitted range are
    not clipped.
    """

    def __init__(self, n_genes: int) -> None:
        super().__init__()
        self.register_buffer("minimum", torch.zeros(n_genes))
        self.register_buffer("scale", torch.ones(n_genes))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, values) -> MinMaxScaler:
        """Measure each gene's range over `values` (samples x genes): the training samples."""
        values = np.asarray(values)
        if values.shape[1:] != self.minimum.shape:
            raise ValueError(f"expected (n, {len(self.minimum)}) values, got {values.shape}")
        low = values.min(axis=0).astype(np.float64)
        span = values.max(axis=0).astype(np.float64) - low
        span[span == 0] = 1.0
        self.minimum.copy_(torch.from_numpy(low))
        self.scale.copy_(torch.from_numpy(span))
        self.fitted.fill_(True)
        return self

    def forward(self, x: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("fit the scaler on the training samples before scaling")
        return (x - self.minimum) / self.scale


class GeneTokenizer(nn.Module):
    """(B, G) values -> (B, G, d) tokens, ReLU(x_g · W_g + P_g) with per-gene W and P."""

    def __init__(self, n_genes: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_genes, d_model))
        self.bias = nn.Parameter(torch.empty(n_genes, d_model))
        bound = d_model**-0.5
        for parameter in (self.weight, self.bias):
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x.unsqueeze(-1) * self.weight + self.bias)


class EncoderLayer(nn.Module):
    """Post-norm transformer layer with exact scaled-dot-product attention.

    `head_dim` is set apart from `d_model / n_heads`: COMPASS's layer runs
    2 heads of 32 on d = 32. With `chunk_size`, queries are attended in
    blocks of that many genes, each recomputed in the backward pass, so
    memory grows with the block rather than with genes². This is for
    backends without a fused kernel (MPS, CPU), where SDPA materializes the
    full G x G attention; the result is the same attention either way.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        dim_ff: int,
        dropout: float,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.n_heads, self.head_dim, self.chunk_size = n_heads, head_dim, chunk_size
        inner = n_heads * head_dim
        self.qkv = nn.Linear(d_model, 3 * inner)
        self.proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def _attention(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        step = self.chunk_size
        if step is None or step >= t:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            blocks = []
            for start in range(0, t, step):
                block = q[:, :, start : start + step]
                if torch.is_grad_enabled():
                    blocks.append(
                        checkpoint(F.scaled_dot_product_attention, block, k, v, use_reentrant=False)
                    )
                else:
                    blocks.append(F.scaled_dot_product_attention(block, k, v))
            out = torch.cat(blocks, dim=2)
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout(self._attention(x)))
        return self.norm2(x + self.ff(x))


def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Row-wise softmax of `logits` broadcast over `mask`'s rows, over its True entries only.

    A row with no True entry is all zeros.
    """
    rows = logits.expand_as(mask).masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(rows, dim=-1) * mask


class ConceptProjector(nn.Module):
    """Contextual gene embeddings -> set scores and concept scores.

    The hierarchy fixes which genes may feed which set and which sets which
    concept; the weights within those are learned. Gene positions are bound
    to a gene selection by `bind`, and kept as buffers.
    """

    def __init__(self, hierarchy: Hierarchy, d_model: int) -> None:
        super().__init__()
        member_set = torch.as_tensor(hierarchy.member_set)
        n_sets, n_concepts = len(hierarchy.sets), len(hierarchy.concepts)
        self.register_buffer(
            "set_member", member_set.unsqueeze(0) == torch.arange(n_sets).unsqueeze(1)
        )
        self.register_buffer(
            "concept_set",
            torch.as_tensor(hierarchy.set_concept).unsqueeze(0)
            == torch.arange(n_concepts).unsqueeze(1),
        )
        self.register_buffer("member_gene", torch.zeros(len(member_set), dtype=torch.long))
        self.register_buffer("member_present", torch.zeros(len(member_set), dtype=torch.bool))
        self.register_buffer("bound", torch.tensor(False))
        self.gene_logits = nn.Parameter(torch.randn(len(member_set)))
        self.set_logits = nn.Parameter(torch.randn(n_sets))
        self.set_scorer = nn.Linear(d_model, 1)

    def bind(self, positions: np.ndarray) -> None:
        """Point each membership at its gene's column; -1 marks a gene the data lacks."""
        positions = torch.as_tensor(np.asarray(positions), dtype=torch.long)
        if positions.shape != self.member_gene.shape:
            raise ValueError(f"expected {len(self.member_gene)} positions, got {len(positions)}")
        self.member_present.copy_(positions >= 0)
        self.member_gene.copy_(positions.clamp_min(0))
        self.bound.fill_(True)

    def set_weights(self) -> Tensor:
        """(n_sets, n_memberships): each set's softmax over its present members."""
        return _masked_softmax(self.gene_logits, self.set_member & self.member_present)

    def concept_weights(self) -> Tensor:
        """(n_concepts, n_sets): each concept's softmax over its sets."""
        return _masked_softmax(self.set_logits, self.concept_set)

    def forward(self, genes: Tensor) -> tuple[Tensor, Tensor]:
        """(B, G, d) gene embeddings -> set scores (B, S) and concept scores (B, C).

        A set none of whose genes the data has scores the scorer's bias.
        """
        if not self.bound:
            raise RuntimeError("bind the hierarchy to the data's genes before projecting")
        members = genes[:, self.member_gene]
        set_features = torch.einsum("sm,bmd->bsd", self.set_weights(), members)
        set_scores = self.set_scorer(set_features).squeeze(-1)
        return set_scores, set_scores @ self.concept_weights().T


class Compass(nn.Module):
    def __init__(
        self,
        n_genes: int,
        hierarchy: Hierarchy,
        d_model: int = 32,
        n_heads: int = 2,
        head_dim: int = 32,
        dim_ff: int = 64,
        n_layers: int = 1,
        dropout: float = 0.2,
        attention_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.scaler = MinMaxScaler(n_genes)
        self.tokenizer = GeneTokenizer(n_genes, d_model)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, head_dim, dim_ff, dropout, attention_chunk_size)
            for _ in range(n_layers)
        )
        self.projector = ConceptProjector(hierarchy, d_model)

    def encode(self, x: Tensor) -> Tensor:
        """(B, G) scaled values -> (B, G, d) contextual gene embeddings."""
        h = self.tokenizer(x)
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(B, G) scaled values -> set scores (B, S) and concept scores (B, C)."""
        return self.projector(self.encode(x))


def cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    return 1 - F.cosine_similarity(x, y, dim=-1)


def triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0) -> Tensor:
    """COMPASS's pretraining loss per triplet, unreduced.

    max(0, d(a, p) − d(a, n) + margin), with d the cosine distance 1 − cos.
    """
    d_pos = cosine_distance(anchor, positive)
    d_neg = cosine_distance(anchor, negative)
    return F.relu(d_pos - d_neg + margin)
