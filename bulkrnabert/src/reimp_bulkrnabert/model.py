"""BulkRNABert architecture and masked-token loss.

  tokens    one per gene: the gene's log TPM binned by `reimp_shared.tokens`'
            `BinTokenizer` (bins 0 … n_bins − 1), plus a mask token, n_bins.
  embedding the token's expression embedding plus a learned embedding of
            the gene at that position. No positional encoding: genes sit in
            a fixed order, and the gene embedding is their only identity.
  encoder   pre-norm transformer blocks over all G gene tokens, then a final
            LayerNorm.
  head      a linear layer from each gene's hidden state to n_bins logits.
  loss      cross-entropy at the positions selected for corruption.
  embedding the mean over genes of the final hidden states.

The paper (§3.1) gives 4 layers, 8 heads, d = 256 and an FFN of 512, with a
Gene2Vec gene embedding projected to d. Here the gene embedding is learned
from scratch (paper.md, "For reimp"). Pre-norm blocks, GELU, BERT's N(0, 0.02)
embedding init and the plain linear head are ours; the paper does not give
them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reimp_shared.tokens import DEFAULT_N_BINS, IGNORE_INDEX


def mlm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean cross-entropy over the selected positions.

    `logits` is (B, G, n_bins); `labels` is (B, G), the original token where
    a position was selected and `IGNORE_INDEX` elsewhere, as `mask_tokens`
    returns them.
    """
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX)


def mlm_accuracy(logits: Tensor, labels: Tensor) -> Tensor:
    """Share of selected positions whose most likely bin is the original one."""
    selected = labels != IGNORE_INDEX
    return (logits.argmax(dim=-1)[selected] == labels[selected]).float().mean()


class Block(nn.Module):
    """Pre-norm transformer block: self-attention over every gene, then a GELU FFN.

    `attn_chunk` splits the queries into chunks of that many genes, each
    attending to every key. The result is the same — softmax runs over keys
    — but no single attention call holds the full B·H·G·G score tensor,
    which with ~20k genes passes the 2^31 elements MPS allows per tensor.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        attn_chunk: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        if attn_chunk is not None and attn_chunk < 1:
            raise ValueError(f"attn_chunk must be a positive number of genes, got {attn_chunk}")
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_chunk = attn_chunk
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def _attention(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        dropout = self.dropout if self.training else 0.0
        chunk = self.attn_chunk or t
        out = torch.cat(
            [
                F.scaled_dot_product_attention(q[:, :, i : i + chunk], k, v, dropout_p=dropout)
                for i in range(0, t, chunk)
            ],
            dim=2,
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, d))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class BulkRNABert(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_bins: int = DEFAULT_N_BINS,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.0,
        attn_chunk: int | None = None,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_bins = n_bins
        self.mask_id = n_bins
        self.token_embedding = nn.Embedding(n_bins + 1, d_model)
        self.gene_embedding = nn.Embedding(n_genes, d_model)
        for embedding in (self.token_embedding, self.gene_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dim_ff, dropout, attn_chunk) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_bins)

    def encode(self, tokens: Tensor) -> Tensor:
        """(B, G) tokens, gene j at position j -> (B, G, d) final hidden states."""
        if tokens.shape[-1] != self.n_genes:
            raise ValueError(f"expected {self.n_genes} gene tokens per sample, got {tokens.shape}")
        x = self.token_embedding(tokens) + self.gene_embedding.weight
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, tokens: Tensor) -> Tensor:
        """(B, G) tokens -> (B, G, n_bins) logits over each gene's bin."""
        return self.head(self.encode(tokens))

    def embed(self, tokens: Tensor) -> Tensor:
        """(B, G) tokens -> (B, d) sample embedding: the mean over genes."""
        return self.encode(tokens).mean(dim=1)
