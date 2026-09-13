"""TifBERT's encoder, masked gene modelling and pooling.

  vocabulary  one token per gene of the loaded selection (its position,
              0 … G − 1), then `[PAD]` = G and `[MASK]` = G + 1. Tokens are
              gene identities only.
  encoder     BERT (`BertForMaskedLM`, the paper's backbone): token plus
              learned absolute position embeddings — position within the
              window, i.e. rank — LayerNorm, post-norm transformer blocks
              with GELU, weights drawn from N(0, 0.02).
  head        BERT's MLM head: dense, GELU, LayerNorm, then logits through
              the transposed token embedding (tied) plus a bias. Evaluated
              only at the masked positions.
  masking     BERT's 15%, 80/10/10, from `reimp_shared.tokens.mask_tokens`;
              random replacements are genes, and padding is never selected.
  pooling     mean of a window's final hidden states over its real tokens.

The paper loads `BertForMaskedLM` from Hugging Face `transformers`; this is
the same architecture in plain torch, without that dependency.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reimp_shared.tokens import IGNORE_INDEX, mask_tokens

LAYER_NORM_EPS = 1e-12  # BERT's
INIT_STD = 0.02  # BERT's initializer_range


def mask_genes(
    tokens: Tensor,
    n_genes: int,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """BERT corruption of (B, T) gene tokens padded with `n_genes`: (inputs, labels).

    The mask token is `n_genes + 1`; a random replacement is a gene in
    [0, n_genes). Padding stays padding and is never a label. With a
    `generator` the draws run on its device, then move to `tokens`'.
    """
    pad_id, mask_id = n_genes, n_genes + 1
    device = tokens.device
    if generator is not None and generator.device != device:
        tokens = tokens.to(generator.device)
    inputs, labels = mask_tokens(tokens, mask_id, n_genes, mask_prob, generator)
    pad = tokens == pad_id
    inputs = inputs.masked_fill(pad, pad_id)
    labels = labels.masked_fill(pad, IGNORE_INDEX)
    return inputs.to(device), labels.to(device)


def mlm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean cross-entropy over (N, V) logits at N masked positions; 0 when N = 0."""
    return F.cross_entropy(logits, labels, reduction="sum") / max(len(labels), 1)


def mean_pool(hidden: Tensor, tokens: Tensor, pad_id: int) -> Tensor:
    """(B, T, d) hidden states -> (B, d), averaged over each row's non-padding tokens."""
    real = (tokens != pad_id).unsqueeze(-1).to(hidden.dtype)
    return (hidden * real).sum(dim=1) / real.sum(dim=1).clamp_min(1.0)


def _init_bert(module: nn.Module) -> None:
    """BERT's initialization: N(0, 0.02) weights, zero biases, unit LayerNorms."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=INIT_STD)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        # Its packed q/k/v projection is a bare parameter, not an nn.Linear.
        nn.init.normal_(module.in_proj_weight, std=INIT_STD)
        nn.init.zeros_(module.in_proj_bias)


class TifBERT(nn.Module):
    """BERT over gene tokens; `forward` gives hidden states, `logits` the MLM head."""

    def __init__(
        self,
        n_genes: int,
        window: int = 512,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dim_ff: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_genes = n_genes
        self.pad_id, self.mask_id = n_genes, n_genes + 1
        self.token_embedding = nn.Embedding(n_genes + 2, d_model, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(window, d_model)
        self.embedding_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_ff,
            dropout,
            activation="gelu",
            layer_norm_eps=LAYER_NORM_EPS,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        )
        self.head_bias = nn.Parameter(torch.zeros(n_genes + 2))
        self.apply(_init_bert)

    def forward(self, tokens: Tensor) -> Tensor:
        """(B, T) tokens, T <= window -> (B, T, d) final hidden states."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(self.embedding_norm(x))
        return self.encoder(x, src_key_padding_mask=tokens == self.pad_id)

    def logits(self, hidden: Tensor) -> Tensor:
        """(..., d) hidden states -> (..., G + 2) token logits, decoder tied to the embedding."""
        return F.linear(self.head(hidden), self.token_embedding.weight, self.head_bias)

    def embed(self, tokens: Tensor) -> Tensor:
        """(B, T) windows -> (B, d): hidden states mean-pooled over the real tokens."""
        return mean_pool(self(tokens), tokens, self.pad_id)
