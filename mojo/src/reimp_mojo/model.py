"""MOJO's RNA U-Net and its masked-token loss.

  tokens       one per gene: an embedded expression bin (BulkRNABert's
               bins, plus [MASK] and [PAD]) summed with a learned gene
               embedding. The gene list is padded with [PAD] to a multiple
               of 2^n_down, so every halving divides evenly.
  down         a convolutional stem, then `n_down` blocks that each convolve
               and halve the length by average pooling, keeping their
               pre-pooling activations as skips. Channels rise
               geometrically from `conv_channels` to `d_model`.
  transformer  pre-LN layers with rotary position embeddings and a SwiGLU
               feed-forward, over the L / 2^n_down pooled positions:
               attention over a few dozen positions rather than over genes.
  up           transposed convolutions that double the length back to one
               position per gene, each adding the skip of its level.
  head         one masked-token head: logits over the bins at every gene.

The sample embedding is the mean over the pooled positions of the last
transformer layer. The paper fixes the stem kernel (15), the halving blocks,
the pre-LN rotary SwiGLU layers and the 2 × d_model feed-forward
(`config.json`). The inner kernel size (5), average pooling and residual 1×1
convolutions follow the released code; the channel schedule, the skip layout
and the head are ours.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reimp_shared.tokens import IGNORE_INDEX

ROPE_BASE = 10_000.0


def padded_length(n_genes: int, n_down: int) -> int:
    """`n_genes` rounded up to a multiple of 2^n_down."""
    step = 2**n_down
    return -(-n_genes // step) * step


def channel_schedule(conv_channels: int, d_model: int, n_down: int) -> list[int]:
    """Channels at each of the `n_down + 1` lengths, geometric from `conv_channels` to `d_model`."""
    ratio = d_model / conv_channels
    return [round(conv_channels * ratio ** (i / n_down)) for i in range(n_down + 1)]


def masked_token_loss(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Cross-entropy and accuracy over the selected positions.

    `logits` is (..., n_bins), `labels` the matching token or `IGNORE_INDEX`
    (as `reimp_shared.tokens.mask_tokens` returns them); unselected positions
    do not count.
    """
    selected = labels != IGNORE_INDEX
    logits, labels = logits[selected], labels[selected]
    return F.cross_entropy(logits, labels), (logits.argmax(dim=-1) == labels).float().mean()


def rotary(x: Tensor) -> Tensor:
    """Rotary position embedding of (B, H, T, D) queries or keys at positions 0 … T − 1.

    Channel pairs (i, i + D/2) turn by t · ROPE_BASE^(−2i/D) (the rotate-half
    form), so a query-key product depends on their positions only through
    their offset.
    """
    t, d = x.shape[-2:]
    freqs = ROPE_BASE ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angles = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freqs
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) · up(x))."""

    def __init__(self, d_model: int, dim_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, dim_ff)
        self.up = nn.Linear(d_model, dim_ff)
        self.down = nn.Linear(dim_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-LN transformer layer: rotary self-attention, then a SwiGLU feed-forward."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        if (d_model // n_heads) % 2:
            raise ValueError("rotary embeddings need an even head dimension")
        self.n_heads = n_heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_ff)
        self.drop = nn.Dropout(dropout)

    def _attention(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(rotary(q), rotary(k), v, dropout_p=dropout)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self._attention(self.norm1(x)))
        return x + self.drop(self.ffn(self.norm2(x)))


class ConvBlock(nn.Module):
    """LayerNorm over channels, a same-length convolution, GELU: (B, L, C_in) -> (B, L, C_out)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_in)
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2))


class DownBlock(nn.Module):
    """Convolve to `c_out` channels, then halve the length: (B, L, C_in) -> (B, L/2, C_out).

    Also returns the pre-pooling (B, L, C_out) activations, the skip its
    `UpBlock` adds back.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = ConvBlock(c_in, c_out, kernel_size)
        self.residual = ConvBlock(c_out, c_out, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        skip = x + self.residual(x)
        return F.avg_pool1d(skip.transpose(1, 2), 2).transpose(1, 2), skip


class UpBlock(nn.Module):
    """Double the length, add the skip, convolve to `c_out`: (B, L, C_in) -> (B, 2L, C_out)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_in)
        # Stride 2 with padding k // 2 and output_padding 1 gives exactly 2L for odd k.
        self.deconv = nn.ConvTranspose1d(
            c_in, c_in, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1
        )
        self.conv = ConvBlock(c_in, c_out, kernel_size)
        self.residual = ConvBlock(c_out, c_out, 1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.gelu(self.deconv(self.norm(x).transpose(1, 2)).transpose(1, 2)) + skip
        x = self.conv(x)
        return x + self.residual(x)


class MOJO(nn.Module):
    """(B, G) bin tokens -> (B, G, n_bins) logits, through the U-Net.

    Token ids 0 … n_bins − 1 are expression bins, `mask_id` = n_bins is
    [MASK], and `pad_id` = n_bins + 1 fills the padding, which the model
    adds and removes itself: callers pass and receive exactly `n_genes`
    positions.
    """

    def __init__(
        self,
        n_genes: int,
        n_bins: int = 64,
        embed_dim: int = 128,
        conv_channels: int = 128,
        d_model: int = 256,
        n_down: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_ff: int = 512,
        stem_kernel: int = 15,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_down < 1 or n_layers < 1:
            raise ValueError("MOJO needs at least one halving block and one transformer layer")
        if stem_kernel % 2 == 0 or kernel_size % 2 == 0:
            raise ValueError("kernel sizes must be odd, so convolutions keep lengths exact")
        self.n_genes = n_genes
        self.n_bins = n_bins
        self.mask_id = n_bins
        self.pad_id = n_bins + 1
        self.length = padded_length(n_genes, n_down)
        self.n_pooled = self.length // 2**n_down

        # nn.Embedding's default N(0, 1) init; gene embeddings learned from scratch.
        self.token_embedding = nn.Embedding(n_bins + 2, embed_dim)
        self.gene_embedding = nn.Embedding(n_genes, embed_dim)
        channels = channel_schedule(conv_channels, d_model, n_down)
        self.stem = nn.Conv1d(embed_dim, channels[0], stem_kernel, padding=stem_kernel // 2)
        self.down = nn.ModuleList(
            DownBlock(channels[i], channels[i + 1], kernel_size) for i in range(n_down)
        )
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        )
        self.up = nn.ModuleList(
            UpBlock(channels[i + 1], channels[i], kernel_size) for i in reversed(range(n_down))
        )
        self.head = nn.Sequential(nn.LayerNorm(channels[0]), nn.Linear(channels[0], n_bins))

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        """(B, G) tokens -> (B, L, embed_dim), padded with [PAD] tokens that carry no gene."""
        if tokens.shape[-1] != self.n_genes:
            raise ValueError(f"expected {self.n_genes} gene tokens, got {tokens.shape[-1]}")
        pad = self.length - self.n_genes
        genes = F.pad(self.gene_embedding.weight, (0, 0, 0, pad))
        return self.token_embedding(F.pad(tokens, (0, pad), value=self.pad_id)) + genes

    def encode(self, tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        """(B, G) tokens -> the last transformer layer's (B, n_pooled, d_model) and the skips.

        Skips run finest first: the stem's (B, L, conv_channels), then each
        halving block's pre-pooling activations.
        """
        x = self.stem(self._embed_tokens(tokens).transpose(1, 2)).transpose(1, 2)
        skips = [x]
        for block in self.down:
            x, skip = block(x)
            skips.append(skip)
        for block in self.blocks:
            x = block(x)
        return x, skips

    def embed(self, tokens: Tensor) -> Tensor:
        """(B, G) tokens -> (B, d_model): the mean over pooled positions of the last layer."""
        return self.encode(tokens)[0].mean(dim=1)

    def forward(self, tokens: Tensor) -> Tensor:
        """(B, G) tokens -> (B, G, n_bins) logits."""
        x, skips = self.encode(tokens)
        for block, skip in zip(self.up, reversed(skips[1:]), strict=True):
            x = block(x, skip)
        return self.head(x + skips[0])[:, : self.n_genes]
