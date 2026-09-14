"""TxFM architecture, output activation and loss.

  encoder   transformer over K unmasked genes, each token `[E_i ; x_i]` — a
            learned (d-1)-dim gene embedding concatenated with the gene's
            log-normalized value — plus a learned CLS token. No positional
            encoding: a sample is a set of genes, not a sequence.
  decoder   residual MLP from the CLS embedding to all G genes.
  output    rectified tanh, bounded in [0, log(L+1)] (paper Eq. 1).
  loss      Poisson NLL in log-rate space, over all G genes (Eq. 2).

Encoder blocks are pre-norm with LayerScale and stochastic depth, the
"standard techniques" of paper §A.1. The paper does not give their settings
or the decoder's width; those defaults are ours.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def rectified_tanh(z: Tensor, library_size: float) -> Tensor:
    """Paper Eq. 1: log(L+1) · ReLU(tanh(z / 4e))."""
    return math.log1p(library_size) * F.relu(torch.tanh(z / (4 * math.e)))


def poisson_loss(x_hat: Tensor, target: Tensor) -> Tensor:
    """Paper Eq. 2 per element, unreduced: e^x̂ − x̂·e^x.

    The Poisson NLL of observation e^x under rate e^x̂, constant dropped;
    `x_hat` and `target` are both in log1p space. Minimized at x̂ = x.
    """
    return x_hat.exp() - x_hat * target.exp()


def sample_unmasked(
    n_rows: int,
    n_genes: int,
    k: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """`k` distinct gene positions per row, uniform without replacement.

    With a `generator` the draw runs on the generator's device and is then
    moved to `device`, so a seeded mask is the same on any accelerator.
    """
    draw_device = generator.device if generator is not None else device
    scores = torch.rand(n_rows, n_genes, generator=generator, device=draw_device)
    return scores.topk(k, dim=1).indices.to(device)


class DropPath(nn.Module):
    """Stochastic depth: drop a residual branch for whole samples."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.dim() - 1))).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale and stochastic depth."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        drop_path: float,
        layer_scale_init: float | None,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.dropout = dropout
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

        def scale() -> nn.Module:
            return (
                nn.Identity() if layer_scale_init is None else LayerScale(d_model, layer_scale_init)
            )

        self.ls1, self.ls2 = scale(), scale()
        self.drop_path = DropPath(drop_path)

    def _attention(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=dropout)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.ls1(self._attention(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class TxFMEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        drop_path_rate: float,
        layer_scale_init: float | None,
    ) -> None:
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be >= 2 to leave room for the value scalar")
        # nn.Embedding's default N(0, 1) init; the paper does not give one.
        self.gene_embedding = nn.Embedding(n_genes, d_model - 1)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Stochastic depth grows linearly with depth, the usual ViT schedule.
        rates = torch.linspace(0, drop_path_rate, n_layers).tolist()
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dim_ff, dropout, rate, layer_scale_init) for rate in rates
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_idx: Tensor, values: Tensor) -> Tensor:
        """(B, K) gene positions and their values -> (B, d) CLS embedding."""
        tokens = torch.cat([self.gene_embedding(gene_idx), values.unsqueeze(-1)], dim=-1)
        x = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, 0])


class MLPDecoder(nn.Module):
    """`decoder_layers` linear layers: residual d->d blocks, then d->G."""

    def __init__(self, d_model: int, n_genes: int, decoder_layers: int, dropout: float) -> None:
        super().__init__()
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be >= 1")
        self.hidden = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
            for _ in range(decoder_layers - 1)
        )
        self.out = nn.Linear(d_model, n_genes)

    def forward(self, e: Tensor) -> Tensor:
        for block in self.hidden:
            e = e + block(e)
        return self.out(e)


class TxFM(nn.Module):
    def __init__(
        self,
        n_genes: int,
        library_size: float = 1e5,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        dim_ff: int = 1536,
        decoder_layers: int = 4,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        layer_scale_init: float | None = 1e-4,
    ) -> None:
        super().__init__()
        self.library_size = library_size
        self.encoder = TxFMEncoder(
            n_genes, d_model, n_layers, n_heads, dim_ff, dropout, drop_path_rate, layer_scale_init
        )
        self.decoder = MLPDecoder(d_model, n_genes, decoder_layers, dropout)

    def forward(self, gene_idx: Tensor, values: Tensor) -> Tensor:
        """(B, K) unmasked genes -> (B, G) reconstruction in log1p space."""
        return rectified_tanh(self.decoder(self.encoder(gene_idx, values)), self.library_size)
