"""BulkFormer architecture, masking and loss.

  tokens    one per gene, all G of them, the sum of
            - a fixed sinusoid of the gene's log1p TPM, `[sin(xθ), cos(xθ)]`,
              θ_i = 100^(−2i/d), zero at masked positions (paper "REE");
            - a learned gene-identity embedding through an MLP d→4d→d;
            - a whole-sample context: an MLP over the sample's full masked
              input vector (masked genes at −10), added to every token;
            then an MLP d→4d→d.
  blocks    N of: LayerNorm, x + GCN(x) over the gene co-expression graph,
            then K pre-norm Performer layers (FAVOR+ attention, FFN ×4).
  head      per gene: LayerNorm, MLP d→4d→1, ReLU — the predicted log1p TPM.
  loss      MSE on the masked genes only, ~15% of each sample.
  embedding final-layer gene tokens pooled over genes: max, or mean.

The GCN is `Â X W + b` with Â the fixed normalized adjacency of
`graph.coexpression_graph`: one sparse matmul, no torch_geometric. The graph
is a buffer set by `set_graph` once the training samples are known, and
saved with the weights.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from reimp_bulkformer.favor import FavorAttention
from reimp_bulkformer.graph import gcn_normalize, sparse_adjacency

Pooling = Literal["max", "mean"]
MASK_VALUE = -10.0  # the paper's placeholder for masked (and absent) genes


def mask_genes(
    n_rows: int,
    n_genes: int,
    mask_ratio: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(n_rows, n_genes) bool: round(mask_ratio · n_genes) genes per row, at least one.

    Uniform without replacement. With a `generator` the draw runs on the
    generator's device and is then moved to `device`, so a seeded mask is
    the same on any accelerator.
    """
    k = min(n_genes, max(1, round(mask_ratio * n_genes)))
    draw_device = generator.device if generator is not None else device
    scores = torch.rand(n_rows, n_genes, generator=generator, device=draw_device)
    mask = torch.zeros(n_rows, n_genes, dtype=torch.bool, device=draw_device)
    mask.scatter_(1, scores.topk(k, dim=1).indices, True)
    return mask.to(device)


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over the masked positions only."""
    return F.mse_loss(pred[mask], target[mask])


def mlp(d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out)
    )


class ExpressionEmbedding(nn.Module):
    """Fixed sinusoid of a scalar value, `[sin(x·θ), cos(x·θ)]`; no parameters."""

    def __init__(self, d_model: int, base: float = 100.0) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError(f"d_model must be even for the sinusoidal embedding, got {d_model}")
        theta = base ** (-2 * torch.arange(d_model // 2) / d_model)
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        angles = values.unsqueeze(-1) * self.theta
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


def graph_conv(adjacency: Tensor, x: Tensor) -> Tensor:
    """Â x for every sample: sparse (G, G) times (B, G, d), as one matmul."""
    b, g, d = x.shape
    flat = x.transpose(0, 1).reshape(g, b * d)
    return torch.sparse.mm(adjacency, flat).view(g, b, d).transpose(0, 1)


class GraphConv(nn.Module):
    """GCNConv on a fixed normalized adjacency: Â X W, bias after aggregation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        return graph_conv(adjacency, self.linear(x)) + self.bias


class PerformerLayer(nn.Module):
    """Pre-norm Performer layer: FAVOR+ attention, then a GELU FFN."""

    def __init__(
        self, d_model: int, n_heads: int, n_features: int | None, dim_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FavorAttention(d_model, n_heads, n_features, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(mlp(d_model, dim_ff, d_model, dropout), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class BulkFormerBlock(nn.Module):
    """LayerNorm, x + GCN(x), then K Performer layers. `use_graph=False` drops the GCN."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_features: int | None,
        dim_ff: int,
        dropout: float,
        use_graph: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gcn = GraphConv(d_model) if use_graph else None
        self.layers = nn.ModuleList(
            PerformerLayer(d_model, n_heads, n_features, dim_ff, dropout) for _ in range(n_layers)
        )
        self.activation_checkpointing = activation_checkpointing

    def forward(self, x: Tensor, adjacency: Tensor | None) -> Tensor:
        x = self.norm(x)
        if self.gcn is not None:
            x = x + self.gcn(x, adjacency)
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


class BulkFormer(nn.Module):
    """Masked-value regression over all G genes of a sample.

    `sample_hidden` is the sample-context MLP's hidden width, `d_model` when
    None (the paper's 2,560 = 4d is the largest single component; see
    `paper.md`). `use_graph=False` is the Performer-only ablation.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_blocks: int = 1,
        n_layers: int = 4,
        n_heads: int = 8,
        n_features: int | None = None,
        mlp_ratio: float = 4.0,
        sample_hidden: int | None = None,
        dropout: float = 0.1,
        use_graph: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.use_graph = use_graph
        dim_ff = int(d_model * mlp_ratio)
        self.expression = ExpressionEmbedding(d_model)
        self.gene_embedding = nn.Embedding(n_genes, d_model)
        nn.init.xavier_uniform_(self.gene_embedding.weight)
        self.gene_mlp = mlp(d_model, dim_ff, d_model)
        self.sample_mlp = mlp(n_genes, sample_hidden or d_model, d_model, dropout)
        self.token_mlp = mlp(d_model, dim_ff, d_model, dropout)
        self.blocks = nn.ModuleList(
            BulkFormerBlock(
                d_model,
                n_layers,
                n_heads,
                n_features,
                dim_ff,
                dropout,
                use_graph,
                activation_checkpointing,
            )
            for _ in range(n_blocks)
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), mlp(d_model, dim_ff, 1), nn.ReLU())
        # Every gene shares the head's output offset at init; at a random
        # sign it can put every prediction in ReLU's flat zero and stall
        # training. A positive bias avoids that; `LitBulkFormer` resets it to
        # the training mean when it fits its statistics.
        nn.init.constant_(self.head[1][-1].bias, 1.0)
        # The graph: undirected |r| edges with self-edges (persistent, so a
        # checkpoint carries the graph it was trained on) and the normalized
        # sparse adjacency derived from them. Empty until `set_graph`.
        self.register_buffer("graph_index", torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("graph_weight", torch.zeros(0))
        self.register_buffer("adjacency", None, persistent=False)

    @property
    def has_graph(self) -> bool:
        return self.graph_index.numel() > 0

    def set_graph(self, index: Tensor, weight: Tensor) -> None:
        """Install a graph from `graph.coexpression_graph` (or a control built like it)."""
        if index.numel() and int(index.max()) >= self.n_genes:
            raise ValueError(f"graph has a node >= n_genes={self.n_genes}")
        device = self.graph_weight.device
        index = index.to(device=device, dtype=torch.long)
        weight = weight.to(device=device, dtype=self.graph_weight.dtype)
        self.graph_index, self.graph_weight = index, weight
        norm = gcn_normalize(index.cpu(), weight.cpu(), self.n_genes)
        self.adjacency = sparse_adjacency(index.cpu(), norm, self.n_genes).to(device)

    def _embed_tokens(self, values: Tensor, mask: Tensor | None) -> Tensor:
        expression = self.expression(values)
        inputs = values
        if mask is not None:
            expression = expression.masked_fill(mask.unsqueeze(-1), 0.0)
            inputs = values.masked_fill(mask, MASK_VALUE)
        gene = self.gene_mlp(self.gene_embedding.weight)
        sample = self.sample_mlp(inputs).unsqueeze(1)
        return self.token_mlp(expression + gene + sample)

    def encode(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        """(B, G) log1p values and an optional (B, G) mask -> (B, G, d) final gene tokens."""
        if self.use_graph and not self.has_graph:
            raise RuntimeError("no gene graph: call set_graph with one fit on training samples")
        x = self._embed_tokens(values, mask)
        for block in self.blocks:
            x = block(x, self.adjacency)
        return x

    def forward(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        """(B, G) values, (B, G) mask -> (B, G) predicted log1p values at every gene."""
        return self.head(self.encode(values, mask)).squeeze(-1)

    def embed(self, values: Tensor, pooling: Pooling = "max") -> Tensor:
        """(B, G) unmasked values -> (B, d): final gene tokens pooled over genes."""
        tokens = self.encode(values)
        if pooling == "max":
            return tokens.amax(dim=1)
        if pooling == "mean":
            return tokens.mean(dim=1)
        raise ValueError(f"unknown pooling {pooling!r}; expected 'max' or 'mean'")
