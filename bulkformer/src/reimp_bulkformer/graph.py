"""The gene co-expression graph BulkFormer's GCN runs on, fit on training samples.

Construction follows the paper (preprint l.464–474): edge weight |Pearson r|
between two genes across samples, at most `k` edges per gene, none with
|r| < `threshold`. Each gene's own edge (r = 1) counts among its `k`, as in
the released graphs, whose genes all have out-degree 20 with self-edges
included; a gene constant over the samples, whose r is undefined, keeps
only its self-edge.

The top-k lists are directed (gene j among gene i's nearest need not make i
among j's); the graph here is their union made undirected, so the GCN's
symmetric normalization `D^-1/2 A D^-1/2` is well defined and the
normalized adjacency is one fixed sparse matrix.

Pass only the training samples: the graph is a fitted statistic
(`shared/EVALS.md`, rule 3). The released graph is TCGA co-expression over
every patient, test patients included (`paper.md`), and is not used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

DEFAULT_K = 20
DEFAULT_THRESHOLD = 0.4


def coexpression_graph(
    values,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
    chunk_size: int = 2048,
) -> tuple[Tensor, Tensor]:
    """(n_samples, n_genes) values -> undirected edges `(2, E)` and their |r| `(E,)`.

    Every gene has a self-edge of weight 1. Edges run both ways and are
    sorted by (source, target). Correlations are computed `chunk_size` genes
    at a time, so memory stays at `chunk_size × n_genes`.
    """
    x = torch.as_tensor(np.asarray(values, dtype=np.float32))
    if x.ndim != 2 or len(x) < 2:
        raise ValueError(f"need a (samples, genes) matrix with at least 2 samples, got {x.shape}")
    if k < 1:
        raise ValueError(f"k must be at least 1 (the self-edge), got {k}")
    n_genes = x.shape[1]
    # Moments accumulate in float64; the matrix stays float32 (one copy).
    z = x - x.mean(dim=0, dtype=torch.float64).to(torch.float32)
    norm = torch.linalg.vector_norm(z, dim=0, dtype=torch.float64)
    # Unit columns, so z_i · z_j is Pearson r; a constant gene's column is 0.
    z.mul_(torch.where(norm > 0, 1 / norm, 0.0).to(torch.float32))

    n_neighbours = min(k - 1, n_genes - 1)
    sources, targets, weights = [], [], []
    if n_neighbours > 0:
        for start in range(0, n_genes, chunk_size):
            stop = min(start + chunk_size, n_genes)
            r = (z[:, start:stop].T @ z).abs_()
            rows = torch.arange(stop - start)
            r[rows, rows + start] = -1.0  # the self-edge is added below
            top, j = r.topk(n_neighbours, dim=1)
            keep = (top >= threshold) & (top > 0)
            sources.append((rows + start).unsqueeze(1).expand_as(j)[keep])
            targets.append(j[keep])
            weights.append(top[keep])
    genes = torch.arange(n_genes)
    src = torch.cat([*sources, *targets, genes])
    dst = torch.cat([*targets, *sources, genes])
    w = torch.cat([*weights, *weights, torch.ones(n_genes)])
    # Union of both directions; a mutual pair appears twice with the same
    # |r| (up to rounding), so keep the larger.
    keys, inverse = torch.unique(src * n_genes + dst, return_inverse=True)
    weight = torch.zeros(len(keys)).scatter_reduce_(0, inverse, w, reduce="amax")
    index = torch.stack([keys // n_genes, keys % n_genes])
    return index, weight


def gcn_normalize(index: Tensor, weight: Tensor, n_nodes: int) -> Tensor:
    """Edge weights of `D^-1/2 A D^-1/2`, D the weighted degree (GCNConv's normalization).

    `index` must already hold both directions of every edge and the
    self-edges; nothing is added here (GCNConv with `add_self_loops=False`).
    """
    degree = torch.zeros(n_nodes, dtype=weight.dtype).index_add_(0, index[0], weight)
    scale = degree.clamp_min(1e-12).rsqrt()
    return scale[index[0]] * weight * scale[index[1]]


def shuffle_genes(index: Tensor, n_nodes: int, generator: torch.Generator | None = None) -> Tensor:
    """The same graph on randomly relabelled genes: the degree-matched random control.

    Every edge count, weight and degree survives; which genes they connect
    does not. Self-edges stay self-edges.
    """
    perm = torch.randperm(n_nodes, generator=generator)
    return perm[index]


def sparse_adjacency(index: Tensor, weight: Tensor, n_nodes: int) -> Tensor:
    """A coalesced sparse COO `(n_nodes, n_nodes)` matrix; row i gathers from its columns."""
    return torch.sparse_coo_tensor(
        index, weight, (n_nodes, n_nodes), check_invariants=False
    ).coalesce()
