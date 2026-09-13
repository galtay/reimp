# reimp-txfm

TxFM — Kenyon-Dean et al., *Effective Biological Representation Learning by
Masking Gene Expression*, ICLR 2026 Workshop on Foundation Models for
Science ([OpenReview](https://openreview.net/forum?id=NqZqClqtTK),
[opentxfm](https://github.com/recursionpharma/opentxfm)).

A transformer encodes K randomly chosen genes of a sample — each token a
learned gene embedding concatenated with the gene's log-normalized value —
plus a CLS token; an MLP decodes the CLS embedding back to all G genes
through a rectified tanh, under a Poisson loss on every gene.

```bash
uv run txfm fit --config txfm/configs/debug.yaml     # minutes, on real data
uv run txfm fit --config txfm/configs/tcga_s.yaml    # TxFM-S on TCGA
uv run txfm-embed --ckpt <ckpt> --out out/txfm_s.parquet
```

[`paper.md`](paper.md) records what the paper did, including its
evaluations; below is how this reimplementation follows it.

## From the paper

| | |
|---|---|
| preprocessing | library-size normalize to L = 1e5, log1p (the shared `lognorm` transform) |
| masking | K = 2048 genes per sample, uniform without replacement |
| token | `[E_i ; x_i]`, E_i ∈ R^(d-1); learned CLS; no positional encoding |
| output | rectified tanh, log(L+1) · ReLU(tanh(z / 4e)) |
| loss | Poisson, e^x̂ − x̂·e^x, averaged over all G genes (masked and visible) |
| decoder | 4-layer residual MLP from the CLS embedding |
| backbones | S 384-d/6 blocks/6 heads, B 768/12/12, L 1024/24/16; stochastic depth 0.1 (0.3 for L) |
| optimizer | AdamW, max lr 1e-3, betas (0.9, 0.999), eps 1e-6, weight decay 1 / (lr · total steps) |
| schedule | one-cycle cosine, 10% warmup |

## Ours

The paper leaves these open; each is a constructor argument.

- Pre-norm blocks with LayerScale (init 1e-4) and stochastic depth rising
  linearly with depth.
- Decoder hidden width = d_model.
- Gene embeddings at `nn.Embedding`'s default N(0, 1) init.
- Weight decay on matrices and embeddings only, not on biases, norms or
  LayerScale gains.
- Gradient clipping at 1.0 in the configs.
- Protein-coding genes (19,944), where K = 2048 masks ~90% — the paper's
  average ratio on DiverseRNA-1.4M. The paper's own TCGA subset had 19,594
  genes.
- Validation reports the Poisson loss on masked and visible genes
  separately, plus per-sample Pearson and R² on the masked ones.
