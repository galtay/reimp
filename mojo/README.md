# reimp-mojo

MOJO — Gélard, Benkirane, Pierrot, Richard, Cournède, *Bimodal masked
language modeling for bulk RNA-seq and DNA methylation representation
learning*, bioRxiv 2025
([10.1101/2025.06.25.661237](https://doi.org/10.1101/2025.06.25.661237),
[instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research)).

MOJO models RNA and DNA methylation together; the dataset has no
methylation, so this is its RNA half. That half is still unlike the other
models here: BulkRNABert's tokens and masked-token objective, but a U-Net
that convolves ~20k gene tokens down to 78 pooled positions before any
attention, and decodes back up to one position per gene to predict the
masked tokens. Scored beside `bulkrnabert/`, it shows what that backbone
changes with the tokens and objective held fixed.

```bash
uv run mojo fit --config mojo/configs/debug.yaml                 # ~1 minute, on real data
uv run mojo-embed --ckpt runs/mojo_debug/fold0/checkpoints/last.ckpt --out out/mojo_debug/fold0.parquet

uv run mojo fit --config mojo/configs/tcga.yaml --data.fold 0    # and so on for folds 1-4
uv run mojo-embed --ckpt runs/mojo/fold0/checkpoints/best.ckpt --out out/mojo/fold0.parquet
uv run reimp-shared probe out/mojo out/pca256 --against pca256
```

Train one model per fold (`--data.fold k`) and embed each with its own
checkpoint. Fold k runs in `runs/mojo/fold<k>/` (config, `metrics.csv`,
TensorBoard events), with the lowest-val-loss checkpoint at
`checkpoints/best.ckpt` and the final one at `checkpoints/last.ckpt`;
rerunning a fold replaces its run. The checkpoint records its fold and
its tokenizer, so `mojo-embed` needs nothing else. [`paper.md`](paper.md)
records what the paper did; below is how this reimplementation follows it.

## Model

| | |
|---|---|
| input | `tpm_unstranded`, `log1p`, the 19,944 protein-coding genes in dataset order |
| tokens | BulkRNABert's (`reimp_shared.tokens.BinTokenizer`): 64 bins, bin 0 for exact zeros, bins 1–63 splitting (0, max] evenly; max fit on the fold's training rows; plus [MASK] and [PAD] |
| token embedding | bin embedding + gene embedding (learned from scratch), summed |
| padding | [PAD] tokens, no gene embedding, up to a multiple of 2^n_down: 19,968 |
| stem | convolution, kernel 15 |
| down | `n_down` = 8 blocks, each conv (kernel 5) + residual 1×1 conv, then average pooling by 2; channels geometric from `conv_channels` to `d_model` |
| transformer | `n_layers` pre-LN layers over the 19,968 / 2^8 = 78 pooled positions: rotary position embeddings, SwiGLU feed-forward of width `mlp_ratio` × `d_model` |
| up | mirror of down: transposed conv doubling the length, + the skip of that level, conv + residual 1×1 conv; then + the stem's output |
| head | one masked-token head: LayerNorm + linear to 64 bin logits per gene |
| objective | 15% of genes selected, 80/10/10 mask / random bin / kept (`reimp_shared.tokens.mask_tokens`); cross-entropy on the selected genes |
| embedding | the mean over the 78 pooled positions of the last transformer layer, `d_model`-d, from unmasked tokens |

Full size (`configs/tcga.yaml`): 128-d embeddings, channels 128 → 256, 4
layers of 8 heads, SwiGLU width 512, 10.4M parameters (2.6M in the
transformer, 2.6M in gene embeddings). AdamW at lr 3e-4, weight decay 0.01
on matrices, kernels and embeddings, 5% linear warmup then cosine decay,
gradient clipping at 1.0, batch 32, up to 50 epochs with the best val loss
kept. Debug (`configs/debug.yaml`): the same 8 halvings at widths 32 → 64,
2 layers, 1.1M parameters, 40 batches of 16.

On an Apple M4 Max (MPS, shared with other jobs) the debug fit takes
~25 s and embeds all 11,505 samples in ~10 s. The full config runs ~5 s
per step, most of it in the convolutions at full gene length: ~20 minutes
per epoch of ~250 steps, so `max_epochs` (and the widths of the first
blocks) are what to trim if a fold must fit in a day.

## Config fields

`model:` (`LitMOJO`; `n_genes` is linked from `data`)

| field | default | |
|---|---|---|
| `n_bins` | 64 | expression bins |
| `mask_prob` | 0.15 | share of genes selected for the loss |
| `token_max` | null | tokenizer maximum; null fits it on the fold's training rows at `fit` and saves it in the checkpoint |
| `gene_order` | dataset | `genome` sorts genes by chromosome and start before the convolutions (the paper's ablation) |
| `embed_dim` | 128 | token and gene embedding width |
| `conv_channels`, `d_model` | 128, 256 | channels after the stem and at the pooled positions (the embedding size) |
| `n_down` | 8 | halving blocks; genes pad to a multiple of 2^n_down |
| `n_layers`, `n_heads`, `mlp_ratio` | 4, 8, 2.0 | transformer over the pooled positions |
| `stem_kernel`, `kernel_size` | 15, 5 | convolution kernels (odd) |
| `dropout` | 0.0 | attention and residual dropout |
| `lr`, `weight_decay`, `warmup_frac`, `betas` | 3e-4, 0.01, 0.05, (0.9, 0.999) | AdamW, linear warmup then cosine decay |
| `seed` | 0 | validation masks |

`data:` is the shared `ExpressionDataModule` (`quantification:
tpm_unstranded`, `transform: log1p`, `fold`, `batch_size`, ...). Other
quantifications run, but the tokens are BulkRNABert's only on log TPM.

## Leakage

The one fitted statistic is the tokenizer's maximum. `LitMOJO.setup` fits
it on `data.rows("train")` of the fold being trained, before the first
batch, and stores it as the `token_max` hyperparameter; `mojo-embed`
restores it from the checkpoint, so val and test samples are binned by a
maximum they did not help set (values above it fall in the top bin).
Pretraining sees only the fold's training rows; val rows only score the
validation loss. Gene embeddings and the genome order depend on no sample.
The tests scramble a fold's val and test rows and check that neither the
fitted maximum nor any training sample's embedding moves.

## Deviations from the paper

- **RNA only.** No methylation tokens, methylation head or
  mutual-information loss: the dataset has no methylation. One
  masked-token head.
- **Genes.** The 19,944 protein-coding genes (padded to 19,968, 78 pooled
  positions) instead of 17,116 genes with methylation probes (17,152,
  67 positions).
- **Gene embeddings from scratch**, not Gene2Vec, as in `bulkrnabert/`.
- **Tokenizer maximum from the fold's training samples**, not the whole
  cohort; natural-log `log1p` in place of log10, which gives the same bins
  (the tokenizer divides by a maximum in the same units).
- **Pretraining on each fold's training patients only**; the released,
  transductive and bimodal weights are not used.
- **Size.** 10.4M parameters with a 256-d embedding, scaled to ~8,000
  training samples per fold, against the paper's 52.3M at 512 channels,
  8 layers and 16 heads.
- **Block internals are ours.** The paper's config fixes the stem kernel,
  the halving count, the pre-LN rotary SwiGLU transformer and its 2×
  feed-forward; the inner kernel (5), average pooling, the residual 1×1
  convolutions, the geometric channel schedule, the stem skip, the head and
  the optimizer settings are our choices.
- **Dataset order by default.** The dataset lists genes by Ensembl ID, so
  neighbouring positions are unrelated genes; the paper found genome order
  changed nothing within error, and `gene_order: genome` runs that variant.

Validation logs the masked-token loss and accuracy on the val rows, with
masks from a generator reset each epoch so epochs compare like with like.
