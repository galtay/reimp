# reimp-bulkrnabert

BulkRNABert — Gélard, Richard, Pierrot, Cournède, *BulkRNABert: Cancer
prognosis from bulk RNA-seq based language models*, ML4H 2024, PMLR
259:384–400 ([proceedings](https://proceedings.mlr.press/v259/gelard25a.html),
[instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research)).

A BERT-style encoder reads a sample as one token per gene: the gene's log
TPM, divided by a training-set maximum and cut into 64 bins. Each token's
expression embedding is added to a learned embedding of its gene; there is
no positional encoding. 15% of tokens are corrupted, 80/10/10 mask / random
bin / kept, and the model predicts their original bins. A sample's
embedding is the mean over genes of the last layer.

```bash
uv run bulkrnabert fit --config bulkrnabert/configs/debug.yaml --data.fold 0  # ~1 min, real data
uv run bulkrnabert-embed --ckpt runs/bulkrnabert_debug/fold0/checkpoints/last.ckpt \
    --out out/bulkrnabert_debug/fold0.parquet --batch-size 4
uv run reimp-shared probe out/bulkrnabert_debug --bootstrap 0

uv run bulkrnabert fit --config bulkrnabert/configs/tcga.yaml --data.fold 2  # one fold, paper size
uv run bulkrnabert-embed --ckpt runs/bulkrnabert/fold2/checkpoints/best.ckpt \
    --out out/bulkrnabert/fold2.parquet
```

Train one model per fold (`--data.fold 0` … `4`) and write each fold's
embeddings to `out/bulkrnabert/fold<k>.parquet`; `reimp-shared probe
out/bulkrnabert` then scores all five together. Fold k runs in
`<trainer.default_root_dir>/fold<k>/` (`reimp_shared.foldcli.FoldCLI`):
config, logs and `checkpoints/` — `best.ckpt` (lowest `val/loss`) and
`last.ckpt` for `tcga.yaml`, `last.ckpt` only for `debug.yaml` — and a
rerun of the fold replaces them. `bulkrnabert-embed` reads the fold, the
data settings and the tokenizer maximum from the checkpoint, and embeds
every sample from uncorrupted tokens.

[`paper.md`](paper.md) records what the paper did, including its
evaluations; below is how this reimplementation follows it.

## From the paper

| | |
|---|---|
| input | TPM → log(1 + x) (`tpm_unstranded`, `transform: log1p`; the maximum makes natural log and log10 give the same bins) |
| tokens | `reimp_shared.tokens.BinTokenizer`: divide by one maximum, 64 equal-width bins, bin 0 for exact zeros only, no per-gene scaling |
| embedding | expression-token embedding + gene embedding, no positional encoding |
| encoder | 4 layers, 8 heads, d = 256, FFN 512 |
| objective | masked language modelling: 15% of tokens selected, zeros included; 80% mask, 10% random bin, 10% kept (`reimp_shared.tokens.mask_tokens`); cross-entropy over the selected positions |
| sample embedding | mean over genes of the final hidden states |
| optimizer | AdamW |
| budget | ~3M tokens per step, 12B tokens: `tcga.yaml` takes 128 samples (~2.6M tokens) per step for 72 epochs (~12B tokens) |

## Standardized, per `paper.md` "For reimp"

- **Training patients only.** Each fold's model is pretrained on that
  fold's training samples, with val for monitoring; the paper pretrained
  on ~95% of TCGA, test samples included.
- **The tokenizer maximum is a training statistic.** It is measured on the
  fold's training rows when fit starts (`LitBulkRNABert.setup`), stored as
  the `token_max` hyperparameter, and so saved in the checkpoint; embedding
  bins every sample by it, and held-out values above it land in the top
  bin. The paper used one dataset-wide maximum (5.547 in log10). Ours, on
  fold 0, is 12.73 in natural log (5.53 in log10).
- **Genes.** The shared default: 19,944 protein-coding genes, in place of
  the paper's 19,062-gene GTEx / ENCODE / TCGA intersection. That list maps
  onto GENCODE v36, so it can be used through `data.gene_ids_path`.
- **No Gene2Vec.** Gene embeddings are learned from scratch, so every gene
  representation comes from TCGA training patients alone.
- **The shared patient-level 5-fold split** instead of 5 seeds of 80/20, and
  the shared linear probes instead of the paper's SVM, MLP and IA3 heads.

## Ours

The paper leaves these open; each is a constructor argument.

- Pre-norm blocks with GELU and a final LayerNorm; BERT's N(0, 0.02) init
  for both embedding tables; a linear MLM head to the 64 bins (the mask
  token is an input only).
- Learning rate 1e-4, weight decay 0.01 on matrices and embeddings only,
  one-cycle cosine schedule with 10% warmup; gradient clipping at 1.0 in
  the configs.
- `attn_chunk`: attention computed in chunks of that many query genes. The
  same result, but no attention call holds the whole B·H·G·G score tensor,
  which at ~20k genes passes the 2^31 elements MPS allows per tensor.
- Validation masks from a generator reset each epoch, so `val/loss` and
  `val/accuracy` (the share of corrupted tokens whose bin is recovered)
  compare like with like across epochs.

## Configs

| field | `debug.yaml` | `tcga.yaml` | |
|---|---|---|---|
| `model.n_bins` | 64 | 64 | expression bins; the mask token is id `n_bins` |
| `model.d_model`, `n_layers`, `n_heads`, `dim_ff` | 32, 2, 1, 64 | 256, 4, 8, 512 | encoder size |
| `model.attn_chunk` | — | 1024 | query genes per attention call; `null` for one call |
| `model.mask_prob` | 0.15 | 0.15 | share of tokens corrupted |
| `model.token_max` | — | — | `null`: fit on the training rows; set only to reuse a known maximum |
| `model.lr`, `weight_decay`, `warmup_frac` | 1e-3, 0.01, 0.1 | 1e-4, 0.01, 0.1 | AdamW and the one-cycle schedule |
| `data.quantification`, `transform` | `tpm_unstranded`, `log1p` | same | the tokenizer's input |
| `data.gene_types` / `gene_ids_path` | protein coding | protein coding | the vocabulary |
| `data.batch_size` | 2 | 8 (× 16 accumulated) | |
| `data.fold` | 0 | 0 | the cross-validation fold, 0–4 |

`n_genes` is linked from the DataModule's gene selection.

Attention spans every gene, so cost grows with the square of the gene
count. On an Apple M4 Max (MPS), `debug.yaml` fits in ~30 s (60 batches of
two, 20 of validation) and embeds all 11,505 samples in ~3.5 min at
`--batch-size 4`; MPS's fast attention path wants a head dimension of at
least 32. `tcga.yaml` is
for a CUDA GPU (add `--trainer.precision bf16-mixed`); on MPS it takes ~6 s
per sample.
