# reimp-tifbert

TifBERT — Hosseini and Sharma, *TifBERT: a self-supervised foundation model
for normalization-robust bulk RNA-seq representation learning*, bioRxiv
2026 ([preprint](https://www.biorxiv.org/content/10.64898/2026.06.08.728683v1),
[mohsenh17/TifBERT](https://github.com/mohsenh17/TifBERT)).

Each sample becomes a sentence: its genes, sorted by a per-sample tf-idf
score, with no expression value — a gene's rank is its position. The
sentence is cut into overlapping windows, and a BERT learns masked gene
modelling over them. A sample's embedding is the mean over its windows of
each window's mean-pooled hidden states.

```bash
uv run tifbert fit --config tifbert/configs/debug.yaml                    # ~1 minute, on real data
uv run tifbert fit --config tifbert/configs/tcga_base.yaml --data.fold 0  # and so on for folds 1-4
uv run tifbert-embed --ckpt runs/tifbert_base/fold0/checkpoints/best.ckpt --out out/tifbert/fold0.parquet
uv run reimp-shared probe out/tifbert out/pca256 --against pca256
```

Train one model per fold (`--data.fold k`) and embed each from its own
checkpoint to `out/tifbert/fold{k}.parquet`. Fold k runs in
`<trainer.default_root_dir>/fold<k>/` — for the full config
`runs/tifbert_base/fold<k>/`, holding `config.yaml`, `metrics.csv`, the
TensorBoard events and `checkpoints/best.ckpt` (lowest val loss) and
`checkpoints/last.ckpt`. Rerunning a fold replaces its run. The debug
config keeps only `runs/tifbert_debug/fold0/checkpoints/last.ckpt`. The
checkpoint carries its fold, its data settings, its gene ranker and the
genes expressed in its training samples, so `tifbert-embed` needs nothing
else; `--batch-size` and `--embed-chunk` (windows encoded at once) trade
speed for memory.

[`paper.md`](paper.md) records what the paper did, including its
evaluations and the decisions below ("For reimp").

## From the paper

| | |
|---|---|
| tokens | gene identities only, one per gene; `[PAD]`, `[MASK]` |
| ranking | per-sample score, highest first (the paper's "TF-IDF") |
| windows | 512 tokens, stride 256, over ~10,000 genes: 39 windows per sample |
| objective | masked gene modelling: 15% of tokens, 80% `[MASK]` / 10% random gene / 10% kept |
| backbone | BERT-base (`BertForMaskedLM`): 12 layers, 768-d, 12 heads, 3,072 feed-forward, GELU, post-norm, learned absolute positions, tied MLM decoder |
| optimizer | AdamW, lr 1e-4 |
| sample embedding | mean-pooled tokens per window (the paper also stacks CLS) |

## Ours

Standardized, as `paper.md`'s "For reimp" section decides:

- **Ranking**: `reimp_shared.ranking.GeneRanker`, `tfidf` with the
  `entropy` weight on TPM (`tpm_unstranded`, as stored), fit on the fold's
  training samples in `LitTifBERT.setup` and saved in the checkpoint. The
  paper's other orderings are one setting away (`model.rank_score`:
  `expression`, `z`, `cohort_l2`; `model.idf_scheme`: `count`, `smooth`).
- **Genes**: the 19,944 protein-coding genes, tokens as positions in that
  selection (Ensembl IDs, not HUGO symbols). No variance or median filter.
- **Split**: our patient-level fold; the model pretrains on training
  patients only, validation is the fold's val patients.
- **Sample embedding**: the mean over windows of the mean-pooled windows,
  for the shared frozen-embedding probes, instead of a per-task trained
  attention pooling head over the stack of windows.

Choices the paper leaves open, each a constructor argument:

- A sentence holds only genes the sample expresses (value > 0) that some
  training sample expressed — a zero has no rank among the others, and a
  gene never expressed in training has nothing to score it by —
  top-ranked first, at most `max_genes` = 10,000. The score doesn't choose
  the genes, only their order: one it weighs 0 (under `idf_scheme: count`,
  any gene detected in every training sample, about half of them) stays
  in the sentence and ranks last. The mask of genes expressed in training
  is fit with the ranker and saved in the checkpoint. Our samples
  express a median 17,239 protein-coding genes (14,614 at least), so the
  cap drops each sample's lowest-ranked genes and keeps the paper's
  sequence length and window count; the paper got there by filtering its
  vocabulary instead. `max_genes: null` keeps them all (~67 windows).
- Windows start at 0, every `stride`, until one reaches the end of the
  sentence; the last is padded. Positions restart at 0 in every window.
- No CLS or SEP token: MLM alone gives CLS no training signal, and the
  embedding is mean-pooled.
- Training draws one window per sample per step, uniformly over that
  sample's windows, so an epoch is one window per training sample. The
  paper's epoch is every window; its 350 epochs would be ~13,650 of ours.
  The full config runs 200 (each window seen about five times).
- AdamW with BERT's other settings: weight decay 0.01 on matrices and
  embeddings (not biases or norms), eps 1e-6, linear warmup over 10% of
  steps then linear decay, dropout 0.1, N(0, 0.02) initialization.
  Gradient clipping at 1.0 in the configs.
- The architecture is written in plain torch rather than loaded from
  `transformers`.
- Validation reports the masked-gene loss and top-1 accuracy, from
  windows and masks drawn by a generator reset each epoch.

## Configs

| file | what |
|---|---|
| `configs/debug.yaml` | 2-layer, 64-d BERT over 128-token windows of the top 2,048 genes, 2 × 30 batches; ~15 s to fit and ~30 s to embed all 11,505 samples on an M4 Max (MPS) |
| `configs/tcga_base.yaml` | BERT-base as above, batch 32, 200 epochs, CSV and TensorBoard logs, `best.ckpt` (val loss) and `last.ckpt` |

Model fields (`model.*`; `n_genes` is linked from the data):

| field | default | |
|---|---|---|
| `rank_score` | `tfidf` | `GeneRanker` score: `expression`, `z`, `tfidf`, `cohort_l2` |
| `idf_scheme` | `entropy` | tf-idf rarity weight: `count`, `smooth`, `entropy` |
| `detection_threshold` | 0.0 | value counted as detected by the `count` / `smooth` weights |
| `max_genes` | 10000 | top-ranked genes kept per sample; `null` for all it expresses |
| `window`, `stride` | 512, 256 | tokens per window and between window starts |
| `mask_prob` | 0.15 | share of tokens selected for masking (80/10/10) |
| `d_model`, `n_layers`, `n_heads`, `mlp_ratio`, `dropout` | 768, 12, 12, 4.0, 0.1 | BERT-base |
| `lr`, `weight_decay`, `warmup_frac`, `betas`, `eps` | 1e-4, 0.01, 0.1, (0.9, 0.999), 1e-6 | AdamW and the schedule |
| `embed_chunk` | 128 | windows encoded at once at predict time |
| `seed` | 0 | validation windows and masks |

Data fields (`data.*`) are the shared `ExpressionDataModule`'s; the
configs use `tpm_unstranded`, protein-coding genes, `transform: none` —
the ranker is fit on values as the model reads them — and `fold`.
