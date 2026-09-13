# reimp-compass

COMPASS, from Shen et al., *Generalizable AI predicts immunotherapy outcomes
across cancers and treatments*, Nature Medicine 2026
([doi](https://doi.org/10.1038/s41591-026-04502-7),
[mims-harvard/COMPASS](https://github.com/mims-harvard/COMPASS)).

This package reimplements the pretraining stage, which is a structured
bottleneck. Each gene's log TPM becomes a token that passes one transformer
layer. The contextual gene embeddings are then read out through 132 fixed
literature gene sets into 43 immune and tumour-microenvironment concepts.
Training is self-supervised, with a cosine triplet loss on the concept
vector. The concept scores (43-d) are the embedding. The set scores (132-d)
are written as a second embedding.

```bash
uv run compass fit --config compass/configs/debug.yaml      # ~10 s on real data
uv run compass fit --config compass/configs/tcga.yaml --data.fold 0
uv run compass-embed --ckpt runs/compass/fold0/checkpoints/best.ckpt \
    --out out/compass/fold0.parquet --sets-out out/compass_sets/fold0.parquet
uv run reimp-shared probe out/compass out/compass_sets out/pca256 --against pca256
```

Train once per fold (`--data.fold 0` … `4`). Fold k's run lands in
`runs/compass/fold{k}/`, with its config, `metrics.csv` and TensorBoard
events. Its checkpoints go in `checkpoints/`: `best.ckpt` (lowest
`val/loss`) and `last.ckpt`. Rerunning a fold replaces its run. Embed each
fold's `best.ckpt` into `fold{k}.parquet` in both directories.
`--model.negatives same_project` trains the same-project-negatives variant.
Give that variant its own run root and output directories, e.g.
`--trainer.default_root_dir runs/compass_sameproj` and
`out/compass_sameproj`. `debug.yaml` runs in `runs/compass_debug/fold{k}/`.

[`paper.md`](paper.md) records what the paper did and which of its choices
are kept. The rest of this file covers what is implemented and how it
departs from the paper.

## What is implemented

| module | |
|---|---|
| `hierarchy` | COMPASS's gene → set → concept table (`data/conception_processed.tsv`, provenance and MIT licence in [`data/README.md`](src/reimp_compass/data/README.md)); `load_hierarchy` reads it or any table with its columns |
| `model` | per-gene min-max scaler (buffers); FT-Transformer gene tokens; post-norm SDPA encoder layer; `ConceptProjector`; `triplet_loss` |
| `triplets` | mask / jitter augmentation; `NegativeSampler` (any other patient, or one in the same project) |
| `lit` | `LitCompass`: fits the scaler and binds genes at `setup("fit")`, and builds triplets, loss, validation metrics and predictions |
| `cli`, `embed` | `compass` (LightningCLI over the shared `ExpressionDataModule`), `compass-embed` |

- **Input.** `tpm_unstranded` with the shared `log1p` transform, over the
  19,944 protein-coding genes. COMPASS takes log2(TPM + 1). The per-gene
  min-max scaling that follows divides out the log base, so the two inputs
  are identical.
- **Scaling.** Per-gene (x − min) / (max − min), like scikit-learn's
  `MinMaxScaler`, which COMPASS uses. It is fit on the fold's training
  samples only (`LitCompass.prepare`) and stored in the checkpoint, so
  embedding reuses it. A gene that is constant on the training samples
  gets scale 1. Values outside the training range are not clipped, as in
  COMPASS. They stay close to it: on fold 0, 1.9 × 10⁻⁴ of held-out values
  fall outside [0, 1], all within −0.23 … 5.6 over protein-coding genes
  (smallest non-zero training span 0.022) and −0.13 … 1.83 over the 916
  concept genes.
- **Gene tokens.** token_g = ReLU(x_g · W_g + P_g), with a per-gene W_g and
  P_g in R^32 (U(±1/√32) initialization, as in COMPASS). There is no CLS
  token and no cancer-type token.
- **Encoder.** One post-norm layer shaped like COMPASS's released Performer
  layer: d = 32, 2 heads of width 32 (inner width 64), GELU FFN of 64, and
  dropout 0.2. Attention is exact (`F.scaled_dot_product_attention`).
- **Bottleneck.** The hierarchy has 1,283 (set, gene) memberships.
  - Genes → sets: each set takes a softmax over one learned logit per
    member gene, independent of the input, and uses it to weight its
    members' 32-d contextual embeddings. One `Linear(32 → 1)`, shared by
    all 132 sets, scores the result.
  - Sets → concepts: a softmax over one learned logit per set.
  - Parameter count: 1,283 + 132 + 33 = 1,448. The paper's 1,514 adds two
    `Linear(32 → 1)` heads, for the cancer token and the patient token,
    which are dropped here. The test suite checks this arithmetic.
- **Gene mapping.** Concept genes are matched to our genes by `gene_name`.
  **All 916 are in the protein-coding default; none are missing.** A
  missing gene would drop out of its set's softmax, with a warning, and a
  set left with no genes would score its bias alone.
  `configs/concept_genes.txt` lists the 916 as Ensembl IDs
  (`uv run python -m reimp_compass.hierarchy` regenerates it).
- **Triplets.** The anchor and the positive are two augmented views of a
  training sample. The negative is an augmented view of a sample from
  another training patient. Each view is left unchanged with probability
  0.1. Otherwise it is either masked (each gene set to 0, i.e. its training
  minimum, with p = 0.1) or jittered (N(0, 0.1²) noise on every gene),
  with equal odds. All three views pass the encoder in one forward pass.
- **Negatives.** The default (`negatives: any`) follows the released code
  with K = 1: any other training patient's sample. The paper's
  `same_project` negatives are the variant. If a project has a single
  patient in the pool, its negatives fall back to the whole pool.
- **Loss.** max(0, (1 − cos(a, p)) − (1 − cos(a, n)) + 1) on the 43-d
  concept vector, `Reference` included.
- **Early stopping.** `val/loss` is the same loss on the fold's val
  patients, with negatives from val patients. Augmentations and negatives
  are redrawn from a generator reset each epoch, so epochs are comparable.
  Also logged: `val/d_pos`, `val/d_neg`, and `val/active` (the share of
  triplets with non-zero loss).
- **Embeddings.** Unaugmented, in eval mode (no dropout): 43 concept scores
  (`Reference` last) and 132 set scores, in `hierarchy.concepts` and
  `hierarchy.sets` order.

## Config fields

`model` (`LitCompass`); `n_genes` is linked from `data`.

| field | default | |
|---|---|---|
| `hierarchy_path` | null | a gene-set table with COMPASS's columns; null reads COMPASS's |
| `d_model`, `n_heads`, `head_dim`, `dim_ff`, `n_layers` | 32, 2, 32, 64, 1 | the released layer |
| `dropout` | 0.2 | residual and FFN dropout |
| `attention_chunk_size` | null | attend this many query genes at a time, recomputed in backward (see Compute) |
| `margin` | 1.0 | triplet margin on cosine distance |
| `mask_prob`, `jitter_std`, `no_augment_prob` | 0.1, 0.1, 0.1 | augmentations |
| `negatives` | `any` | `any` (K = 1) or `same_project` (the paper's, a variant) |
| `lr`, `weight_decay` | 1e-3, 1e-4 | Adam with L2 decay on every parameter, as COMPASS |
| `seed` | 0 | validation draws |

`data` is the shared `ExpressionDataModule`. Both configs use
`quantification: tpm_unstranded` and `transform: log1p`; the full config
adds `batch_size: 128`. `debug.yaml` sets `gene_ids_path` to the concept
genes. `trainer` in `tcga.yaml` has early stopping with patience 10 on
`val/loss`, at most 500 epochs, CSV and TensorBoard loggers, and keeps
`best.ckpt` and `last.ckpt`. `debug.yaml` has no logger and keeps only
`best.ckpt`. Both name a run root in `trainer.default_root_dir`, and
`compass` (a `FoldCLI`) runs fold k in `<root>/fold{k}/`.

## Deviations from the paper, and why

- **No cancer-type token, so no CANCER dimension.** The token is a label
  fed in as input, and it conditions every gene token through attention.
  With it, the cancer-type probes would score the given label rather than
  the expression. The embedding is 43-d, not 44-d (paper.md, For reimp).
- **No patient/CLS token.** The released model has one, but no output reads
  it (`proj_pid=False`).
- **Exact attention in place of Performer.** Performer is there for cost
  (paper.md). With a fused kernel (CUDA flash attention), exact attention
  is affordable. On MPS and CPU, SDPA materializes the genes² attention
  matrix: 21 GB for two samples at 19,944 genes. `attention_chunk_size`
  bounds that by attending query blocks under activation checkpointing.
  It gives the same attention, tested against the unchunked layer.
- **Data.** Each fold's training patients only, not all 10,184 TCGA
  tumours. The scaler is fit on training samples. Genes are the 19,944
  protein-coding default, not their 15,672-gene intersection with the
  immunotherapy cohorts (still available through `gene_ids_path`). TPM is
  GDC's, not re-derived. All samples of a training patient are used
  (tumours and normals), not one FFPE-free tumour per patient.
- **Negatives are always another patient.** COMPASS's data has one sample
  per patient, so "another sample" was "another patient". Here a patient
  can have several samples, and a same-patient sample is never used as a
  negative.
- **Negatives are augmented like the other views,** as the paper describes.
  The released code's `mix` defaults leave negatives unaugmented
  (`mask_n_prob = jitter_n_std = 0`).
- **Augmentation strengths.** Masking uses the paper's p = 0.1. The paper
  gives no jitter σ; 0.1 is the code's standalone `FeatureJitterAugmentor`
  default. The code's `mix` defaults are mask 0.01 and jitter 0.01, and I
  did not read the released checkpoint's values.
- **Early stopping.** Uses our val patients, not a random 1%. Patience is
  10, as in the paper (the released run used 20, capped at 500 epochs).
  One seed per fold, not the lowest validation loss of three seeds.
- **Kept as COMPASS has it:** the two sets that list a gene twice
  (`SLC4A10` in `Tcell_IL7Rmax_sc`, `XCR1` in `cDC1_sc`), and the post-norm
  layer.

The paper's fine-tuning heads, NFT prototypes and every
immunotherapy-cohort evaluation are out of scope (paper.md).

## Compute

The model has ~1.3M parameters at 19,944 genes, 98% of them the per-gene
W and P. Attention cost grows with genes², over three views per sample.
Measured on an Apple M4 Max (MPS, shared with other jobs), full genes,
`attention_chunk_size` 256:

- training: 8–13 s per step at batch 8 (9 GB), and 30–40 s at batch 32
  (36 GB), i.e. about 1 s per training sample, or hours per epoch;
- embedding: ~1 s per 32 samples, i.e. minutes for the 11,505.

The full config is meant for a GPU with flash attention. paper.md estimates
under a minute per epoch on an A100, and 1–3 GPU-hours for five folds.
`debug.yaml` uses only the 916 concept genes as tokens: two short epochs
train in ~10 s, and embedding every sample takes ~7 s.

## Not here

These ideas come from paper.md's evaluation list and are not implemented
in this package:

- the training-free gene-set baseline: mean z-scored log TPM per set,
  averaged into concepts;
- a 43-component PCA;
- concept fidelity: each concept's correlation with its own genes;
- external immune-state labels.
