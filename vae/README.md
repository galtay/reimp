# reimp-vae

Autoencoders with a regularized Gaussian latent, in two published
configurations of one small package:

- **Tybalt** (Way and Greene, PSB 2018;
  [greenelab/tybalt](https://github.com/greenelab/tybalt)): one dense layer
  to a 100-d latent, KL-regularized, over the 5,000 most variable genes.
  [`paper.md`](paper.md), with the BioBombe follow-up that compares it with
  PCA, ICA and NMF across latent sizes.
- **Tissue-supervised MMD autoencoder** (Pande, Uyar and Akalin, bioRxiv 2026;
  [BIMSBbioinfo/flexynesis_tissue_vae_manuscript](https://github.com/BIMSBbioinfo/flexynesis_tissue_vae_manuscript)):
  MMD-regularized rather than KL, a 121-d latent, and a tissue classifier
  trained jointly on the latent. [`tissue_vae.md`](tissue_vae.md).

One `LitVAE` and one `VAEDataModule` cover both; each model is a config.
Both embed a sample as its posterior mean.

```bash
uv run vae fit --config vae/configs/debug.yaml          # Tybalt, about a minute on real data
uv run vae fit --config vae/configs/debug_mmdae.yaml    # MMD-AE + organ head, likewise

# One model per fold, 0-4; each run is a new version_N under the config's run directory.
uv run vae fit --config vae/configs/tybalt.yaml --data.fold 0
uv run vae fit --config vae/configs/mmdae_none.yaml --data.fold 0
uv run vae fit --config vae/configs/mmdae_organ.yaml --data.fold 0
uv run vae fit --config vae/configs/mmdae_project.yaml --data.fold 0   # optional, not ranked

# The fold is read from the checkpoint; keep one directory per model and variant.
uv run vae-embed --ckpt runs/tybalt/version_0/checkpoints/best.ckpt --out out/tybalt/fold0.parquet
uv run vae-embed --ckpt runs/mmdae_organ/version_0/checkpoints/best.ckpt --out out/mmdae_organ/fold0.parquet
uv run reimp-shared probe out/pca256 out/tybalt out/mmdae_none out/mmdae_organ --against pca256
```

## The two models

| | Tybalt (`tybalt.yaml`) | MMD-AE (`mmdae_{none,organ,project}.yaml`) |
|---|---|---|
| input | `unstranded` counts, `lognorm` (the PCA baseline's input) | `tpm_unstranded`, log1p |
| genes | the 5,000 protein-coding genes with the largest median absolute deviation | all 19,944 protein-coding genes |
| scaling | per-gene min-max; val and test clipped to [0, 1] | per-gene z-score |
| encoder | genes → 100; mean and log-variance heads each Dense → BatchNorm → ReLU | genes → 3,989 (0.2 · genes; LeakyReLU 0.2, BatchNorm) → linear heads to 121 |
| decoder | 100 → genes, sigmoid | 121 → 3,989 (LeakyReLU, BatchNorm) → genes, linear |
| sampling | z = μ + exp(logvar / 2) · ε | the same, with logvar capped at 0 |
| reconstruction | per-gene BCE, summed over genes | MSE, averaged over genes |
| regularizer | KL, β = 0 in epoch 0 then raised by κ = 1 per epoch to 1 | MMD between the batch's z and 200 N(0, I) draws, kernel exp(−‖x − y‖² / d²), weight 1 |
| supervision | none | `none`; `organ` (26 classes) or `project` (33): a 121 → 32 → classes head (BatchNorm, ReLU, dropout 0.1) on the sampled z, cross-entropy at weight 1 |
| optimizer | Adam 5e-4, batch 50, at most 50 epochs | Adam 1.72e-3, batch 32, at most 500 epochs |
| stopping | early stopping on the fold's val loss (patience 10), best checkpoint kept | the same |
| init | Glorot-uniform weights, zero biases (Keras's default) | PyTorch's default |
| parameters | 1.5M | 160M |
| embedding | μ, 100-d and non-negative (the ReLU'd head) | μ, 121-d |

Tybalt's BatchNorm + ReLU on both heads is what its released code does,
and every published Tybalt feature came from it: posterior means are
non-negative and every posterior variance is at least 1. `heads: linear`
gives the textbook VAE.

**Supervised variants are reported as supervised.** On TCGA alone, tissue
supervision is cancer-type supervision, which the `project_id` probe then
partly reads back. Report `mmdae_organ` and `mmdae_project` as their gap to
`mmdae_none`, never as a rank among unsupervised models. Under `organ` the
within-organ probes (LUAD / LUSC, KICH / KIRC / KIRP, COAD / READ, GBM /
LGG) stay an honest test, since the label ties inside each; survival,
pathway, invertibility and confounder probes are fair for every variant.

The organ map (`reimp_vae.organs.PROJECT_ORGAN`) is fixed and committed,
the same in every fold. Normals get their project's organ, so tumour vs
normal stays unsupervised. Besides the four within-organ groups above, UCEC
and UCS share `uterus` and ACC and PCPG share `adrenal_gland`; every other
project is its own organ. LAML, sampled as peripheral blood, is `blood`.

## Fitted statistics

Every statistic comes from the fold's training samples (`shared/EVALS.md`,
rule 3):

- the gene ranking by median absolute deviation, and the per-gene minimum
  and maximum or mean and SD: `GeneScaler.fit` on `data.rows("train")` in
  `VAEDataModule.setup`. The fitted scaler is the DataModule's state, so it
  is saved in every checkpoint, and `vae-embed` restores it rather than
  refitting;
- BatchNorm running statistics and every weight, from training batches.
  Val patients only choose when to stop and which checkpoint to keep.

The supervision labels are a fixed function of the project, not fitted.

## Config fields

`n_genes` and `n_classes` are linked from the DataModule, from its gene
selection (and `top_genes`) and its `supervision`.

| `model.` | |
|---|---|
| `latent_dim` | 100 (Tybalt), 121 (MMD-AE) |
| `hidden_dim`, `hidden_factor` | hidden width: `hidden_dim` if set, else round(`hidden_factor` · n_genes); 0 means no hidden layer |
| `heads` | `bn_relu` (Tybalt's code) or `linear` |
| `reconstruction` | `bce` (sigmoid output; needs min-max input) or `mse` (linear output) |
| `regularizer` | `kl` (warmed up by `kappa` per epoch) or `mmd` (against `mmd_prior_samples` N(0, I) draws) |
| `logvar_max` | cap on the log-variance before exp; 0.0 for the MMD-AE, `null` (none) for Tybalt, whose KL restrains it |
| `class_hidden`, `class_dropout` | the classifier head, when `n_classes` > 0 |
| `glorot_init` | Glorot-uniform weights and zero biases |
| `lr`, `seed` | Adam's learning rate; the seed of the validation noise |

| `data.` | |
|---|---|
| every `ExpressionDataModule` field | `quantification`, `gene_types`, `transform`, `fold`, `batch_size`, ... |
| `top_genes` | keep this many genes by median absolute deviation over the training rows; `null` keeps all |
| `scaling` | `minmax`, `zscore` or `none`, fit on the training rows |
| `supervision` | `none`, `organ` or `project`; adds `label` to every batch |
| `project_organ` | replaces the committed project → organ map |

Logged: `train/` and `val/` `loss`, `recon`, `kl` or `mmd`, and with a
head `class` and `accuracy`; `train/beta` for Tybalt. `val/loss` is the
full objective with β = 1 whatever the warm-up, so early stopping compares
like with like; its noise (ε and the prior draws) is reseeded every
validation loop.

## Deviations from the papers, and why

Tybalt:

- **Median, not mean, absolute deviation** for the gene ranking, as the
  paper and BioBombe describe it; the released code used pandas' mean
  absolute deviation.
- **Fit on the fold's training samples**: the ranking, minima and maxima
  were fit on all samples in the paper. Val and test values are clipped to
  [0, 1], which BCE and the sigmoid need.
- **Library-normalized log counts** (`unstranded`, `lognorm`) over
  protein-coding genes instead of Xena's `HiSeqV2`, which the paper calls
  "log2(FPKM + 1) transformed RSEM values". The quantification is not what
  defines Tybalt, and this is the PCA baseline's input, so the matched-k
  comparison with PCA differs only in Tybalt's 5,000-gene selection and
  min-max scaling. Min-max scaling removes the log base.
- **The fold's val patients** replace the random 10% sample hold-out, and
  early stopping (patience 10) on their loss, with the best checkpoint kept,
  bounds the paper's fixed 50 epochs.
- The warm-up is capped at β = 1. Keras's callback could overshoot 1 for a
  κ that does not divide 1; at κ = 1 it is the same.
- BCE is computed from logits, which equals sigmoid then BCE without Keras's
  1e-7 clipping. BatchNorm keeps PyTorch's defaults (momentum 0.1, eps
  1e-5; Keras used 0.01 and 1e-3).

MMD-AE:

- **The two code quirks are fixed**, as `tissue_vae.md` decided: the
  decoder output is linear instead of a sigmoid, which cannot represent
  z-scores, and σ = exp(logvar / 2) instead of the raw log-variance head
  used as a standard deviation.
- **The log-variance is capped at 0** (`logvar_max: 0.0`), a posterior no
  wider than the N(0, I) prior. The two fixes above remove what bounded the
  original's loss, the sigmoid. With a linear output, σ = exp(logvar / 2)
  and no KL term, nothing holds the log-variance down: the MMD's kernel
  (bandwidth d² = 14,641 at d = 121) barely sees scale, and the decoder's
  training-mode BatchNorm normalizes away one sample's huge z. In the debug
  run the head passed 100 on some samples within 40 steps, exp overflowed,
  and the val loss was inf, so early stopping chose nothing. 0 is where a
  KL term holds an uninformative dimension. The embedding, μ, is untouched.
- **No 121 → 121 "fusion" layer** after each head. Flexynesis adds one for
  a single omics layer; two linear maps in a row are one linear map, so it
  changes only the optimization.
- **TCGA only**: training patients of the fold, tumours and normals; 19,944
  protein-coding genes without Flexynesis's 1% variance filter, so the
  hidden layer is 3,989 wide and the model has about 160M parameters.
  `hidden_dim` sets a narrower one.
- **Organ labels from a fixed project → organ map**, 26 classes, instead of
  the paper's 42 UBERON classes over ARCHS4, GTEx and TCGA. `tissue_vae.md`
  expected about 25, naming lung, kidney, brain and colorectal; merging
  UCEC / UCS and ACC / PCPG too gives 26.
- **Early stopping on the fold's val patients**; the paper stopped and
  chose its checkpoint on its test split.
- The denoising variant is not implemented; the paper found it
  indistinguishable from the standard model.

Neither paper's released weights are used: both saw most TCGA test
patients.

## Tests

`uv run pytest vae/tests`, offline, against the miniature dataset:

- `test_model`: BCE equals Tybalt's G · mean BCE; KL matches
  `torch.distributions`; the KL warm-up schedule; the MMD kernel is
  exp(−‖x − y‖² / d²), MMD is 0 for one sample and pulls a shifted sample
  towards the prior; σ = exp(logvar / 2); Tybalt's heads are non-negative;
  the hidden layer and mirrored decoder; Glorot init.
- `test_scaling`: median (not mean) absolute deviation, top-gene order and
  ties, min-max with clipping, z-score, statistics from the rows fit.
- `test_data`: the scaler is fit on training rows only (rescaling every val
  and test value leaves it unchanged, and it equals a fit on the training
  rows alone), labels for tumours and normals, and a checkpoint restores the
  scaler without refitting.
- `test_organs`: all 33 projects mapped; each within-organ probe group
  shares one organ; 26 classes; labels depend on the project alone.
- `test_lit`, `test_cli`: training steps, the warm-up in a fit, the
  supervised head, posterior-mean predictions, every config runs a batch,
  and a one-fold fit then `vae-embed` writes a file `read_embeddings`
  accepts.
