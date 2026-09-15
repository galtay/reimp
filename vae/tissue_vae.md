# Tissue-supervised bulk VAE (Pande et al.) — what the paper did

Pande, Uyar, Akalin (BIMSB, Max Delbrück Center, Berlin). *An atlas-scale
generative model for unified representation learning of bulk RNA-seq data*.
bioRxiv, v1 posted 24 June 2026 (CC-BY-NC-ND 4.0).
[doi 10.64898/2026.06.18.733198](https://doi.org/10.64898/2026.06.18.733198),
[MDC edoc 26636](https://edoc.mdc-berlin.de/26636/).

Code: [BIMSBbioinfo/flexynesis_tissue_vae_manuscript](https://github.com/BIMSBbioinfo/flexynesis_tissue_vae_manuscript)
(MIT). It is built on [Flexynesis](https://github.com/BIMSBbioinfo/flexynesis)
`supervised_vae`, and **Flexynesis is PolyForm Noncommercial 1.0.0**.

Weights, compendium and embeddings:
[Zenodo 10.5281/zenodo.20595537](https://doi.org/10.5281/zenodo.20595537)
(CC-BY-4.0). The record is dated 12 June 2026 and titled "Tissue-supervised
latent representations from a curated 118K-sample multi-source bulk RNA-seq
compendium". It holds an 8 GB `.pth`, a 4.6 GB HDF5 tarball, 121-dimensional
train and test embeddings, and the joblib artifacts. A 53 MB int8
TorchScript model is in the GitHub repo. The demo is
[akalinLab/flexynesis-tissue-vae](https://huggingface.co/spaces/akalinLab/flexynesis-tissue-vae);
the URL resolves, but the Space was not run.

Read: the preprint PDF in full (22 pages, from MDC edoc), the supplementary
docx (Tables S1–S3), and the repo's `train_denoising_vae{,_h5}.py`,
`h5_dataloader.py`, `csv_to_h5.py`, `get_embeddings.py`,
`build_v3_webapp_artifacts.py` and tutorials. Also read: Flexynesis
`models/supervised_vae.py`, `modules.py` (Encoder, Decoder, MLP) and
`data.py` (DataImporter) at HEAD, Sep 2026. Git history shows the Encoder
and Decoder, including the sigmoid output, unchanged since before June 2026.

**Not in the repo**: the ARCHS4/GTEx/TCGA download, the UBERON mapping and
the split. `csv_to_h5.py` starts from already-processed CSVs.

Internal inconsistencies:

- The main-text per-sample ρ at 38K is 0.604; Table 2 gives 0.642.
- The candidate pool is 236,580 in the text and 238,031 in Table 1 (with
  DepMap).
- The repo README says the Zenodo DOI is "[to be added]" in one place and
  gives it in another.
- The HDF5 attributes say "log2(count+1)" where the paper says log2 TPM or
  RPKM.

## Model

- Flexynesis `supervised_vae` with one omics layer:
  - Encoder: 16,115 → 3,226 (`hidden_dim_factor` 0.2002), then LeakyReLU(0.2)
    and BatchNorm.
  - Mean and log-variance heads (3,226 → 121). For a single layer, each is
    followed by a further 121 → 121 linear "fusion" layer.
  - Decoder: symmetric (121 → 3,226, LeakyReLU, BatchNorm → 16,115), with a
    **sigmoid output**.
  - Tissue head: MLP 121 → 32 → 42, with ReLU, BatchNorm and dropout 0.1.
  - About 105M parameters (Fig. 6 caption). 2 × 16,115 × 3,226 ≈ 104M
    agrees.
- Objective: an unweighted sum (`use_loss_weighting=False`) of three terms.
  1. MSE reconstruction, averaged over all entries.
  2. MMD between the batch's sampled `z` and 200 N(0, I) draws, with a
     Gaussian kernel `exp(−mean‖·‖² / d)` (InfoVAE, Zhao et al.). There is
     **no KL term and no β or warm-up**: this is an MMD-regularized
     autoencoder, called a VAE.
  3. Cross-entropy of the tissue head applied to the **sampled z**.
- **This is how tissue supervision enters the loss**: a classifier on the
  latent code trained jointly, with weight 1 relative to the MSE term.
- Quirks in the code as run. Neither is discussed in the paper.
  - Reparameterization is `z = mean + var · ε`, where `var` is the raw
    log-variance head output used directly as a standard deviation. It can
    be negative.
  - The sigmoid decoder outputs values in (0, 1), while the targets are
    per-gene z-scores. The reconstruction therefore cannot represent
    below-mean expression. The paper's reconstruction metrics are Spearman
    correlations, which only see ranks.
- Denoising variant: 20% of genes are zeroed per sample during training, and
  the loss is taken against the unmasked input. It is indistinguishable from
  the standard model (per-gene ρ 0.934 vs 0.935). The Standard model is the
  one used for classification and TARGET.
- Hyperparameters were chosen by a 3-trial Bayesian search on an earlier
  run: latent 105/67/121 at batch 32/128/32, with 121 best. Then fixed: Adam,
  lr 1.72e-3, batch 32, early stopping with patience 10, maximum 500 epochs.
  The 118K Standard model stopped at epoch 37 (val loss 0.8012).
- Sample embedding: the **posterior mean** (`FC_mean`, 121 dimensions), as
  computed in `build_v3_webapp_artifacts.py`, `regen_fig1_v3.py` and
  `get_embeddings.py`. Flexynesis's own `transform()` returns the *sampled*
  z instead, a trap for anyone reusing it.
- Compute: one RTX 4060 (8 GB). About 5 h for 118K and about 3 h for 75K
  (Table S1). Peak RAM for data loading is 25 GB with HDF5.

## Training data

- Sources:
  - ARCHS4 v2.5 (888,821 human samples). 605,614 are bulk by the
    single-cell score (< 0.5), and 411,318 of those are a random subset
    (seed 42). Free-text tissue annotations are mapped to UBERON by keyword
    matching and manual curation, which labels 212,412.
  - GTEx v8: 14,768 normal samples.
  - TCGA via GDC: **9,400 tumour samples, no normals**. GDC release and
    quantification not stated.
- Filters: ENCODE/CCLE-style identifiers remove cell lines. This removes all
  DepMap samples; cell lines described only in free text remain. Each
  UBERON class is capped at 10,000, which affects only blood (about 55,000).
- Final: 146,537 samples = 123,127 ARCHS4 + 14,072 GTEx + 9,338 TCGA.
  **118,263 train / 28,274 test**, split 85/15 stratified by tissue at the
  *sample* level. No grouping by patient or GEO series.
- **The claimed "118,263 samples" is the training split, not the
  compendium.**
- 42 classes (`label_mapping.json`). They include non-anatomical labels:
  `other`, `stem_cell`, `fibroblast`, `soft_tissue`. TCGA tumours sit in
  organ classes, so LUAD and LUSC are both `lung`, KIRC, KIRP and KICH are
  all `kidney`, and GBM and LGG are both `brain`.
- Genes: HGNC symbols shared by all sources, 16,292. 163 "near-zero-variance"
  genes are removed, leaving 16,115. That count equals Flexynesis's default
  of dropping the lowest 1% of genes by variance
  (`variance_threshold=0.01`); this is an inference, not stated.
- Values: log2 expression ("TPM or RPKM depending on source"). The released
  scaler has a median per-gene mean of 5.7 and a maximum of 14.8, consistent
  with a log2 scale. `StandardScaler` is fit on the training split only
  (`n_samples_seen_` = 118,263).
- **Model selection used the test set.** In both training scripts
  `val_loader` is the test split, and early stopping and the best checkpoint
  both monitor its loss. All reported test numbers come from the model chosen
  on those same samples. How the 3-trial search was validated is not stated.

## Evaluations as published

All on the 28,274-sample test split unless noted. Table 2 compares three
compendium sizes (38K, 75K, 118K); Table S1 compares 75K with 118K
metric by metric.

1. **Tissue classification** (42 classes, supervised head): balanced
   accuracy 94.9%, weighted F1 96.2% (Fig. 2, Fig. S1). The 38K and 75K runs
   gave 89.5% and 90.7% balanced accuracy.
   - Per class: 100% for pituitary, pleura, salivary gland, spinal cord and
     vagina. The lowest are biliary tract (75%, n = 4), thymus (81%), eye
     (83%) and spleen (85%). Lymphoid is 91.9% and bone marrow 95.5%.
2. **Baselines**, kNN with k = 5 and cosine distance, on the same split
   (Fig. 5, Table S1).

   | baseline | balanced accuracy |
   |---|---|
   | top-2,000 HVG + kNN | 93.4% |
   | all genes + kNN | 92.6% |
   | all genes + PCA(121) + kNN | 92.0% |
   | PCA/UMAP variants (range) | 84.2–92.0% |
   | scGPT zero-shot (CLS) + kNN | 61.0% |

   The supervised VAE is +1.5 points over the best kNN baseline, and a
   supervised head against unsupervised kNN is not a like-for-like
   comparison.
3. **Latent geometry**: a t-SNE (Kobak–Berens protocol) coloured by organ
   system and by source (Fig. 1).
   - kNN source mixing (k = 20): 0.012, down from 0.047 at 75K.
   - LISI: 1.02 out of 3.0.
   - The authors state that they did not test whether the remaining source
     structure is tumour vs normal or technical.
4. **Reconstruction** (Fig. 3): median per-gene Spearman ρ 0.935, median
   per-sample ρ 0.829.
   - Imputation of randomly zeroed genes: ρ 0.881, 0.879 and 0.874 at 10%,
     20% and 30% masking (Standard).
5. **TARGET transfer** (734 paediatric tumours, from Xena; Fig. 4, Table
   S2): 84.6% (621/734) of samples fall in a developmentally "expected" adult
   tissue, with kNN (k = 5, Euclidean) against the reference embeddings.
   - The inputs were per-gene z-scored, then rescaled to the training means
     and SDs.
   - By cancer: AML 99%, ALL 96%, neuroblastoma 72%, Wilms 65%, clear-cell
     sarcoma of the kidney 0%.
   - BulkFormer-93M under the same protocol reaches 76.8%, and Wilms tumour
     accounts for the gap (Fig. 6, Table S3).
   - Cell-line ablation: 87.1% with cell lines, 84.6% without (Fig. S3).
6. There is no survival, subtype, mutation or cancer-type (TCGA project)
   evaluation. The authors list these as untested.

## For reimp

The paper's contribution is data curation and scale (ARCHS4 + GTEx + TCGA,
UBERON labels) plus a supervised head. On TCGA alone, only the second part
survives, and the published results cannot be compared with anything we
compute. **Neither the released weights nor the embeddings are usable**:
they were trained on non-TCGA data, and on about 85% of TCGA tumours
chosen by sample, which includes our test patients.

**Method-defining**, kept:

- An MMD-regularized autoencoder (no KL term) with Gaussian reparameterized
  sampling, and one hidden layer of about 0.2 × n_genes, with LeakyReLU and
  BatchNorm.
- MSE reconstruction of per-gene z-scored log expression.
- A jointly trained 2-layer classification head on the sampled latent, in an
  unweighted sum of the three losses.
- Embedding: the posterior mean.
- Input: `tpm_unstranded` + `log1p`, per-gene `StandardScaler` fit on
  training samples. log1p is natural log; z-scoring removes the base.
  *Note added 2026-09-13:* val and test z-scores are clipped to each
  gene's training range. Genes expressed in a few training samples have a
  near-zero SD, and on fold 0 put held-out values at |z| up to 238, against
  91 in training; see `README.md`.

**Incidental**, standardized:

- The latent size: 256, reimp's common embedding size (the PCA baseline's
  and the transformers'), instead of the paper's 121, which a 3-trial
  search picked among 67, 105 and 121. (Changed 2026-09-15 from 121, before
  the first 5-fold run, as Tybalt's was: reimp compares methods at a common
  size, not paper by paper.)
- TCGA training patients only, about 8,300 samples per fold, tumours and
  normals.
- Our default 19,944 protein-coding genes, not their 16,115 HGNC set. That
  gives a hidden layer of 3,989 and about 160M parameters. On 8,300 samples
  that is badly over-parameterized; expect early stopping within a few
  epochs, and expose the hidden width as an option.
- The low-variance filter is dropped. It is a 1% cut, and the protein-coding
  set is already a filter.
- Early stopping on the fold's val patients, never on test.
- The two code quirks are fixed and documented: a linear decoder output
  instead of the sigmoid, which cannot represent z-scores; and
  `σ = exp(logvar / 2)` instead of the raw head output. A reimplementation
  written from scratch also sidesteps Flexynesis's noncommercial licence.
  *Note added in implementation:* with both fixes and no KL term, nothing
  bounds the log-variance, and on TCGA it passed 100 within a few dozen
  steps, overflowing exp. `reimp-vae` caps it at 0 (`logvar_max`), a
  posterior no wider than the prior; see `README.md`.

**Supervision on TCGA.** The paper's tissue label is an organ. TCGA alone
offers only labels derived from the project, so any supervision here is
cancer-type supervision, which our cancer-type probe then partly reads back.
The labels come from training patients only, so this is not leakage, but the
probe then measures the supervised head, not an unsupervised
representation. Handle it with explicit variants, tagged as supervised in
every report:

- `supervision: none`: the same network with the head removed. The
  reference point, an unsupervised MMD-AE.
- `supervision: organ` (closest to the paper). The label is a fixed
  project → organ mapping. It is cohort-independent and therefore fold-safe,
  about 25 classes: lung, kidney, brain, colorectal, and one organ per other
  project.
  - The 33-way `project_id` probe is then partly supervised.
  - **The within-organ probes (LUAD/LUSC, KICH/KIRC/KIRP, COAD/READ,
    GBM/LGG) become the honest test**: the supervised label ties inside each
    of them, so they measure what the latent keeps beyond it.
  - Normals get their organ's label, as GTEx normals did in the paper, so
    `tumor_vs_normal` stays unsupervised.
- `supervision: project`: optional, and not ranked. Every classification
  probe except `tumor_vs_normal` is then just a readout of training
  supervision.

For each supervised variant, report the gap to `none` rather than a rank
among unsupervised models. Survival, pathway, invertibility and confounder
probes are fair comparisons for every variant, and they are where
supervision may cost something: a class-driven latent can discard
within-class variation.

**Leakage in the original:**

- Early stopping and checkpoint selection on the test split.
- A sample-level split: GEO series and TCGA patients straddle train and
  test, which likely inflates the 94.9%.
- The released model has seen most TCGA tumours.

**Compute**: rough, not measured. About 260 steps per epoch at batch 32. An
epoch costs a few TFLOP, seconds on a single modern GPU. With early
stopping, minutes per fold, and CPU is feasible. Memory: a few GB,
about 160M parameters plus Adam state. Their 5 h on an RTX 4060 was for 14×
the data.

**Implementation note**: this is in the same family as Tybalt. One `vae/`
package could cover both, with these options:

| option | values |
|---|---|
| regularizer | `kl` / `mmd` |
| gene selection | top-k by MAD / all protein-coding |
| scaling | min-max / z-score |
| decoder output | sigmoid / linear |
| supervision | `none` / `organ` / `project` |

Tybalt and this method would then be two configs.

**Evaluation ideas to adapt:**

- kNN source mixing and LISI → already covered by the shared
  `confounder_probe` (plate and tissue-source-site neighbour enrichment).
  The paper's source-mixing score is also the cautionary example: it counts
  neighbours from other sources inside tissue clusters, where source is
  confounded with tumour vs normal.
- kNN (k = 5) tissue accuracy → covered by `geometry_probe` precision@k.
- Imputation of randomly masked genes, from the embedding function → the
  same idea as BulkRNABert's missing-gene robustness, which is not adopted
  yet because it needs each model's encoder, not an embeddings file.
- TARGET developmental transfer and the cross-source scaling study → not
  possible on our dataset.
