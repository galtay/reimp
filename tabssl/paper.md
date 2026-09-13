# Tabular SSL on bulk expression (SCARF / VIME / BYOL) — what the paper did

Dradjat, Hamidi, Bartet, Hanczar. *Self-supervised representation learning
on gene expression data*. Bioinformatics 41(11):btaf533, 2025 (received
6 Jan 2025, accepted 17 Sep 2025, online 1 Oct 2025; CC BY 4.0).
[journal](https://academic.oup.com/bioinformatics/article/41/11/btaf533/8269452),
[doi 10.1093/bioinformatics/btaf533](https://doi.org/10.1093/bioinformatics/btaf533),
[PMC12611300](https://pmc.ncbi.nlm.nih.gov/articles/PMC12611300/),
[arXiv 2507.13912](https://arxiv.org/abs/2507.13912).
Code: the paper prints `github.com/kdradjat/ssrl-rnaseq`, which does not
exist; the repository is
[kdradjat/SSRL_RNAseq](https://github.com/kdradjat/SSRL_RNAseq) (51
commits, Apr 2024 – May 2025, no licence file). The data are linked from
its README as a Google Drive folder; not checked.

Read: the published version in full (Europe PMC full text) and its
supplement (Tables S1–S4, Figs S5–S6); arXiv v2 (17 Sep 2025, identical to
the published text) in full; v1 (18 Jul 2025) skimmed. v1 has the same
data sizes and splits; it describes SCARF's replacement as "the uniform
distribution over the values that feature takes on", which v2 sharpens to a
continuous uniform on [min, max]. Every result is a plot and there are no
tables of numbers, so the figures below are read off the published figures
(±0.01–0.02). Code read at commit `c486dba` (2025-05-16). **The released
code does not run as is**: `byol.py` has a syntax error (missing comma in
the projector MLP), the BYOL pretraining script uses undefined `target` and
`history_name`, `_4layers` uses an undefined `dropout`, and
`vime_self_4layers` deletes the encoder before returning it. There is no
evaluation or plotting code and no results files, although the abstract
promises "code and results". Where paper and code disagree, both are given
below.

## Model

One encoder shared by all three objectives, with an objective-specific
pretext head:

- **Encoder**: MLP, 4 × [Linear 256 → BatchNorm → ReLU → Dropout], picked
  by supervised validation accuracy on the fine-tuning set from a grid
  (Table S4: 2–10 layers, width 128–1024, dropout 0–0.5 → 0.2). ~14.6M
  parameters on their 56,902 inputs, almost all in the first layer. The
  **representation is the 256-d output of the fourth block** (after
  ReLU and dropout; `hidden_layer=-2` in BYOL, `layers[16]` in VIME).
- **SCARF** (§2.2.1; `scarf/model.py`). For each sample, exactly
  ⌊0.3·d⌋ features, chosen uniformly without replacement, are replaced by
  draws from **Uniform(min_j, max_j)**, the feature's range over the
  pretraining set. This is *not* SCARF's default (the feature's empirical
  marginal, corruption rate 0.6). Fig. S6 compares the two: uniform
  replacement beats the from-scratch baseline at small label fractions,
  while empirical-marginal replacement does no better than the baseline.
  The positive pair is the **clean** profile and one corrupted copy.
  Projection head: 2-layer MLP 256 → 256 (Linear-BN-ReLU-Linear).
  NT-Xent with cosine similarity, τ = 1.0. The code symmetrizes over all
  2N views (2N − 2 negatives); the paper's Eq. 1 is the one-directional
  form. Adam, lr 1e-4, batch 256, 1,000 epochs; 10% of the pretraining
  set held out to track the loss.
- **VIME** (§2.2.2; `vime/vime_self.py`, Keras/TensorFlow — the other two
  are PyTorch). Mask m_ij ~ Bernoulli(0.3). A masked entry is replaced by
  the same gene's value in a random other sample (a column-wise
  permutation, i.e. a draw from the gene's empirical marginal). The mask
  target is 1 where the value actually changed. Mask decoder and feature
  decoder are each 4 × Dense 256 (ReLU), then Dense d with a sigmoid. Loss
  = BCE(mask) + α·MSE(features), α = 2.0. The code reconstructs every
  feature, not just the corrupted ones as the text says. RMSprop lr 1e-3,
  batch 32, 500 epochs. **The corruption is drawn once, before training**,
  and reused every epoch, as in the original VIME code. Code wrinkle: the
  pretraining loader ignores `preprocess_type` and always z-scores, so the
  sigmoid feature decoder can never output the negative targets.
- **BYOL** (§2.2.3; `byol/byol.py`). The online branch is encoder +
  projector + predictor; the target branch is an EMA copy of encoder +
  projector. Projector and predictor are each Linear 4096 → BN → ReLU →
  Linear 256 (the code adds Dropout 0.2). View 1 is the clean profile;
  view 2 is VIME's corruption, with the permutation done **within the
  mini-batch**, so the "marginal" is the batch's. Loss 2 − 2·cos. The code
  symmetrizes, the paper describes one direction. EMA decay λ = 0.9
  (fixed; BYOL uses 0.996 → 1). Table S3 says SGD lr 1e-4, momentum 0.9,
  batch 32, 50 epochs; the code uses Adam lr 1e-4 and defaults to 200
  epochs.
- Corruption rate 0.3 for all three (Table S3). Fig. S5 (a supervised MLP
  trained on corrupted inputs loses no accuracy) is their argument that
  corruption preserves the label.

## Pretraining data

- **TCGA**: 9,349 samples × 56,902 genes (`mRNA.omics.parquet`, not in the
  repo). The quantification and GDC release are not stated. Gene IDs run to
  ENSG00000288675, which points to a recent (v36-era) GENCODE; the label
  file carries multi-omics barcodes (mRNA, miRNA, CNV, DNAm, protein), so
  the matrix probably comes from an in-house multi-omics compilation
  (unverified). There are 19 classes: 18 cancer types (BRCA, UCEC, KIRC,
  LGG, HNSC, LUAD, THCA, LUSC, PRAD, SKCM, COAD, OV, STAD, BLCA, LIHC,
  CESC, KIRP, SARC) plus **one pooled "Normal" class** of 669 normal-tissue
  samples. Table S1 has two slips. "Lung (LOAD)" is COAD (460 in the
  released `classes.parquet`). And it counts each type's normals into that
  type, so the stated majority class (12.9%, BRCA) should be 11.8%
  (1,101 / 9,349) under the labels actually used.
- **ARCHS4** (v2.2 h5): 53,282 samples × 67,128 genes; 19 tissues in
  Table S2, but the downstream task uses 10 (Discussion; notebook "10
  classes version"). ARCHS4 → TCGA uses the 55,747 shared genes.
- **Preprocessing** (Supp. A.1; `archs4_preprocessing_pipeline.ipynb`):
  drop all-zero genes, fill missing values with the gene mean,
  log2(x + 1), quantile normalization, then pyComBat (`mean_only`, per
  tissue). "We applied the same normalization process to the TCGA
  dataset" — whether that includes ComBat is unclear. **All of it is done
  before the split.** The code then z-scores each gene (`StandardScaler`),
  fitting separately on each file it loads. No gene filter: a
  protein-coding option exists but is off.
- **Split** (`split_data_tcga.ipynb`): by sample, stratified by class,
  random state 0. TCGA: 7,291 pretraining / 1,029 fine-tuning / 1,029
  test. ARCHS4: 50,998 / 1,142 / 1,142. The paper says 15% of each set is
  validation. The code uses 10% of the pretraining set for SCARF's loss
  tracking, and 1% (SCARF, VIME) or 10% (BYOL) of the fine-tuning file.

## Evaluations as published

Shared protocol (§3.2.1): cancer-type (TCGA) or tissue (ARCHS4)
classification; **accuracy only**; one fixed test split. The pretrained
encoder gets a single linear softmax head and is either fine-tuned end to
end ("unfrozen") or kept frozen, with only the head trained ("frozen" —
the analogue of a linear probe). Head training: Adam lr 1e-4, batch 8,
≤ 100 epochs, early stopping with patience 30. The paper says on
validation loss; the code uses validation accuracy and evaluates test
accuracy every epoch, and which epoch is reported is not stated. The
fine-tuning set (~1,000 samples) is subsampled to fractions
p = 0.02–1.0 in steps of 0.01. Subsets are nested and deterministic (the
first ⌈p·n_c⌉ of each class), so the 5 repeats per p (3 in the SCARF
script) vary only the initialization. Shaded bands are over those repeats.

**The only baseline is the same 4 × 256 MLP trained from scratch** on the
fine-tuning subset. There is no PCA, no logistic regression or SVM on
expression, and no random-initialization frozen encoder.

1. **Label efficiency** (Fig. 3; approximate).
   - Unfrozen: all three reach the baseline's plateau — TCGA ~0.93–0.94,
     ARCHS4 ~0.96–0.97, ARCHS4 → TCGA ~0.94–0.95 at p = 1 — and lead at
     small p. The headline is that the baseline's maximum is reached
     with 49% (TCGA, BYOL), 63% (ARCHS4, VIME) and 34% (ARCHS4 → TCGA,
     BYOL) less labelled data.
   - **Frozen, TCGA** at p = 1: baseline ~0.92, VIME ~0.90, SCARF ~0.84,
     BYOL ~0.79.
   - Frozen, ARCHS4: SCARF ≈ VIME ≈ baseline ~0.96, BYOL ~0.90.
   - Frozen, ARCHS4 → TCGA: baseline ~0.92, SCARF ~0.82, BYOL ~0.68,
     VIME ~0.60.
   - The authors: "In all cases, frozen fine-tuning leads to poor
     performance". The text also contradicts itself on ARCHS4 → TCGA
     (BYOL "strongly outperforms", yet transfer "does not yield better
     results than training from scratch").
2. **Architecture sweep** (Fig. 4): depth 2–10, width 256 or 1,024,
   unfrozen. The score is the difference in area under the accuracy curve
   for p ∈ [0.02, 0.3] versus the matching from-scratch MLP; the dataset
   is not stated. Range −0.03 to +0.07.
   - SCARF gains most (up to 0.07, width 1,024, depth 7–10).
   - BYOL gains 0.01–0.04.
   - VIME is negative at width 1,024 for depths 2, 3 and 9.
   - Covariance spectra of the embeddings (after Jing et al. 2021): VIME
     collapses (~100 of 1,024 non-zero singular values at depth 9), SCARF
     partly (≤ 20% zero at 256, ≤ 50% at 1,024), BYOL not at all.
3. **Pretraining-set size** (Fig. 5; ARCHS4, p = 0.1, unfrozen): baseline
   0.887. With the full pretraining set, SCARF ~0.93, VIME ~0.925, BYOL
   ~0.905. Pretraining on fewer than ~2,000–2,500 samples (VIME: fewer
   than ~10,000) hurts.
4. **Recommendations** (§5): fine-tune unfrozen; use SCARF when
   pretraining and fine-tuning data are homogeneous, BYOL when they are
   not; be cautious with VIME.

The abstract's "first work that deals with bulk RNA-Seq data and
self-supervised learning" is contestable. Its own reference, Gross et al.
2024, benchmarks representation learners on bulk RNA-seq.

## For reimp

**One directory, three objectives, one encoder.** This mirrors the
paper's own design, and it is the cleanest objective-versus-objective
comparison reimp could hold: same MLP, same inputs, same embedding layer,
with only the pretext task changing. A config field `objective: scarf |
vime | byol` selects the head and the loss; one `fit` per fold per
objective, and each objective writes its own embeddings directory
(`out/tabssl_scarf/…`). The directory name reflects that.

**Method-defining**, kept:
- The encoder: 4 × [Linear 256, BN, ReLU, Dropout 0.2]; embedding = the
  256-d output of block 4, with no corruption at embedding time.
- SCARF: a fixed 30% of features per sample, replaced from
  Uniform[min, max] of the feature; clean anchor versus one corrupted
  view; 2-layer 256 projection head; NT-Xent, τ = 1.0, symmetric.
  Empirical-marginal replacement (the original SCARF) as an option,
  since the paper's own Fig. S6 separates the two.
- VIME: Bernoulli(0.3) mask; replacement by the gene's value in another
  training sample; 4 × 256 mask and feature decoders; BCE + 2.0 · MSE.
- BYOL: clean versus VIME-corrupted view; 4096-hidden projector and
  predictor; EMA target, λ = 0.9; 2 − 2·cos, symmetric.
- Corruption rate 0.3, and each method's batch size and epochs
  (SCARF 256 / 1,000; VIME 32 / 500; BYOL 32 / 50).

**Incidental**, standardized:
- TCGA only, the shared patient-level split, pretraining on training
  patients only; val patients for loss tracking. No ARCHS4.
- Genes: our 19,944 protein-coding default instead of all 56,902.
- Input: `lognorm` (or `log1p` of TPM) from `reimp_shared.data`, then a
  per-gene z-score **fit on training samples**. No quantile normalization
  or ComBat: both are fit across the cohort, and they are not part of the
  methods. (2026-09-13: the z-scores are also clipped to each gene's
  training range, at training and at embedding. On fold 0, genes nearly
  constant over training gave held-out samples |z| up to 1,398, against
  91 in training. See the README.)
- VIME's corruption redrawn every batch rather than once per run; the
  fixed draw is an artefact of the reference code. Note this in the
  README as a deviation.
- BYOL's in-batch permutation replaced by the same training-set
  permutation VIME uses (their loader does not shuffle, so a batch's
  marginal depends on file order).
- One framework (PyTorch / Lightning) and one optimizer per objective as
  in Table S3, except BYOL: Adam 1e-4 as in the code (SGD at lr 1e-4 is
  implausibly slow; record the choice).
- Unfrozen fine-tuning is out of scope: reimp scores frozen embeddings
  with shared linear probes. That is the paper's "frozen" setting —
  **the one where all three methods did worst**.

**Leakage in the paper**, and what removes it:
- The split is by sample: 706 of 8,635 patients have more than one sample,
  and 652 of the 669 normals have a tumour from the same patient, free to
  land on the other side of pretraining / fine-tuning / test. → Patient
  folds.
- Log2, quantile normalization and (ARCHS4) ComBat were done on the whole
  dataset before splitting. → Per-sample transforms only; everything else
  fit on training samples.
- `StandardScaler` is fit on each loaded file: the fine-tuning file
  includes the test samples, and the encoder is pretrained under a
  different scaler than it is fine-tuned under (a shift as well as a
  leak). → One scaler per fold, fit on training samples and saved with
  the model.
- Corruption statistics: SCARF's min/max came from the pretraining set,
  and VIME's and BYOL's replacements from pretraining samples, so no test
  leak here. → In reimp, compute min/max and draw replacements from the
  fold's training samples only. The min/max range follows single outlier
  samples; keep it as published, but it is one reason to offer the
  empirical-marginal option.
- Test accuracy was computed every epoch, and the reported epoch is not
  stated. → Early stopping on val patients only.

**Compute** (estimate, not measured): ~8,300 training samples per fold;
the encoder is ~5.3M parameters on 19,944 genes, and the whole training
matrix (0.66 GB float32) fits on one GPU.
- SCARF: ~33k steps of batch 256 × 2 views; minutes on one GPU once the
  reference code's per-row `randperm` loop is vectorized.
- VIME: ~130k steps of batch 32 with two 256 → 19,944 decoders
  (~10M more parameters); tens of minutes on a GPU, hours on CPU.
- BYOL: ~13k steps; minutes.
- Three objectives × 5 folds ≈ 15 runs, roughly 1–2 GPU-hours in all —
  the cheapest model in reimp.

**Evaluation ideas to adapt:**
- Cancer-type classification → already the shared `classification_probe`
  (33 projects, tumours only; their pooled "Normal" class corresponds to
  our `tumor_vs_normal`).
- **Label-efficiency probe** (their central claim, Fig. 3): a linear probe
  fit on nested fractions of training patients (e.g. 1, 2, 5, 10, 25, 50,
  100%), scored on the fold's test patients. It works on any embeddings
  file, so it is fair to every model. A candidate for `shared/EVALS.md`.
- **Untrained-encoder control**: embeddings from the same MLP at random
  initialization. A linear probe on random ReLU features of 20k genes can
  be strong; without this control, "SSL helps" is unfalsifiable. Cheap to
  add as `objective: none`.
- Dimensional collapse (Fig. 4) → already covered by `geometry_probe`
  (effective rank, top-eigenvalue share).
- Pretraining-set size (Fig. 5) → possible by subsampling training
  patients; low priority.
- Expectation: in the paper the frozen embeddings trail a supervised MLP
  trained from scratch, and PCA was never tried. They may not beat
  PCA + logistic regression on our probes, which is itself worth
  knowing, and cheap to find out.
