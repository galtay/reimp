# COMPASS — what the paper did

Shen, Moon, Nguyen, Li, Huang, Nair, Marbach, Zitnik. *Generalizable AI
predicts immunotherapy outcomes across cancers and treatments*. Nature
Medicine 32(8):3010 (2026), published 3 July 2026 (PMC date).
[doi 10.1038/s41591-026-04502-7](https://doi.org/10.1038/s41591-026-04502-7),
[PMC13472881](https://pmc.ncbi.nlm.nih.gov/articles/PMC13472881/) (CC BY-NC-ND 4.0).
Preprint: [medRxiv 10.1101/2025.05.01.25326820](https://doi.org/10.1101/2025.05.01.25326820),
v1 2025-05-05, v2 2025-11-04, v3 2026-03-24, v4 2026-06-25 (CC BY-NC). PubMed
40385399 indexes the preprint, and 42399673 the journal article.
Code: [mims-harvard/COMPASS](https://github.com/mims-harvard/COMPASS) (MIT;
PyPI `immuno-compass`, v2.5.2), pretraining code included. Weights:
[immuno-compass.com/download](https://www.immuno-compass.com/download/) (the
TCGA-pretrained model, PFT/LFT fine-tuned models, leave-one-cohort-out and
drug-specific models; the download page states no licence). The repo also
ships `example/model/pretrainer.pt` (26 MB).

Read: the Nature Medicine main text, Methods and Extended Data captions in
full (PMC XML). I did not read the Supplementary Information (Figs 1–37,
Tables 1–12, Supplementary Methods 1–5) or the source data, so per-cohort
numbers (Supplementary Table 3) and the augmentation/negative-sampling
sensitivity analyses (Supplementary Methods 5) are unread. medRxiv v1 and v4
were compared by keyword only. Their Methods agree with the journal on every
fact below. v1's abstract says "precision by 8.5%" (its body and later
versions say accuracy), and v1's headline examples differ (83.7% on held-out
STAD, 76.1% for anti-CTLA-4; the journal gives 76.5% on LUAD and 70.8%). The
author order changed in v3. Code read at commit `0e5c876` (2026-05-19). I
loaded the released `pretrainer.pt` and `finetuner_pft_all.pt` to read their
stored hyperparameters. Where these disagree with the paper, both are given
below.

## Model

- **Input**: an integer cancer type (0–32) and TPM for 15,672 protein-coding
  genes. The code takes log2(TPM + 1), then applies a per-gene min-max
  scaling fitted on the pretraining set (`Datascaler("minmax")`). The paper's
  Methods say only "TPM" and never mention the scaling. There is no binning.
- **Gene tokens** (FT-Transformer numerical tokenizer): token_g =
  ReLU(x_g · W_g + P_g), with a per-gene W_g and P_g in R^32. The paper calls
  P the "learnable positional encoding". A cancer-type token from a 33 × 32
  table is prepended. The code also prepends a patient/CLS token, but no
  output reads it (`proj_pid=False`).
- **Encoder**: one Performer layer (linear attention with a ReLU kernel),
  d = 32, 2 heads (inner dimension 64), FFN 64, GELU, dropout 0.2. The
  released model has 1,019,421 parameters. 1,003,008 of them are the per-gene
  W and P, the transformer layer is ~13k and the projector 1,514. Performer is
  there for cost: "supports flash attention as an optional alternative".
- **Concept projector** (the bottleneck). There are 132 literature gene sets
  (916 unique genes; 1–51 genes per set, median 7), in the repo's
  `compass/tokenizer/conception_processed.tsv` and Supplementary Data 1.
  - **Set score**: a softmax-weighted sum of the member genes' 32-d contextual
    embeddings, then one Linear(32 → 1) shared by every set. The weights are
    one learned weight per gene per set and do not depend on the input.
  - **Concept score**: 43 "high-level concepts", each a softmax-weighted sum
    of its sets' scores.
  - **Cancer concept**: the 44th dimension is a Linear(32 → 1) of the cancer
    token's output.
  - **What the 43 are**: 42 biological concepts (B, T/NK, myeloid,
    mesenchymal/tissue lineages; apoptosis, IFNγ, TGFβ, cytokine,
    proliferation, TLS and genome-integrity pathways) plus `Reference`, a
    housekeeping set (ACTB, GAPDH, B2M, ...). The paper counts Reference among
    the 43.
- **Gene-set sources**: 54 of the 132 sets come from Besca (Mädler et al.
  2021, Roche's single-cell toolkit, "data-derived v8/v9"). 15 come from
  Mariathasan et al. 2018 (the IMvigor210 TGFβ paper: TGFβ/EMT, cell-cycle
  and DNA-repair sets). 10 come from Combes et al. 2022 and 8 from Danaher et
  al. 2017. Cytokine families come from Carrasco Pro et al. 2018, and there
  are TIDE sets (Jiang 2018), Sade-Feldman, Li 2019 and others. None are
  MSigDB collections (only one TLS set mentions "hallmark" genes). Two of the
  authors are at Roche, one of whom curated the sets.
- **No concept supervision**: there are no concept labels and no loss on
  concept values. A concept is defined only by which genes can feed it. A
  score's sign and scale are free, and the paper reports that several scores
  (NK, B cell, plasma cell, ILC, CD4) *anti*-correlate with their own genes
  (Supplementary Figs 21–23). This is a structured bottleneck, not a
  concept-supervised CBM in the Koh et al. sense.
- **Sample representation**: the 44 concept scores, which are the only input
  to every downstream head. `model.extract` returns gene-level (15,672),
  set-level (133 = 132 + cancer) and concept-level (44) scalar scores.
  `model.project` returns 32-d vectors per set and concept.
- **Heads**:
  - MLP: BatchNorm(44) → Linear 16 → BN → Linear 2, with a learned
    temperature.
  - NFT: cosine similarity to responder / non-responder prototypes, τ = 0.1.
  - Fine-tuning modes: FFT trains everything (~1.02M parameters), PFT the
    projector and head (2,144), LFT the head only (182), NFT nothing.

## Pretraining objective

- **Loss**: triplet loss on the 44-d concept vector, CANCER and Reference
  included: max(0, (1 − cos(a, p)) − (1 − cos(a, n)) + 1).
- **Triplets**: the anchor and the positive are two augmented views of one
  tumour, and the negative is another tumour, also augmented. Each view gets
  either random masking (zero each gene) or Gaussian jitter, chosen at
  random. The paper gives p_mask = 0.1. The code's `mix` defaults are mask
  0.01 and jitter σ 0.01, and I did not read the released checkpoint's
  augmentor values.
- **Self-supervised**: `task_loss_weight = 0` in the released checkpoint. The
  only label used is the cancer-type *input* token.
- **Paper vs code, negatives**: the paper draws the negative from "a
  different patient within the same cancer type" and upweights small cancer
  types by "balanced sampling with replacement". The released code does
  neither. `TCGAData` draws the negative from the nearest fraction K of
  samples by Euclidean distance on scaled expression, and the checkpoint has
  K = 1, i.e. any other sample. `PreTrainer` uses a plain shuffled
  DataLoader. Supplementary Methods 5 (unread) reportedly finds smaller K
  better.
- **Optimization**: Adam, lr 1e-3, batch 128; weight decay 1e-4 (checkpoint).
  1% random validation split; early stopping with patience 10 in the paper,
  20 with at most 500 epochs in the checkpoint. Three seeds (24, 42, 64),
  keeping the lowest validation loss. A100 80GB. Wall-clock time is not
  reported. The released run is dated 2024-07-16.

## Pretraining data

- **Source**: TCGA via GDC release 37 (TCGAbiolinks), STAR 2.7.5c, GENCODE
  v36. TPM was recomputed from counts with gene effective length, by the same
  pipeline they apply to the immunotherapy cohorts.
- **Sample selection**: 11,274 samples → excluding normals, 10,534 →
  excluding previously treated and FFPE samples, 10,305 → one per patient,
  **10,184** tumours from 33 cancer types. The paper literally says
  "non-FFPE", which would remove most of TCGA; the 229 samples removed fit
  FFPE. How one sample per patient was picked is not stated.
- **Genes**: the 15,672 protein-coding genes shared with the immunotherapy
  cohorts. All carry GENCODE v36 IDs in the repo's `conceptor_gene_map.csv`.
  I did not check their overlap with our 19,944.

**Pretraining is transductive**: it used every TCGA tumour patient, and the
per-gene min-max scaling was fit on all of them.

## Evaluations as published

All scored evaluations are immunotherapy response. 16 cohorts, 1,133
pretreatment patients (346 responders by RECIST CR/PR; 787 SD/PD), seven
cancer types, six drugs. 22 baselines, each a logistic regression on
single-gene or signature scores (GridSearchCV, C ∈ [0.1, 1]). Accuracy uses
a fixed 0.5 threshold; AUPRC, AUROC and MCC are also reported; COMPASS runs
use 3 seeds. Fine-tuning hyperparameters were "tuned per mode and dataset"
by internal cross-validation.

1. **Leave-one-cohort-out** (Fig. 2c,d; per-cohort numbers in
   Supplementary Table 3, unread). PFT/LFT against the second-best method,
   averaged: accuracy +8.5%, AUPRC +15.7%, MCC +12.3%. FFT falls behind when
   a large cohort is held out.
2. **Within-cohort leave-one-patient-out** (Supplementary Figs 3–6). COMPASS
   has the best average, and NFT wins on cohorts under 30. NetBio is the best
   baseline on medium and large cohorts.
3. **Cohort-to-cohort transfer**, 240 pairs, counting a success when
   accuracy beats the target's prevalence-based reference: LFT 163, PFT 155,
   PGM 130, Teff 118, NetBio 117.
4. **Held-out indication / therapy / target** (Fig. 3): 76.5% accuracy on
   held-out LUAD; 70.8% on anti-CTLA-4 when trained on anti-PD-(L)1; 85.3% on
   ipilimumab + pembrolizumab when trained on monotherapies. Ablating the
   cancer token during fine-tuning costs "moderately" (Supplementary
   Fig. 10).
5. **Multi-stage fine-tuning** (Fig. 4): atezolizumab on KIRC, 73.7% vs 70.3%
   (drug cohort only) vs 60.7% (pan-ICI only). Pembrolizumab on LUAD
   (n = 33), 91% vs 67%.
6. **IMvigor210 overall survival** (Fig. 5), PFT fine-tuned without this
   cohort:
   - n = 298: predicted responders vs non-responders give HR 4.7, log-rank
     P = 1.7 × 10⁻⁷; 1-year OS 86% vs 40%.
   - n = 234: HR 4.37 (2.29–8.32), against TMB 1.67, PD-L1 IC2+ 1.75 and
     immune phenotype 1.85. Brier score 0.212 vs 0.241 for TMB.
   - Ridge-Cox on the 132 or 44 concept features stratifies worse than the
     response probability (Supplementary Fig. 20).
7. **Representation benchmark** (Extended Data Fig. 3): logistic regression
   on geometric-mean scores, ssGSEA scores and COMPASS features (132-d and
   43-d), leave-one-cohort-out over the 16 cohorts. COMPASS is higher on
   AUROC and AUPRC; the numbers are in the source data, which I did not read.
8. **TCGA itself: nothing scored.** There is a UMAP of TCGA and ICI concept
   embeddings (Extended Data Fig. 1b) and concept–immunobiology associations
   such as MSI (Supplementary Fig. 14, unread). No cancer-type,
   reconstruction or TCGA survival numbers.

The paper also benchmarks ENLIGHT, EaSIeR and IRnet (Supplementary Fig. 28)
and feeds COMPASS concepts into a Clinical Transformer. Its stated caveats:
explanations are not validated by ablation, there is no non-ICI arm (so
prognostic and predictive signal are mixed), and there is no covariate
adjustment.

## For reimp

The whole pretraining stage is TCGA-only and reimplementable, and its frozen
concept vector is a per-sample embedding our probes can score. Fine-tuning,
NFT, multi-stage fine-tuning and every published evaluation need
immunotherapy cohorts, which we do not have.

**Method-defining**, kept:
- The fixed gene → 132 set → 43 concept hierarchy from
  `conception_processed.tsv`, with its softmax-attention aggregation and the
  shared linear set scorer.
- The 43-d concept-score vector as the embedding, and the 132-d set scores
  as a second embeddings file.
- FT-Transformer-style gene tokens (a per-gene weight and bias, then ReLU)
  into a single encoder layer, d = 32, 2 heads.
- The self-supervised cosine triplet loss (margin 1) on the concept vector,
  with masking/jitter augmentations.
- log2(TPM + 1) with per-gene min-max scaling. TPM is available as
  `tpm_unstranded`.

**Incidental**, standardized:
- Pretraining on each fold's training patients only. Early stopping on our
  val patients rather than a random 1%.
- Min-max statistics fit on training samples.
- **The cancer-type token is dropped.** It is a label fed in as input. The
  CANCER dimension is a function of `project_id`, and the token conditions
  every gene token through attention, so the cancer-type and within-organ
  probes would score the given label, not the expression. Our embedding is
  therefore 43-d.
- Genes: our 19,944 protein-coding default. The 916 concept genes are
  addressed by symbol, so check that all are present. Their 15,672 list,
  which is an intersection with the immunotherapy cohorts, stays available
  through `gene_ids_path`.
- GDC's TPM instead of their re-derived TPM (same STAR / GENCODE v36
  source). All training-patient samples, as for the other models, instead of
  one FFPE-free tumour per patient.
- Negatives: the released code's behaviour (K = 1, any training sample) by
  default. Same-project negatives, as the paper describes, are a variant. It
  uses training labels, which our rules allow, but it should be reported as a
  variant.
- Exact attention (SDPA/flash) in place of Performer, whose role is cost.
  Heads, fine-tuning modes and prototype classification are not applicable.

**Leakage**:
- The released weights saw all 10,184 TCGA tumour patients, and their
  scaling was fit on all of them. They cannot be scored. Use them only to
  sanity-check a reimplementation, e.g. that concept scores correlate
  similarly.
- The cancer token, above.
- The gene-set prior comes from studies that partly used TCGA (Danaher 2017
  chose markers by TCGA co-expression) or IMvigor210 data. This is a
  cohort-level prior with no per-patient fitting, in the same category as our
  MSigDB labels, so it is accepted but noted.

**Circularity with the ssGSEA probe**: this is overlap, not a leak. No fitted
statistic crosses between the two, but the bottleneck is wired to read many
of the genes our default Hallmark scores are built from. Measured against the
MSigDB Hallmark v7.0 GMT (ours uses 2026.1; Hallmark membership is nearly
stable):
- 467 of the 916 concept genes (51%) are in some Hallmark set.
- Overlap with any single set is small. The largest are Cytokine ×
  INFLAMMATORY_RESPONSE (29 genes); Cell_proliferation with E2F_TARGETS (20
  of its 46) and G2M_CHECKPOINT (19); Genome_integrity with both (16 each);
  IFNg_pathway with ALLOGRAFT_REJECTION (15 of 26); Apoptosis with APOPTOSIS
  (13 of 16); Stroma with EMT (13 of 21).
- The Hallmark sets most covered by concept genes: IL6_JAK_STAT3 44%,
  ANGIOGENESIS 39%, ALLOGRAFT_REJECTION 38%, INFLAMMATORY_RESPONSE 32%,
  EMT 28%, IFNγ response 22%.

Expect COMPASS to do relatively well on immune, EMT and proliferation
pathways, and poorly on metabolic and hormonal ones, which no concept reads.
Report per-pathway R² grouped by overlap. I did not compute the overlap with
the Reactome, PID, oncogenic or cancer_cell_atlas collections.

**Evaluation ideas to adapt:**
- **A training-free gene-set baseline**: per sample, the mean of z-scored log
  TPM (z statistics from training samples) over each of the 132 sets,
  averaged into the 43 concepts. This is the paper's own comparison
  (Extended Data Fig. 3, geometric mean and ssGSEA). It shows what the
  encoder and contrastive training add over the prior alone. Also score a
  43-component PCA beside PCA-256, since a 43-d bottleneck is otherwise
  handicapped on invertibility and classification.
- **Concept fidelity** (COMPASS-specific): within project, each concept's
  correlation with the mean expression of its own genes. The paper's sign
  flips suggest many concepts will not mean what their names say.
- **Immune-state labels not taken from the expression data**: leukocyte
  fraction from methylation and TIL fraction from slides (Thorsson 2018 /
  Saltz 2018; external resources). TMB and MSI come from the mutations in
  `tcga-patients-open` (the EVALS genomic-labels candidate). The
  Genome_integrity concept is designed to track TMB.
- **Not adopted**: immunotherapy response, leave-one-cohort-out and
  cohort-to-cohort transfer, multi-stage fine-tuning, and IMvigor210 survival.
  None of these cohorts are in our data. NFT prototypes are close to our
  geometry probe's precision@k.

**Compute** (not reported by the paper; this is my estimate): ~1.0M
parameters, 98% of them per-gene token parameters. One layer at d = 32 over
~20k gene tokens, three forward passes per triplet. With exact flash
attention, I estimate under a minute of A100 time per epoch on ~8k training
samples. With early stopping, that is about 1–3 GPU-hours for five folds.
Memory is small. The authors' extraction notebook embeds 10,184 samples in
~5 minutes (80 batches of 128; device not stated). This is far cheaper than
TxFM or BulkRNABert.
