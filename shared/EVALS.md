# Evaluation catalogue

What `reimp_shared.eval` measures, where each evaluation comes from, and
what is queued. The source papers' own protocols are recorded in each
model's `paper.md`; everything here is our standardized version: one
dataset, one patient-level 5-fold cross-validation, one set of label
definitions, frozen embeddings, the same readout for every model.

## Rules every evaluation follows

These are the points on which the surveyed papers' numbers stop being
comparable, so they are fixed here once.

1. **Patients, not samples.** Splits are by case. Aliquots of one patient
   never straddle train and test (TifBERT splits by sample).
2. **No test patients in pretraining**, even unlabelled. BulkRNABert's
   TCGA model was pretrained on ~95% of all TCGA samples, test included.
3. **Every fitted statistic is fit on training samples**: normalization
   maxima, per-gene scalings or rankings, gene filters, tokenizers, PCA,
   probe standardization, per-project centring. Labels must not depend on
   the cohort either: ssGSEA scores are the raw, per-sample ones, not
   GSVA's cohort-normalized score. TifBERT fits most of these on the whole
   cohort.
4. **Scores within cancer type where cancer type would otherwise do the
   work**: survival, pathway activity, technical confounders, and any
   per-sample regression with large between-project differences. Pooled
   pan-cancer C-indexes (both papers) mostly measure cancer type.
5. **Every score beside baselines** that need no model: PCA of the
   log-normalized counts, the 5,000 most variable of those genes as they
   are (Gross et al. 2024's strongest per-cohort survival representation,
   which asks what compressing buys), and a constant where the metric has
   one (0.5 for a within-project C-index, the mean profile for per-sample
   Pearson, 0 for within-project R² and batch enrichment). A model's gap to
   a baseline is a paired difference: the same bootstrap draws, differenced
   replicate by replicate (`probe --against`).
6. **Every score with an interval** where the bootstrap is sound: a 95%
   patient-bootstrap percentile interval, 1,000 replicates by default.
   Replicates resample patients with all their rows, and with the same
   seed they are the same draws for every model, so two models' intervals
   are paired. Three scores are reported without one because resampling
   biases them, and a percentile interval would miss the estimate: NMI
   and ARI (duplicated patients form pairs that agree by construction)
   and per-gene R² (a resample that misses the few samples carrying a
   gene's variance sends its R² far below zero). The bootstrap is
   conditional on one partition into folds and on the fitted models, and
   Gross et al. 2024 show that a lead on one test set often fails to hold
   on another. So the baselines report redraws the folds under other salts
   and measures how far the PCA baseline's scores move: a gap between two
   models smaller than that is a tie.
7. **Out of fold, pooled.** Every model is trained once per fold, and each
   fold's model and probes see only that fold's train and val patients.
   Every patient is scored once, by the fold that held it out, and scores
   are computed over the pooled predictions, not averaged over folds.
   Scores that compare two samples' predictions
   compare them within a fold (C-index pairs); scores that describe a whole
   embedding space are per fold, then averaged (spectrum, NMI, ARI).

## Implemented

| evaluation | what it asks | scores | sources |
|---|---|---|---|
| `classification_probe` / `project_id` | cancer type from a tumour sample, 33 classes | accuracy, balanced accuracy, macro-F1, weighted-F1 | BulkRNABert, TifBERT (pan-cancer classification); TxFM (linear probe) |
| `classification_probe` / `tumor_vs_normal` | tumour or adjacent normal | as above | ours |
| `classification_probe` / `lung`, `kidney`, `colorectal`, `glioma` | cancer type among those sharing an organ: LUAD / LUSC; KICH / KIRC / KIRP; COAD / READ; GBM / LGG | as above | TifBERT (LUAD vs LUSC) |
| `invertibility` | expression back from the embedding, by ridge regression | pooled and per-gene R²; per-sample Pearson beside a mean-profile baseline | TxFM (Bendidi "inv"); TifBERT (expression prediction) |
| `pathway_probe` | ssGSEA pathway activity (MSigDB Hallmark by default, `--pathways` for others) from the embedding, within cancer type, tumour samples | within-project R², pooled over pathways and averaged per pathway | TifBERT (PARADIGM pathway regression) |
| `survival_probe` | risk ranking within cancer type; PFI by default, `--endpoint` for OS/DSS/DFI; one primary tumour sample per patient | C-index over within-project pairs; mean of per-project C-indexes (projects with ≥ 10 events); each project's own C-index (`survival_projects`) | BulkRNABert, TifBERT; Gross et al. 2024 (per-cohort C-indexes) |
| `geometry_probe` | how the embedding is laid out against cancer type | effective rank, top-eigenvalue share, precision@1 and @10, NMI, ARI, silhouette | TifBERT (spectrum, precision@k); TxFM (kNN, clustering) |
| `confounder_probe` | how much processing the embedding carries within cancer type (lower is better) | within-project R² of log total reads, assigned-read fraction, strand balance; neighbour enrichment for plate and tissue source site | TxFM (batch mixing, as a TCGA analogue) |

Labels: cancer type and sample type from the expression dataset's
`samples`; survival endpoints and ssGSEA scores from `tcga-patients-open`
via `reimp_shared.labels`; technical covariates from the expression
dataset's barcodes and read tallies (`labels.technical_covariates`).

**Survival samples are primary tumours**, as TCGA-CDR (Liu et al. 2018)
recommends and SurvBoard and MultiSurv do; patients without one are not
scored. That leaves out 371 patients with tumour samples: 366 in SKCM,
sampled only at metastasis, and 5 with only a recurrence (4 GBM, 1 OV).
Metastatic samples would bias the probe in two ways, measured on SKCM with
the date of the submitted tumour
(`clinical_supplement.patient.submitted_tumor_dx_days_to` in
`tcga-patients-open`, known for 353 of the 366):

- *Guaranteed survival.* The endpoints start at diagnosis, but these
  samples were taken a median 689 days after it (IQR 105–1,653; 64% more
  than a year), and no patient died before their sample. Their median OS
  is 1,617 days counted from diagnosis but 578 counted from sampling,
  against 446 for SKCM patients sampled at their primary (a median 0 days
  after diagnosis). Within SKCM, many comparable pairs would set an early
  primary-sampled event against a metastasis-sampled patient guaranteed to
  last years, rewarding an embedding for telling metastatic from primary
  tissue rather than for prognosis.
- *Outcome before the sample.* For 88 of the 353 (25%, and 32% of their
  events) the first PFI event falls on or before the sample date: the
  sample is often the progression it would be scored on predicting. The
  same holds for 2 of 102 patients sampled at a primary.

Gross et al. 2024 and Thorsson et al. 2018 include such samples
uncorrected. The corrections — time from sampling, as TCGA's melanoma
study did, or delayed entry into risk sets — work for OS but not for PFI,
since TCGA-CDR records only the first progression. Neither is adopted; the
cost is a thin SKCM row, 102 patients and 37 PFI events.

COAD vs READ is close to one disease — TCGA's colorectal study analysed
them together — so near-chance scores there are expected. Tissue source
sites also differ in their patients, so their enrichment is not purely
artefact. Pathway scores are computed from each sample's own expression,
so the pathway probe is a biologically weighted relative of invertibility
rather than an independent label.

## Candidates, roughly in priority order

1. **Genomic labels** (none of the surveyed papers): driver-gene mutation
   status (TP53, KRAS, PIK3CA, ...), oncogenic-pathway alteration
   (Sanchez-Vega 2018), microsatellite instability — ground truth that is
   not derived from expression. Mutations are in `tcga-patients-open`.
2. **Molecular subtypes** within a cancer type (e.g. BRCA PAM50) — harder
   than the within-organ tasks, which PCA mostly solves; needs an external
   subtype resource.
3. **Gene-level relationship recall** (TxFM): gene representations —
   embedding tables, decoder weights — scored against CORUM, StringDB,
   Reactome and others. A model-level evaluation for models that have
   per-gene parameters; needs the external databases.

Labels that prove out are candidates for publishing with the expression
dataset (a per-case `cases` config), which `reimp_shared.labels` would
then read.

## Not adopted

| evaluation | source | why not |
|---|---|---|
| 5-cohort classification | BulkRNABert | saturated: PCA + SVM reaches 0.968 weighted-F1 |
| per-cohort survival, cohort transfer | BulkRNABert | a fixed 10% test split leaves 10–25 events per cohort; revisit on cross-validated embeddings, which score every patient |
| expression-prediction error by token position | TifBERT | specific to windowed sequence models; invertibility covers the rest |
| GTEx transfer | TifBERT | not in our dataset |
| gene-order permutation tests | TifBERT | specific to rank-ordered models |
| robustness to missing genes | BulkRNABert | needs each model's embedding function, not an embeddings file; possible later |
| single-cell perturbation and cell-type benchmarks | TxFM | not bulk TCGA |
| sequencing centre as a confounder | ours | covered by plate: every one of the 250 plates belongs to a single centre. 86% of samples come from one centre (UNC), and centres mix within only GBM and STAD |
| PARADIGM pathway scores | TifBERT | an external Xena file, z-scored over the whole cohort; tcga2hf's per-sample ssGSEA scores replace it |
| gene essentiality | Gross et al. 2024; BulkFormer | needs DepMap cell lines |
| a pooled pan-cancer C-index | BulkRNABert; Gross et al. 2024 | mostly measures cancer type (rule 4) |
| tuning each representation, or an MLP-Cox head, on the downstream label | Gross et al. 2024 | answers "best pipeline"; here every frozen embedding gets the same linear probe |
| a win criterion over repeated retraining (≥ 75% of splits) | Gross et al. 2024 | retraining every model per split is too costly; the baselines' fold redraws stand in (rule 6) |
| metastatic or recurrent samples in survival | Gross et al. 2024; Thorsson et al. 2018 | sampled after diagnosis: guaranteed survival, and for PFI outcomes that precede the sample (see above) |
