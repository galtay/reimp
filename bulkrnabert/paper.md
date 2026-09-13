# BulkRNABert — what the paper did

Gélard, Richard, Pierrot, Cournède. *BulkRNABert: Cancer prognosis from
bulk RNA-seq based language models*. ML4H 2024, PMLR 259:384–400.
[proceedings](https://proceedings.mlr.press/v259/gelard25a.html),
[bioRxiv 10.1101/2024.06.18.599483](https://doi.org/10.1101/2024.06.18.599483).
Code and weights: [instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research),
[InstaDeepAI/BulkRNABert](https://huggingface.co/InstaDeepAI/BulkRNABert).

Read: the PMLR version in full (page numbers below refer to it). bioRxiv
v4 has the same numbers. bioRxiv v1 differs: it adds a 12,403
highly-variable-gene model and lacks Table 4, Figures 3–4 and Appendix D.
The follow-up MOJO (Gélard et al., bioRxiv 10.1101/2025.06.25.661237,
v2 May 2026) is bimodal, RNA plus 450k methylation; its RNA half, a
convolutional U-Net on the same tokens, has its own stub in `mojo/`. It
re-scores BulkRNABert only on the 9,252 samples with both modalities,
after pretraining MOJO on the whole paired cohort. There is no
separate survival paper.

## Model

- BERT-style encoder: 4 layers, 8 heads, d = 256, FFN 512, ~6M
  parameters (§3.1, p.388). No positional encoding; a Gene2Vec embedding
  per gene (dim 200, projected to 256) is added to each token's
  expression embedding.
- Expression tokens: TPM → log10(1 + x) → divided by a dataset-wide
  maximum (5.547 for the TCGA model) → 64 linear bins. Bin 0 is exact
  zero; no per-gene scaling.
- Objective: masked language modelling. 15% of tokens are selected, zeros
  included, and replaced 80/10/10 mask/random/kept. 12B tokens, AdamW,
  ~3M tokens per batch, TPU v4.
- Sample embedding: mean over genes of the last layer.
- Genes: those shared by GTEx, ENCODE and TCGA. The paper says 19,042;
  the repo's `data/bulkrnabert/common_gene_id.txt` has 19,062 unversioned
  Ensembl IDs. All 19,062 are in GENCODE v36: 16,905 protein-coding,
  1,244 lncRNA, the rest pseudogenes and small RNAs.
- Weights released for all three pretrained models and two downstream
  heads. The released heads are frozen-encoder probes, not the headline
  IA3-finetuned models.

## Pretraining data

Three models: TCGA only (11,274 samples, all projects), GTEx + ENCODE
(20,406), and all three. A random 5% of each dataset is held out, at the
sample level, for monitoring the MLM loss. GDC release not stated.

**Pretraining is transductive**: the TCGA model saw ~95% of all TCGA
samples, including every downstream test sample.

## Evaluations as published

Shared protocol (§3.2, p.388): 80/20 train/test, stratified by class and
grouped by patient, repeated over 5 seeds; mean ± (presumably) SD. No
validation set. Downstream hyperparameters not stated. Whether normal
tissue samples are included is not stated for any task. The repo has no
evaluation code — only preprocessing, the model, and inference.

1. **Pan-cancer 33-way classification** (Table 1 p.390; Table 6 p.398).
   Heads: SVM on frozen embeddings, or an MLP [256, 128] (SELU) with
   optional IA3 finetuning of the encoder. Macro-F1 / weighted-F1:
   BulkRNABert(TCGA) + MLP + IA3 0.918 / 0.942; + SVM 0.902 / 0.933;
   PCA + SVM 0.849 / 0.877; NMF + SVM 0.749 / 0.800. Pretraining ablation
   (Table 4 p.395), macro-F1: pretrained 0.918, GTEx+ENCODE pretrained
   0.914, random init + full finetune 0.894.
2. **5-cohort classification** — BRCA, BLCA, GBM+LGG merged, LUAD, UCEC
   (Table 5 p.397). Weighted-F1: BulkRNABert 0.991–0.993; PCA + SVM 0.968.
   Saturated.
3. **Pan-cancer survival** (§3.5.1; Table 2 p.391, Table 7 p.398). 11,035
   samples; endpoint "time until the death … from the time of diagnosis",
   source not stated (example rows match GDC `demographic.days_to_death`).
   MLP [512, 256] with LayerNorm under the Cox partial likelihood, frozen
   encoder. Three C-indexes: pooled over all test samples, per-cohort
   averaged weighted by cohort size, and per-cohort unweighted. TCGA model:
   0.765 / 0.642 / 0.656. The pooled score mixes in between-cohort survival
   differences; the per-cohort scores are the ones comparable to a
   within-cancer-type evaluation.
4. **Per-cohort survival**, five cohorts (Table 3 p.392). C-index,
   BulkRNABert vs an MLP on raw expression: GBMLGG 0.844 vs 0.831, LUAD
   0.648 vs 0.550, UCEC 0.703 vs 0.609, BLCA 0.627 vs 0.598, BRCA 0.604
   vs 0.676. SDs 0.02–0.07. GBMLGG largely measures GBM vs LGG.
5. **Pan-cancer → per-cohort transfer** (Fig. 5 p.392): per-cohort model
   vs pan-cancer model vs pan-cancer with the cohort left out, on the five
   cohorts plus CHOL, ACC, UCS, DLBC. Box plots only.
6. **Minor**: cosine clustering of cohort mean embeddings (Fig. 4);
   robustness to missing genes, imputed by the model — ~90% of full
   performance up to 70% missing, collapse at 90% (App. D, Fig. 12); MLM
   reconstruction accuracy 0.5–0.6 (Fig. 7).

Baselines are thin: CustOmics and MAE numbers are copied from other papers
with other splits, and there is no logistic regression on log TPM.

## For reimp

**Method-defining**, kept: masked language modelling over binned
expression tokens; the 64-bin linear tokenizer on normalized log TPM; a
per-gene embedding added to the expression embedding; mean pooling over
genes. TPM input is available as `tpm_unstranded`, and because the
tokenizer divides by a maximum, natural-log `log1p` gives the same bins as
log10.

**Incidental**, standardized:
- Pretraining on TCGA training patients only, which removes the
  transductive leak.
- The normalization maximum computed on training samples only.
- The shared patient-level split instead of 5 seeds of 80/20.
- The gene set: our default protein-coding set (19,944), rather than
  their GTEx/ENCODE/TCGA intersection. Their list maps 1:1 onto GENCODE
  v36, so it remains available through `gene_ids_path` for a
  closer-to-paper variant.
- Gene2Vec is left out: gene embeddings are learned from scratch, as in
  TxFM. It is an external resource trained on other data, and learning
  them keeps every model's gene representations a product of TCGA
  training patients alone.

**Evaluation ideas to adapt:**
- Pan-cancer classification → already the shared `classification_probe`
  (ours excludes normals). Add weighted-F1 alongside macro-F1. For
  context only: our PCA + logistic regression reaches macro-F1 0.948 on
  our test split against their PCA + SVM 0.849 — different setups, but
  their PCA baseline looks weak.
- Pan-cancer survival → already the shared `survival_probe`
  (`--endpoint os` for their endpoint). Consider also reporting the
  unweighted per-project mean C-index; ours pools comparable pairs within
  projects instead.
- Per-cohort survival and cohort transfer → only with repeated
  cross-validation; our fixed 10% test split leaves 10–25 events per
  cohort.
- Missing-gene robustness → a possible generic evaluation, but it needs
  each model's embedding function, not just its embeddings file.
