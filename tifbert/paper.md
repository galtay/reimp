# TifBERT — what the paper did

Hosseini, Sharma (York University). *TifBERT: a self-supervised foundation
model for normalization-robust bulk RNA-seq representation learning*.
bioRxiv, v1 posted 2026-06-11,
[10.64898/2026.06.08.728683](https://www.biorxiv.org/content/10.64898/2026.06.08.728683v1).
Probably also the ICML 2026 paper *Self-Supervised Contextual
Representation Learning for Transcriptomic Generative AI* ("SpecFormer"),
[OpenReview L22k0p6Oyr](https://openreview.net/forum?id=L22k0p6Oyr), same
authors and headline numbers. Code:
[mohsenh17/TifBERT](https://github.com/mohsenh17/TifBERT) (15 commits,
2026-01-26 to 2026-06-15, no releases).

Read: the preprint's full text and Tables 1, 2 and 4. Table 3 and
supplementary tables S1–S6 were not retrievable (bioRxiv rate limiting);
Table 3's numbers below come from the text. The repo does not run as
committed (old package name in imports, hard-coded cluster paths), ships
no weights, data, splits or vocabulary — despite the paper saying weights
are public — and disagrees with the paper on several hyperparameters.

## Model

- BERT-base via Hugging Face `BertForMaskedLM`: 12 layers, 768 hidden, 12
  heads, ~93M parameters.
- **Tokens are gene identities only.** Each sample becomes its genes
  sorted by a per-sample score (called TF-IDF); no expression value enters
  the model, so a gene's rank is its position. Masked gene modelling, 15%
  masked, 80/10/10. Word-level vocabulary over HUGO symbols.
- ~10,000 genes per sample, cut into 512-token windows with stride 256
  (~39 windows per sample; some configs use 2048). Downstream, a sample
  is its stack of per-window CLS and mean-pooled embeddings.
- The ranking score. The paper describes within-sample TF times per-gene
  IDF. The code (`compute_gene_tfidf`) fits sklearn's `TfidfTransformer`
  on a genes × samples matrix, so each gene is L2-normalized across the
  cohort: the ranking is expression divided by that gene's cohort-wide
  norm. Built from Toil TPM (a comment says originally expected counts).
- Genes: UCSC Xena PANCAN, HUGO symbols, those in all five Xena
  representations, minus genes with median < 0.1 or bottom-10% variance —
  9,932 in the final vocabulary.
- Pretraining: TCGA only, 9,201 samples split 90/5/5 *by sample* (8,315 /
  406 / 480). Paper: AdamW 1e-4, 350 epochs, 4 H100s (the code hard-codes
  2e-5).

## Evaluations as published

Protocol for 1–4: frozen encoder; per task a trained head pools the
windows (an attention aggregator — called "gated" in the paper, not gated
in the code) into an MLP [512, 256]; AdamW 5e-4, 100 epochs, best epoch
on validation. One split, no repeats, a test set of 480 samples.

1. **Cancer type, 33 classes** (Table 1), with the trained head. Labels
   from Xena `_primary_disease`; whether normals are included is not
   stated. TF-IDF ordering: accuracy 0.908 [0.881, 0.933], macro-F1 0.853,
   weighted-F1 0.904, MCC 0.904, top-3 0.973, macro AUC 0.997. Ordering by
   raw expression 0.848 accuracy, by z-score 0.800. Bootstrap 95% CIs.
   Only baseline (Table 2): logistic regression on a 0/1 indicator of each
   sample's top-200 genes, accuracy 0.650. No PCA or linear baseline.
2. **Expression prediction.** Frozen encoder + a token-level head
   predicting five normalizations per gene; per-position error, UMAPs,
   Procrustes of cancer-type centroids (Figs. 2–4). No headline number.
   The code trains this head on the test split.
3. **Pathway activity** (Table 3, numbers from the text). Regress 1,387
   PARADIGM pathway scores from the Xena Pan-Cancer Atlas, z-scored over
   all samples. Sample-wise / pathway-wise Pearson 0.754 / 0.762 (TF-IDF
   ordering), 0.716 / 0.744 (expression), 0.713 / 0.730 (z-score). No
   baseline outside TifBERT.
4. **Survival**, TCGA-CDR OS, DSS, PFI, DFI. Deep Cox head; C-index pooled
   across all cancers, not stratified (per-cancer C-indexes, time-dependent
   AUC, integrated Brier and log-rank are computed in code). TF-IDF
   ordering: OS 0.646, DSS 0.672, PFI 0.612, DFI 0.671; ordering by
   expression best on OS at 0.684. A PCA-100 + Cox baseline exists in
   code; its results are not reported.
5. **GTEx transfer**: 500 samples, 27 tissues; Procrustes disparity
   between tissue centroids of actual and predicted expression, 0.2065.
   No predictive task.
6. **Representation quality vs BulkRNABert** (Table 4): 480 samples, 11
   cancer types with ≥ 20 samples each — inside BulkRNABert's pretraining
   data. None of this code is in the repo.

   | metric | TifBERT | BulkRNABert |
   |---|---|---|
   | effective rank | 95.6 | 6.3 |
   | top-eigenvalue share | 0.114 | 0.585 |
   | LUAD (26) vs LUSC (23) linear probe | 69.3 ± 6.5% | 71.6 ± 19.2% |
   | precision@1 / @5 / @10 / @20, cosine kNN by cancer type | 0.644 / 0.524 / 0.460 / 0.371 | 0.684 / 0.589 / 0.519 / 0.431 |
   | 5-fold linear probe, cancer type | 73.4 ± 3.8% | 83.4 ± 3.5% |

   With a linear readout of frozen embeddings, BulkRNABert is ahead on
   classification and retrieval; TifBERT's advantages are geometric
   (effective rank, spread of the spectrum) and a steadier lung probe.

Leakage the numbers carry: splits are by sample, so a patient's tumour
and normal can straddle train and test; the ranking statistics, z-scores,
gene filters, tokenizer and pathway z-scoring are all fit on the whole
cohort before splitting.

## For reimp

**Method-defining**, kept: masked language modelling over gene-identity
tokens with no expression values; each sample as its genes ranked by a
normalized score; windowed sequences; mean pooling. Tokens become Ensembl
IDs instead of HUGO symbols.

**Incidental**, standardized: GDC data instead of Xena/Toil; our
patient-level split, with pretraining on training patients only; every
fitted statistic — the per-gene normalization behind the ranking, any
gene filters — fit on training samples only; the gene set from our
defaults. For the shared frozen-embedding protocol, the sample embedding
is the mean over windows rather than a per-task trained pooling head.

**Ranking**: parameterized, in `reimp_shared.ranking.GeneRanker` —
`expression`, `z`, `tfidf` with a `count`, `smooth` or `entropy` rarity
weight, and `cohort_l2`, the per-gene cohort normalization the released
code actually computes. The default is `tfidf` with the entropy weight,
idf_g = log N − H_g, H_g the entropy of the gene's share of its total
across training samples. It comes from the RNA-caption topic-model work
(`biospecimen-captions-prototype`), where on dense bulk data the count
form gave zero weight to 91% of genes at a detection threshold of 0. On
our training samples (protein-coding TPM, threshold 0) it zeroes 51.7% of
genes — every gene detected in all 9,190 samples — against 1.9% for the
entropy and smoothed forms (genes never expressed in training). The
entropy form needs no threshold and equals the count form for a gene
uniform on the samples that express it. Every statistic is fit on
training samples.

**Evaluation ideas to adapt:**
- 33-way cancer type → already `classification_probe`, which is a linear
  readout, so their comparable number is Table 4's 73.4%, not Table 1's
  90.8%. Worth taking: bootstrap confidence intervals over test patients,
  and weighted-F1.
- Effective rank, top-eigenvalue share and precision@k retrieval by
  cancer type — cheap, model-agnostic geometry scores.
- LUAD vs LUSC → a family of harder within-organ discriminations (lung,
  kidney, colorectal, glioma), which we have enough samples to score.
- Survival on all four endpoints → already `survival_probe --endpoint`,
  scored within project; their pooled C-index is not comparable.
- Pathway regression → with our own ssGSEA scores (MSigDB, from
  `tpm_unstranded`) rather than PARADIGM; mostly another invertibility
  test, since the targets come from the same expression.
- Top-k indicator + logistic regression → a rank-based model-free
  baseline, weak next to PCA (0.650 accuracy).
