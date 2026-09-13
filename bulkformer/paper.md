# BulkFormer — what the paper did

Kang, Fan, Yi, Cui, Cui. *BulkFormer: A large-scale foundation model for
bulk transcriptomes*. Cell Systems 17(7):101657, 2026 (online 1 July 2026;
PMID 42385705).
[doi 10.1016/j.cels.2026.101657](https://doi.org/10.1016/j.cels.2026.101657),
preprint [bioRxiv 10.1101/2025.06.11.659222](https://doi.org/10.1101/2025.06.11.659222)
("A large-scale foundation model for bulk transcriptomes"; v1, 17 June 2025,
the only version).
Code: [KangBoming/BulkFormer](https://github.com/KangBoming/BulkFormer)
(MIT). Data and resources: Zenodo
[15559368](https://doi.org/10.5281/zenodo.15559368) (May 2025, with the
preprint's checkpoint) and [15744294](https://doi.org/10.5281/zenodo.15744294)
(Dec 2025, current; both CC-BY 4.0). Current weights for five model sizes
are on Google Drive, linked from the README.

Read: bioRxiv v1 in full, including the supplement (line numbers below
refer to it). The journal version is paywalled: we read only its abstract
(Europe PMC) and what the repo README reports from it. The repo at HEAD
(`5bcf5b9`, 3 Jul 2026) with its history back to 31 May 2025, both Zenodo
records, and GitHub issues #1–#9. Differences between versions:
- Pretraining profiles: 522,769 in the preprint, 581,503 in the journal.
- Tasks: the preprint has six. The journal and the December 2025 README
  have five: compound-perturbation prediction (LINCS) was dropped because
  L1000 profiles are not bulk RNA-seq.
- Added in the journal: a cardiovascular-disease and a GTEx tissue
  classification task, and State and Cell2Sentence as baselines.
- Some journal metrics differ: drug response and gene essentiality are now
  "mean PCC" (0.373 and 0.186) rather than the preprint's pooled PCC (0.910
  and 0.931).
- Code (Dec 2025): gene identities are learned (`nn.Embedding`) instead of
  ESM2, three sample summary scalars feed the output head, the layout is 1
  GCN + 12 Performer layers, and the gene graph file changed from
  `G_gtex.pt` to `G_tcga.pt`. Details below.

## Model

- Input: counts → TPM using the repo's gene-length table → natural `log1p`.
  Genes absent from a sample, and masked positions, are both set to −10.
- **Expression embedding ("REE")**: fixed sinusoid of the scalar value,
  `[sin(x·θ), cos(x·θ)]` with `θ_i = 100^(−2i/d)` (`utils/Rope.py`). It has
  no parameters, and masked positions get a zero vector. Values are not
  binned.
- **Gene identity**:
  - Preprint: a trainable table initialised from ESM2 (the canonical
    protein, mean-pooled, 1280-d), then an MLP 1280→2560→640 (l.431–436).
  - Current code: `nn.Embedding(20010, 640)` with Xavier init, then an MLP
    640→2560→640. ESM2 is kept only as an optional concatenation to
    gene-level outputs.
- **Sample context**: an MLP 20,010→2,560→640 over the whole masked input
  vector, broadcast-added to every gene token (l.452–458). At 52.9M
  parameters it is the largest single component.
- The three embeddings are summed, then pass through an MLP 640→2560→640.
- **Blocks**: each block is LayerNorm → `x + GCNConv(x, G)` → K Performer
  layers. The Performer layers come from performer-pytorch 1.1.4 with
  pre-LN, 8 heads × 80, FFN ×4 with GELU, and dropout 0.05/0.1 (v1 code:
  0.2/0.2).
  - Preprint (Supp. Table 5): N = 3 blocks × K = 4 Performer layers, d = 640.
  - Current `model/config.py` (147M): N = 1, K = 12, d = 640.
  - The other released sizes are 127M (d 640, K 8), 93M (d 512, K 6),
    50M (d 256, K 2) and 37M (d 128, K 1).
  - The May 2025 code also had an optional "bins" path: genes sorted by a
    learned score and passed through a local Performer per bin. It was
    disabled (`bins=0`).
- **Gene graph**: gene co-expression, chosen over GO-similarity and PPI
  graphs in an ablation (Supp. Fig. 5). Construction (l.464–474):
  - edge weight = |Pearson r|;
  - keep the top 20 edges per gene;
  - drop edges with r < 0.4.

  The preprint does not say which expression data the correlations were
  computed on. Released files:
  - `G_gtex.pt` (Zenodo v1): 366,899 edges, including 18,965 self-edges.
  - `G_tcga.pt` (Zenodo v2, loaded by the current notebook): 329,728
    edges, including 16,875 self-edges. Every gene has out-degree exactly
    20.
  - The two graphs share 5% of their edges.

  `GCNConv(add_self_loops=False)` uses symmetric normalisation, and the
  graph is cached.
- **Output head**:
  - v1: an MLP 640→2560→1 with a final ReLU.
  - Current: the 640-d token, three per-sample scalars (the mask ratio,
    the mean observed expression and the non-zero fraction), LayerNorm,
    then an MLP with a ReLU output. The prediction is shifted at masked
    positions so that the mean prediction on observed genes matches their
    observed mean.
- **Objective**: about 15% of genes per sample are replaced by −10, and the
  loss is MSE on those positions (l.494–503). There is no 80/10/10 scheme.
- **Optimisation**: AdamW, peak learning rate 1e-4, linear warmup over 5%
  of steps, 29 epochs.
  - Batch: 4 per device with 128 accumulation steps (512 per device). The
    paper does not say whether that is multiplied by the 8 GPUs.
  - Hardware: 8 × A800, "approximately 350 GPU hours" (l.507–512). Supp.
    Table 2 instead gives ~96 h per epoch on one GPU, which over 29 epochs
    would be ~2,800 GPU-hours. The two figures disagree.
- **Sample embedding**:
  - Preprint: max pooling over genes of the final Performer layer
    (l.519–520).
  - Current notebook: pools only over a fixed 2,000-gene list,
    `interested_gene_list.pt`. It is identical to v1's
    `high_var_gene_list.pt` and starts H4C3, S100A9, SRGN, CD74, S100A6,
    LYZ, HBA2. It is described as highly variable, but on which data is
    not stated.
  - Notebook default is mean pooling (also max, median, or "all" = their
    sum).
  - The output is 643-d: 640 plus the three appended scalars.
- **Genes**: 20,010 protein-coding genes (Ensembl). 19,907 match GENCODE
  v36 by unversioned ID, and 19,829 of those are protein-coding there.
  Use the repo's `data/bulkformer_gene_info.csv`. The v1 Zenodo copy has a
  stray row `35991` and a different order (issue #1).
- **Parameters**, counted by instantiating the code:
  - May 2025 code with the preprint configuration: 148.6M. That matches
    the 596 MB fp32 checkpoint on Zenodo v1. It breaks down into a 25.6M
    ESM2 table, the 52.9M sample MLP, and a 60.3M trunk.
  - Current "147M" configuration: **132.1M** (12.8M gene table, 52.9M
    sample MLP, 59.5M trunk).
  - The other size names are similarly inflated: "37M" counts 13.4M,
    "93M" counts 75.9M.
- **Released weights**:
  - v1 checkpoint (`Bulkformer_ckpt_epoch_29.pt`) on Zenodo 15559368.
  - Five current checkpoints on Google Drive. The links were
    permission-locked once (issue #5, since fixed).
  - The graph is an input, not a parameter. So a checkpoint loads with any
    graph, and "All keys matched" does not show which graph it was trained
    with.

## Pretraining data

"PreBULK" (l.399–424):
- Sources: human bulk RNA-seq from GEO and ARCHS4, samples with raw counts
  only, deduplicated by GSM.
- Genes: zero-padded to the 20,010-gene list.
- QC: profiles with fewer than 14,000 non-zero genes are removed, to
  exclude scRNA-seq.
- Content: nine physiological systems, healthy and diseased.
- Size: 522,769 profiles in the preprint, 581,503 in the journal. The
  journal's composition is unverified (it is in its Supp. Table S10, which
  we could not read).
- Release: `PreBULK.h5ad.zip` (35.3 GB) on Zenodo v2, whose `obs` holds
  only a train/test `split`. In July 2026 the authors posted a row map to
  ARCHS4 `meta_index`, GSM and GSE (issue #8).
- How the split was made is not stated; presumably random by sample.

**No TCGA or GTEx in pretraining**, as far as can be checked. PreBULK
rows map to ARCHS4, which holds public GEO/SRA samples, and TCGA and GTEx
raw data are not deposited there. The paper uses TCGA only downstream.
ARCHS4 does contain cell-line RNA-seq, which matters for their GDSC and
DepMap tasks but not for us.

**The gene graph is another matter** (see Leakage). We checked the current
graph against our data. For each gene, we took its top-20 |r| neighbours
over all 11,505 TCGA samples of our dataset (19,907 mapped genes).
Fraction of each graph's non-self edges that are among those neighbours:

| graph | linear TPM | log1p TPM | log1p, tumours only |
|---|---|---|---|
| `G_tcga.pt` | 50% | 33% | 32% |
| `G_gtex.pt` | 8% | 8% | 8% |

By chance one would expect 0.1%. The current graph is almost certainly TCGA
co-expression, computed on TCGA samples that include the paper's own
evaluation patients. That `G_gtex.pt` comes from GTEx is inferred from its
name only.

## Evaluations as published

Shared protocol for sample-level tasks (l.241–279, 513–533):
- Embeddings are reduced by PCA (to how many components is not stated) and
  read out by a random forest.
- 10-fold cross-validation. Patient grouping, the scope of the PCA fit,
  and any validation set are not stated.
- Baselines are single-cell foundation models only (Geneformer,
  GeneCompass, scGPT, scFoundation, scLong; the journal adds State and
  Cell2Sentence). These receive the top 2,000 expressed genes, max-pooled.
- There is no PCA-of-expression, raw-expression or linear baseline on the
  classification or prognosis tasks.
- The repo has no evaluation code. Issue #4 reports that the drug-response
  and perturbation numbers could not be reproduced.

Numbers are preprint (Supp. Table 1) unless marked "journal". Journal
numbers come from the README.

1. **Imputation** (Fig. 3a–c).
   - 15% of genes masked on the PreBULK test split: PCC **0.954**, against
     VAE 0.806, gene-wise mean 0.754, median 0.743, scFoundation 0.142,
     scLong 0.041.
   - Varying the mask ratio: 0.949 at 15%, 0.752 at 35%.
   - 1,000 random TCGA patients: **0.914**.
   - The PCC is pooled over masked entries. It rewards getting each gene's
     average level right, which is why the gene-mean baseline already
     reaches 0.754.
2. **Imputation applications** (Fig. 3d–i), anecdotal:
   - Extra DEGs in a pancreatic cancer cohort (GSE132956).
   - "New" prognostic genes in 8 TCGA cohorts among genes missing from more
     than 95% of samples: H4C1 in KIRC (HR 5.2), PDE6H in PAAD (HR 0.26).
3. **Disease annotation**, DiSignAtlas (Fig. 4a–b). 23 diseases with 655 to
   3,199 samples each (Supp. Table 3).
   - Weighted-F1 **0.939** (scGPT 0.885). Journal: 0.949.
   - Journal only: cardiovascular-disease classification 0.978, GTEx tissue
     classification 0.969.
4. **TCGA 33-way cancer type** (Fig. 4c; Supp. Table 4 lists 9,784
   samples). UCEC has only 181, against ~550 TCGA cases, so some filter was
   applied; it is not stated.
   - Weighted-F1 **0.833**, against scGPT 0.830, scFoundation 0.791,
     GeneCompass 0.761, Geneformer 0.473, scLong 0.347.
   - Journal: 0.907 (State 0.851, scGPT 0.813).
   - For scale: our PCA + logistic regression reaches macro-F1 ~0.95.
     Both the random-forest-on-PCA readout and the absent expression
     baseline flatter the comparison.
5. **Prognosis** (Fig. 5a–c). About 10,000 TCGA patients from 33 cancers.
   - Label: binary alive/dead status, not time-to-event, and censoring is
     ignored.
   - Pooled pan-cancer AUROC **0.747**, AUPRC 0.549 (scFoundation
     0.726/0.520). Pooled, the score mostly reflects cancer type and
     follow-up length.
   - Also (Fig. 5d–e): random-forest risk scores from gene-level
     embeddings in 8 cancers. For example, RPS27 in LGG goes from HR 1.0
     to 4.77.
6. **Compound perturbation**, preprint only. PRnet's LINCS split: PCC
   0.493, against PRnet 0.408. Later withdrawn.
7. **Drug response** (Fig. 6f–g). GDSC, 255 compounds × 700 cell lines.
   Compound features from KPGT plus the cell-line embedding, into an MLP;
   10-fold CV.
   - PCC **0.910** (scFoundation 0.880).
   - Journal: "mean PCC" 0.373, a different metric.
8. **Gene essentiality** (Supp. Fig. 4). DepMap, 17,862 genes × 1,103 cell
   lines; gene-level embeddings into an MLP.
   - PCC **0.931**, SCC 0.759.
   - Journal: mean PCC 0.186.
9. **Minor**:
   - GO/KEGG enrichment of k-means clusters of the gene embeddings
     (Fig. 2e–j).
   - UMAP of TCGA embeddings (Fig. 4d).
   - The graph ablation (Supp. Fig. 5).
   - A scaling-law curve over the five sizes (README).
   - Training cost versus single-cell models (Supp. Table 2).

## For reimp

**Method-defining**, kept:
- Masked-value regression on continuous `log1p` TPM: about 15% of genes
  replaced by a placeholder, MSE on those positions.
- The fixed sinusoidal expression embedding plus a gene-identity
  embedding, and the whole-sample MLP context added to every token.
- The hybrid block over **all ~20k genes**: a GCN on a sparse gene
  co-expression graph (top-20 |r| ≥ 0.4, with self-edges), then Performer
  linear attention. This is what separates it from BulkRNABert (binned
  tokens) and TxFM.
- Sample embedding by pooling final-layer gene tokens:
  - Default: max over all genes, as in the paper. Mean is a variant.
  - Drop the three appended scalars. Two of them are raw per-sample
    statistics (mean expression, non-zero fraction) and would feed depth
    and quality straight into the confounder probe.
  - Skip the 2,000-gene pooling list: its source is unknown.

**Incidental**, standardized:
- **The gene graph is fit per fold, on training samples only** (rule 3).
  The released `G_tcga.pt` cannot be used, because it contains the test
  patients' co-expression. Building it is cheap: our check ran in 6 s on
  CPU.
  - Use `tpm_unstranded`. Linear TPM reproduced their graph better than
    `log1p`, but either is defensible; pick one and record it.
  - `G_gtex.pt` (external, non-TCGA) or a STRING PPI graph would not leak.
    Like Gene2Vec for BulkRNABert, they stay out of the default and can be
    ablation variants.
- Gene identities are learned from scratch, not ESM2, as the authors
  themselves switched to in Dec 2025.
- Pretraining on ~8,000 TCGA training patients per fold instead of 0.5M
  ARCHS4 profiles. This is the large gap: the paper's pitch is scale.
- Genes: our default protein-coding set (19,944). Their list maps to 19,907
  GENCODE v36 IDs, so it is available through `gene_ids_path`.
- No other fitted statistics. TPM is per sample and the expression
  embedding is fixed.
- **Size**: a smaller model than the 132M one.
  - At 8,000 samples, the 52.9M-parameter sample MLP (20,010→2,560) alone
    can memorise its training set.
  - Start from d = 256, N = 1, K = 4, with the sample MLP's hidden width cut
    to d. That is roughly 5M in the gene table, 5M in the sample MLP and
    3M in the trunk.
  - Keep the full configuration as a variant.
- **Dependencies**:
  - performer-pytorch is unmaintained (1.1.4, 2021-era). Take FAVOR+
    attention from it or reimplement it.
  - A fixed, normalised adjacency makes the GCN a single `torch.sparse`
    matmul, so torch_geometric is not needed.

**Compute** (rough). A sample is 20k tokens:
- Full model: ~60M trunk parameters → ~2.5 TFLOPs forward, ~7.5 TFLOPs per
  training sample.
- Their own figure: 350 A800-hours ÷ (522,769 × 29) ≈ 0.08 GPU-s per
  sample-epoch. That is ~11 min per epoch over 8,000 samples.
- But 29 epochs of 8,000 samples is only ~450 optimizer steps at an
  effective batch of 512, against their ~30k. Reaching a comparable step
  count means hundreds of epochs: ~50–200 A800-hours per fold for the full
  model, or 250–1,000 across 5 folds.
- If Supp. Table 2's 96 h per epoch is right instead, multiply by ~8.
- The d = 256, K = 4 model is roughly 20× cheaper per token, so a few
  GPU-hours per fold.
- Memory: 20k tokens × d × layers of activations (their 4 samples per
  device) calls for bf16 and activation checkpointing.

**Leakage in the published work:**
- **The current gene graph is TCGA co-expression** (our check above). Its
  TCGA classification, prognosis and imputation numbers were produced by a
  model whose fixed architecture encodes second-order statistics of the
  evaluation cohort. The paper does not say which data the graph came
  from.
- The pooling gene list's source is unstated. If it came from TCGA, that
  is a second leak.
- PCA before the random forest, and all 10-fold splits: neither the fit
  scope nor patient grouping is stated.
- Pretraining itself (ARCHS4) does not appear to contain TCGA. So,
  unlike BulkRNABert, the expression values of test patients were not
  seen, only their correlations.

**Evaluation ideas to adapt:**
- 33-way cancer type → already `classification_probe/project_id`, with
  weighted-F1. Their random forest on PCA is not adopted.
- Prognosis (binary alive/dead, pooled AUROC) → already `survival_probe`
  (`--endpoint os`, within-project C-index). Binary status is not adopted:
  it ignores censoring.
- Imputation → `invertibility` covers expression-from-embedding but not
  masked imputation.
  - A generic **masked-reconstruction** eval for masked-value models
    (BulkFormer, BulkRNABert, TxFM): mask 15% of each test patient's genes
    and predict them.
  - Score per sample, centred per gene, beside the gene-mean baseline
    their pooled PCC hides.
  - Like missing-gene robustness, it needs each model's forward pass, not
    an embeddings file.
- **Graph ablation** as a model-level experiment, run on the shared
  probes: no graph (Performer only), a degree-matched random graph, the
  training-fold co-expression graph, and a STRING PPI graph. It is the one
  question this reimplementation can answer cheaply that the paper leaves
  open.
- Gene-embedding enrichment → queued candidate 3 (gene-relationship
  recall). Avoid co-expression-derived references there: the graph builds
  co-expression in, so the score would be circular.
- Not adopted, because the data are not TCGA bulk: DiSignAtlas, the
  cardiovascular-disease and GTEx tissue tasks, GDSC, DepMap, LINCS. The
  imputed-biomarker case studies are not evaluations.
