# PLIER and MultiPLIER — what the papers did

PLIER is the method; MultiPLIER is PLIER trained on a large compendium and
then used to project other datasets. One reimplementation covers both.

**PLIER.** Mao, Zaslavsky, Hartmann, Sealfon, Chikina. *Pathway-level
information extractor (PLIER) for gene expression data*. Nature Methods
16(7):607–610, 2019 (published 27 June 2019).
[doi 10.1038/s41592-019-0456-1](https://doi.org/10.1038/s41592-019-0456-1),
[PMC7262669](https://pmc.ncbi.nlm.nih.gov/articles/PMC7262669/) (author
manuscript), preprint [bioRxiv 10.1101/116061](https://doi.org/10.1101/116061).
Code: [wgmao/PLIER](https://github.com/wgmao/PLIER) (R; mirror
[chikinalab/PLIER](https://github.com/chikinalab/PLIER)).

**MultiPLIER.** Taroni, Grayson, Hu, Eddy, Kretzler, Merkel, Greene.
*MultiPLIER: a transfer learning framework for transcriptomics reveals
systemic features of rare disease*. Cell Systems 8(5):380–394.e4, 2019.
[doi 10.1016/j.cels.2019.04.003](https://doi.org/10.1016/j.cels.2019.04.003),
[PMC6538307](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538307/), preprint
[bioRxiv 10.1101/395947](https://doi.org/10.1101/395947) (its title ends
"rare autoimmune disease"). Code:
[greenelab/multi-plier](https://github.com/greenelab/multi-plier) (analysis)
and [greenelab/rheum-plier-data](https://github.com/greenelab/rheum-plier-data)
(recount2 processing and training); data and models on figshare,
[10.6084/m9.figshare.6982919.v2](https://doi.org/10.6084/m9.figshare.6982919.v2).

Read:
- The PLIER author manuscript (PMC7262669) in full: main text, Methods and
  the table and figure captions. The Supplementary Notes, which cover
  robustness to λ, single-cell data and cross-study use, were not read.
  Neither was the typeset journal version or the bioRxiv preprint.
- The MultiPLIER accepted manuscript (PMC6538307) in full, as text through
  NCBI's BioC API. Its equations did not survive extraction, so the
  transfer formula below comes from the code, which the text describes in
  words. bioRxiv v2 was not read (the fetch was rate-limited).
- Code: `R/Allfuncs.R` at wgmao/PLIER HEAD (DESCRIPTION 0.99.0, dated
  2019-12-31). MultiPLIER pinned an older commit, `a2d4a2`; I did not diff
  the two, so any changed defaults are unverified. Also read
  multi-plier's `util/plier_util.R` and rheum-plier-data's `recount2/1-3`
  scripts.
- Follow-up: *PLIERv2: bigger, better and faster*, bioRxiv
  [10.1101/2025.06.05.658122](https://doi.org/10.1101/2025.06.05.658122)
  ([pivlab/plier2](https://github.com/pivlab/plier2), R). Read only for its
  PLIERv1 runtimes.

The claims checked out: PLIER factorizes expression while pushing some
latent variables toward gene sets (70% of them by default), and
MultiPLIER trains on recount2 and projects small cohorts into that space.
One addition: MultiPLIER is not a new model. "No alterations to the
underlying PLIER framework were made" (STAR Methods); it is PLIER plus the
projection formula.

## Model

Notation (paper, §Problem setting): Y is genes × samples, z-scored per
gene. C ∈ {0,1}^(genes × gene sets) is the prior. Z is genes × k loadings,
B is k × samples latent variables (LVs), and U is gene sets × k.

Objective, as the paper states it:

    min ‖Y − ZB‖²_F + λ1‖Z − CU‖²_F + λ2‖B‖²_F + λ3‖U‖_L1   s.t. U ≥ 0, Z ≥ 0

The first term is reconstruction. The second keeps each loading column
close to a sparse, non-negative combination of gene sets. The third is a
ridge penalty on B, and the fourth keeps U sparse (column-wise). Non-
negativity makes the genes of a set co-vary positively within an LV.

Algorithm (block coordinate descent; code `PLIER()`):
- **Init**: randomized SVD of Y (`rsvd`, q = 3; rank max(200, n/4) when
  n > 500, else full SVD). Then B = (VD)ᵀ[1:k] and
  Z = YBᵀ(BBᵀ + λ1 I)⁻¹. A column that is entirely ≤ 0 is sign-flipped,
  then Z is clipped at 0.
- **Loop**, up to `max.iter = 350`:
  - Z ← (YBᵀ + λ1 CU)(BBᵀ + λ1 I)⁻¹, then negatives are set to 0.
    Non-negativity is a projection, not a constrained solve. The paper
    prints the inverse as (Bᵀ + λ1 I)⁻¹, a typo.
  - B ← (ZᵀZ + λ2 I)⁻¹ZᵀY.
  - U: for each LV, a non-negative glmnet fit of Z[:, j] on columns of C
    (`alpha = 0.9`, so elastic net rather than pure L1; `lower.limits = 0`,
    `intercept = TRUE`, `standardize = FALSE`). Only the top
    `maxPath = 10` gene sets per LV are candidates, ranked by a ridge
    regression of Z on C: pseudo-inverse of CᵀC with α = 5.
    `pathwaySelection = "complete"` by default, which pools every LV's
    candidates.
- **The prior starts at iteration 20.** Before that U = 0, which makes
  it a non-negative ridge factorization. PLIERv2 describes this as "the
  first 30 iterations"; the code says 20.
- **Stop** when ‖ΔB‖²/‖B‖² < `tol = 1e-6` (the paper says 5 × 10⁻⁶), or
  when that relative change stops falling.

Hyperparameters, which cannot be cross-validated on reconstruction
because λ1 = 0 always wins (paper, §Optimization constants):
- **k**: `num.pc` estimates the number of significant PCs, by an elbow on
  the smoothed second difference of the singular values, or optionally by
  permutation (Leek 2007). The package multiplies that by 2 and caps it at
  0.9·n. MultiPLIER used 1.3 ×, following an older vignette. The paper
  says LVs "persist when k is increased". For DGN, k was tuned to
  maximize the number of significant LVs.
- **λ2 = d_k**, the k-th singular value of Y, and **λ1 = d_k / 2**. The
  factor of 2 accounts for thresholding Z at zero.
- **λ3**: every 20 iterations, the code takes glmnet's fixed λ path
  (exp(−4)…exp(−12), steps of 0.125) and keeps the value at which the share
  of LVs with at least one gene set is closest to `frac = 0.7`. The paper
  says "binary search"; that is what the experimental `PLIERsparse` does,
  not `PLIER()`.
- **Genes**: `allGenes = FALSE` by default, so only genes present in the
  prior matrix are modelled, not the whole transcriptome. Gene sets with
  fewer than 10 genes (`minGenes`) are zeroed.
- **Gene-holdout cross-validation** (`doCrossval = TRUE`): a random fifth
  of each gene set's member genes is removed from C before training.
  Afterwards, each positive entry of U is scored by how well that Z column
  ranks the held-out members above non-members: Wilcoxon AUC, p-value,
  then BH FDR. The paper's "annotated with high confidence" means
  AUC > 0.7 and FDR < 0.05. This only annotates LVs; it does not change B.

Output: B (the embedding), Z, U, the U-AUC/p/FDR tables, λ1, λ2, λ3, and
the residual.

## Transfer (MultiPLIER)

New samples are projected with the trained Z and λ2 (`GetNewDataB`):

    B_new = (ZᵀZ + λ2 I)⁻¹ Zᵀ Y_new

This is the same map as PLIER's final B update, so projected and
training samples are embedded the same way. The PLIER package's
`projectPLIER` uses `Zproject`, which only `simpleDecomp` returns, so it
does not apply to `PLIER()` output.

Y_new is z-scored per gene **within the target dataset** (`rowNorm` on
the new data). Genes the model has but the target lacks are set to 0,
which is the mean after z-scoring. There is no cross-platform
adjustment: the RNA-seq-trained model is applied to microarrays. The
paper's limitations section concedes "minimal effort to adjust the data
distributions between platforms".

## Training data

**PLIER**:
- A new validation set: 35 whole-blood samples with RNA-seq and CyTOF
  (GSE130824). STAR/featureCounts to RPKM; separately, quantile-normalized
  log counts; batch-corrected.
- DGN whole-blood RNA-seq, 922 individuals (restricted, via NIMH), already
  "trans"-normalized.
- NESDA microarrays (dbGaP phs000486.v1) for replication.
- Priors: ship with the package as binary matrices keyed by HGNC symbol.
  - `bloodCellMarkersIRISDMAP`: 61 sets, derived by the authors from IRIS
    and DMAP.
  - `svmMarkers`: 22 sets, CIBERSORT LM22.
  - `canonicalPathways`: 545 sets on 6,023 genes, MSigDB C2:CP. By name:
    252 REACTOME, 122 KEGG, 116 PID, 25 BIOCARTA, 30 other.
  - `oncogenicPathways`: 189 sets, MSigDB C6.
  - `chemgenPathways`: 3,395 sets, MSigDB C2:CGP.
  - `immunePathways`: 1,910 sets, C7.
  - `xCell`: 489 sets.
  - MSigDB release not stated.

**MultiPLIER**:
- recount2 via the `recount` package v1.4.6, `subset = "sra"` only. TCGA
  and GTEx were not included.
- Samples without metadata were dropped, leaving **37,032 samples**.
  Gene-level RPKM, experiments concatenated with no further processing,
  genes z-scored. The prep scripts apply no log transform.
- Ensembl IDs mapped to HGNC symbols with biomaRt.
- Prior: `bloodCellMarkersIRISDMAP` + `svmMarkers` + `canonicalPathways`,
  628 sets on 6,847 genes (the union, before intersecting with recount2).
- k = round(1.3 × `num.pc`); λ1 and λ2 left at their defaults. Result:
  **987 LVs**.

Compute: neither paper reports it. PLIERv2 reports that PLIERv1 took
~42 h on recount2 ("~30K samples, ~30K genes") and ~26 h on GTEx v8
(17,382 samples × ~56K genes), on a 32-core Xeon with 256 GB of RAM.
PLIERv2 took 6.0 h and 0.64 h.

## Evaluations as published

**PLIER** (main text):
1. **Blood cell proportions vs CyTOF** (Fig. 1), 35 samples. Priors: "60
   cell-type markers and 555 canonical pathways", where the package has
   61 and 545. Default parameters, k = 30.
   - 14 LVs were annotated with confidence; 8 matched CyTOF cell types.
   - Mean Spearman 0.71, range 0.58–0.78.
   - Beat NMF and sparse PCA (both k = 30; best-correlated component per
     cell type). Beat both reference-based methods, NNLS and CIBERSORT,
     on 4 of the 8 cell types.
2. **DGN trans-eQTL** (Tables 1–2, Fig. 2). Prior: 4,445 sets (C2:CP +
   C2:CGP + cell markers + cytokine signatures); k tuned.
   - 86 LVs with a gene set at FDR < 0.05, covering 318 sets; 29 LVs
     "unambiguously" biological.
   - 12 LVs had genome-wide significant SNP associations.
   - Replication in NESDA: at gene-level FDR 0.2, π1 ≈ 0.6 for
     pathway-centric eQTLs against ≈ 0.2 for gene-centric ones.
   - Example: ARHGEF3 rs1354034 acts on two megakaryocyte/platelet LVs in
     opposite directions.
3. Single-cell and cross-study analyses are in Supplementary Note 1 (not
   read).

**MultiPLIER**:
1. **SLE whole-blood compendium**: 7 microarray datasets, n = 1,640, with
   its own PLIER model.
   - PCA of pathway-associated LVs (AUC > 0.75) shows less dataset
     separation than all LVs or non-pathway LVs (Fig. 2, S1).
   - Neutrophil counts (E-GEOD-65391): SLE-WB LV87 R² 0.29; MultiPLIER
     LV603 R² 0.36; LV603 vs MCPcounter neutrophil R² 0.83 (Fig. 3).
2. **Held-out gene sets**: the recount2 model covers **0.767** of the C6
   oncogenic sets, which were never in its prior (Results; notebook 27).
   On the input prior itself: 0.419 of gene sets covered, 0.20 of LVs
   with a set, FDR < 0.05 (notebook 02, not in the paper).
3. **Sample size and training context** (Fig. 4).
   - Setup: random recount2 subsets of 500–32,000 and MetaSRA contexts
     (blood n = 3,862, cancer 8,807, tissue 12,396, cell line, other), 5
     models each.
   - Larger training sets gave larger k and higher pathway coverage, but a
     smaller share of LVs with a pathway, blood excepted.
   - The cancer-context model captured no NK-cell sets.
4. **Pathway separation** (Fig. 5A): IFN type I vs II, neutrophil vs
   monocyte/macrophage, G1 vs G2. The full model separates all three; the
   SLE-WB model found only type I IFN.
5. **Agreement with a dataset-specific model** (Fig. 5B). NARES nasal
   brushings, n = 79, whose own PLIER model has 34 LVs. Latent variables
   matched by best loading correlation have positively correlated B, more
   so for pathway-associated ones. LV603 vs MCPcounter neutrophil R² 0.9
   (Fig. 3D).
6. **AAV** (vasculitis) differential LVs, limma, BH.
   - GPA PBMCs: LV599 at FDR 1.2e-8; ANCA antigens are among its top
     genes.
   - 22 LVs are differential in all three tissues (nasal, glomeruli,
     PBMC). LV10 (M0 macrophage) and LV937 (HIF-1α) are up in active or
     severe disease (Fig. 6, S8).
7. **Medulloblastoma**: 617 of 987 LVs differ by subgroup (FDR < 0.05) in
   both cohorts; translation-related LVs are shown (Fig. 7).
8. **λ2 = 0 ablation** (notebook 39): Z is no longer sparse and pathway
   coverage drops.

Nothing is scored by a supervised probe, and no downstream task is
compared against PCA at equal dimension. Evaluations are correlations and
differential tests on single chosen LVs.

## For reimp

**Method-defining**, kept:
- The objective and its alternating solver.
- SVD initialization.
- Non-negative Z by clipping.
- The non-negative elastic net on U (α = 0.9), with `maxPath` candidate
  preselection.
- λ1 = d_k/2 and λ2 = d_k.
- λ3 tuned to `frac = 0.7`.
- The prior entering after 20 iterations.
- k from `num.pc` × 2.
- Per-gene z-scoring.
- Gene-holdout cross-validation, for the U annotations.
- The embedding: B = (ZᵀZ + λ2 I)⁻¹Zᵀy, applied the same way to
  training, validation and test samples. This is MultiPLIER's projection
  formula.
- A prior of curated gene sets that includes cell-type markers, chosen
  as below.

*2026-09-13: rewritten from the paper.* For licensing reasons (reimp is
to be published under a permissive licence), the solver was rewritten from
the PLIER and MultiPLIER papers alone. It no longer follows the R package;
the earlier numpy port of it was deleted unread. Kept from the papers: the
objective, block coordinate descent from the SVD, Z clipped at 0, λ1 =
d_k/2 and λ2 = d_k, λ3 set for `frac` = 0.7, k = 2 × a count of principal
components, 5e-6 stopping, per-gene z-scoring, the held-out fifth with AUC
> 0.7 and FDR < 0.05, and the projection. Several items in the list above
came from the package's code, not the papers, and are now reimp's own
decisions (plier/README.md, "Our decisions"):
- **U**: a pure non-negative lasso over every gene set, as the paper
  writes it. The package's elastic net (α = 0.9) and its `maxPath`
  candidate preselection are gone.
- **When the prior enters**: once the U = 0 factorization stops by the
  stopping rule, not at iteration 20.
- **k**: the chord elbow of the training singular values, × 2, not
  `num.pc`. It gives k = 492 on fold 0. A Gavish–Donoho threshold is the
  option standing in for significance.
- **λ3**: the closed-form value that meets `frac`, re-set every 10
  iterations, not a search along glmnet's path.
- **Gene sets** with fewer than 5 genes are left out of C, not 10.
- **Annotation negatives** are the genes not in the set, as the paper
  says, not the genes in none of the LV's sets.
- **Stopping**: ‖ΔB‖/‖B‖ (not squared) below 5e-6, or no new low in 20
  iterations.

**Incidental**, standardized:
- **Input transform**: z-score log1p library-normalized unstranded counts
  on the 19,944 protein-coding genes (the PCA baseline's input). Their
  z-scored RPKM without a log is a closer-to-paper variant through
  `fpkm_unstranded`.
- **Gene universe**: `allGenes = TRUE`. Every protein-coding gene is
  modelled, and genes in no set get empty rows of C. The package default
  would keep only prior genes: ~4.9k with the prior below, ~6.8k with
  MultiPLIER's. That would weaken invertibility for reasons unrelated to
  the method. The default can be a variant.
- **Genes to symbols**: map the prior's HGNC symbols to GENCODE v36
  `gene_name` and report the unmapped count (unmeasured).
- **Training data**: one model per fold on that fold's training patients
  (~8,300 samples), instead of recount2.
- **MultiPLIER's "train on a big external compendium"** is incidental to
  our TCGA-only setup. Per-fold PLIER plus projection of validation and
  test samples is MultiPLIER's mechanics exactly (source-trained Z,
  λ2-ridge projection of unseen samples). It does not test MultiPLIER's
  thesis: that breadth of the source compendium buys transfer to other
  tissues, platforms and diseases. Our test patients come from the same
  33 projects as training.
- **The released recount2 model**: projecting TCGA into it would be a
  genuine MultiPLIER run with no TCGA training, since recount2's SRA
  subset excludes TCGA. But it breaks the TCGA-only rule, as Gene2Vec did
  for BulkRNABert, and its prior contains Reactome and PID. It could be
  reported as a labelled reference point at most.
- **k = 256**, reimp's common embedding size, so PLIER sits beside PCA-256
  and the other models at equal dimension. The package rule, computed on
  training samples per fold, is a variant (`--model.k null`; k = 492 on
  fold 0). (Changed 2026-09-13 from the rule as the default: reimp
  compares methods at a common size, not paper by paper.)

**Leakage**, following rule 3:
- **Fit on training samples only**: gene means and SDs, the SVD, `num.pc`,
  λ1/λ2 (from d_k), λ3, and dropping zero-variance genes.
- **Validation and test** samples are z-scored with training means and
  SDs, then projected. MultiPLIER instead z-scores each target dataset on
  itself. That is a cohort statistic, and on our splits it would recentre
  every test fold, so it must not be copied.
  *2026-09-13:* each value is first clipped to its gene's training range.
  Genes nearly constant over training gave held-out z-scores up to 1,398
  on fold 0, against at most 91 in training (README, "Ours, for reimp").
- **Exclude the C2:CGP prior** (`chemgenPathways`, which PLIER's DGN
  analysis used). It contains TCGA-derived signatures:
  `TCGA_GLIOBLASTOMA_{COPY_NUMBER_UP,DN,MUTATED}`,
  `VERHAAK_GLIOBLASTOMA_{CLASSICAL,MESENCHYMAL,NEURAL,PRONEURAL}` (from
  TCGA GBM) and `VERHAAK_AML_*` (TCGA LAML). These are gene sets defined
  on TCGA patients, test patients included, and they encode subtypes
  close to our glioma task.

**Pathway-probe circularity.** MultiPLIER's default prior contains two of
the collections the pathway probe scores. I measured best-match Jaccard
between each probe set and every prior set. MSigDB's download server
returned 503, so the probe side used Enrichr copies, not our MSigDB
2026.1 sets; exact overlap with those, and anything about Cancer Cell
Atlas, is unverified.

| probe collection | vs full canonicalPathways | vs canonical minus REACTOME/PID | vs cell markers | vs C6 prior |
|---|---|---|---|---|
| PID (NCI-Nature 2016, 210 sets) | median 0.79; 49% ≥ 0.8 | 0.14; none ≥ 0.5 | 0.02 | 0.02 |
| Reactome (2022, 1,814 sets) | 0.16; 14% ≥ 0.5 | 0.09; 4% ≥ 0.5 | 0.01 | 0.02 |
| Hallmark (50 sets) | 0.11; none ≥ 0.5 | 0.08; none ≥ 0.5 | 0.02 | 0.04 |
| C6 oncogenic (190 sets) | 0.02 | 0.02 | 0.02 | **0.96; all ≥ 0.8** |

Notes on the table:
- PLIER's canonicalPathways holds MSigDB's own Reactome sets. The low
  Reactome row reflects Enrichr's newer, finer Reactome release, not
  distinct content.
- Gene coverage of the recommended prior (canonical minus REACTOME/PID,
  plus cell markers): its member genes cover 43% of Hallmark genes, 30% of
  Reactome's and 53% of PID's. Some gene overlap is unavoidable for any
  pathway prior.

**Decision (2026-09-13), replacing the recommendation below.** The prior is
whole MSigDB 2026.1 collections, fetched by name through
`reimp_shared.genesets`, and nothing is left out for overlap with the
probes. PLIER is a baseline in reimp, not a model to pursue, so the
circularity is reported beside its pathway scores rather than engineered
away, and whole collections keep the code free of per-set special cases.
The default is MultiPLIER's recipe within MSigDB's licences: C2:CP's
REACTOME, PID, WIKIPATHWAYS and KEGG_MEDICUS, plus C8 cell-type
signatures in place of IRIS/DMAP and LM22. Left out for their licences:
KEGG_LEGACY and BioCarta (KEGG and BioCarta license them to the Broad
Institute alone), LM22 (free to academic users on registration) and
IRIS/DMAP (journal supplements with no stated licence). C2:CGP stays out
for the leak above. Any GMT file can be added to the prior by path.

Recommendation (superseded by the decision above):
- **Prior**: IRIS/DMAP (61) + LM22 (22) + canonicalPathways without
  REACTOME and PID (122 KEGG, 25 BioCarta, 30 other). That is 260 sets on
  4,932 member genes; 257 have ≥ 10 genes.
- **Exclude from the prior** every collection we probe: Reactome, PID, C6
  `oncogenicPathways`, Hallmark, and Cancer Cell Atlas (C4). The pathway
  probe then scores held-out collections, as MultiPLIER held out C6. There
  are no near-duplicate sets, but some gene overlap remains; say so beside
  the scores.
- **Circular variant**: MultiPLIER's exact prior could be a
  closer-to-paper run, but its PID and Reactome pathway-probe scores must
  be marked circular (and C6's, if `oncogenicPathways` is added).

**Feasibility and compute**:
- The costs are small. In numpy on this machine (9,000 genes × 8,000
  samples):
  - Randomized SVD at PLIER's default rank max(200, n/4) ≈ 2,000: 6.6 s.
  - One Z + B iteration at k = 300: 0.12 s.
  - With all 19,944 genes: ~2.2 × longer; Y is 1.3 GB in float64.
- The unmeasured part is the U step: k small non-negative elastic-net fits
  per iteration after iteration 20, plus a 65-value λ path every 20
  iterations. Estimate: minutes to tens of minutes per fold on a CPU, no
  GPU. recount2's ~42 h came from k ≈ 1,000 and 37k samples.
- **No maintained Python port.** PyPI's `plier` is an unrelated PLY
  utility.
- [IDSIA/FPLIER](https://github.com/IDSIA/FPLIER) (created May 2026, no
  licence file) embeds a numpy/scikit-learn translation of the PLIER loop
  in a Flower federated app: `solveU` via sklearn `ElasticNet`, `num_pc`,
  `pinv_ridge`, `cross_val`/`AUC`. It is a cross-check, not a dependency.
- Plan: write our own, about 300 lines of numpy plus
  `sklearn.linear_model.ElasticNet(alpha=λ3, l1_ratio=0.9, positive=True)`.
  Its objective matches glmnet's, but glmnet standardizes y internally for
  the Gaussian family, so check the λ3 scale.
- Test parity against R PLIER (R is installed locally) on the package's
  bundled `dataWholeBlood`: compare k, λ1/λ2, U support and B up to
  column permutation.

**Expectations.** The embedding is a fixed linear map of z-scored genes,
so at equal k PLIER should not beat PCA on invertibility, which is PCA's
own objective. Cancer-type probes will likely be near PCA. Any gain
would be in denoising: within-project pathway and survival probes, and
lower confounder scores on the pathway-associated LVs. That is untested.
Its value to reimp is as the only model that uses prior knowledge, and a
cheap one: it asks directly whether the prior buys anything.

**Evaluation ideas to adapt:**
- **No-prior ablation**, the key one: the same solver with U = 0 (λ1‖Z‖²
  instead of λ1‖Z − CU‖²; PLIERv2's "PLIERbase") and the same k. The
  difference between the two isolates the prior.
- **Held-out collection coverage** (MultiPLIER's C6 test): the share of a
  collection's sets that some loading column recovers at FDR < 0.05, by
  Wilcoxon AUC. It works for any model with per-gene loadings — PCA
  loadings, PLIER's Z, a linear decoder. It fits EVALS candidate 3 (gene-
  level relationship recall), and Hallmark, Reactome, PID and C6 are
  already on hand.
- **Confounders split by annotation** (SLE-WB Fig. 2): run
  `confounder_probe` on pathway-annotated LVs and on the rest separately.
  PLIER's claim is that technical signal collects in unannotated LVs.
- **Sample-size curve** (MultiPLIER Fig. 4): train per fold on
  500…8,000 training samples and probe the same test patients. Cheap for
  PLIER and generic to any model. It is the closest TCGA-only analogue of
  MultiPLIER's argument.
- **Leave-projects-out transfer**: train on a subset of projects and
  project held-out projects. A within-TCGA stand-in for cross-tissue
  transfer (like BulkRNABert's Fig. 5).
- **Not adopted**:
  - CyTOF and cell-count correlations: TCGA has no such labels. An
    immune-deconvolution resource would be external.
  - trans-eQTLs: no germline genotypes.
  - Pathway separation (IFN I/II, G1/G2): specific to annotated LVs.
