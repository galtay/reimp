# TxFM — what the paper did

Kenyon-Dean, Bendidi, Selega, Sorokin, Bertinetto, Errington, Kraus,
Donnella. *Effective Biological Representation Learning by Masking Gene
Expression*. ICLR 2026 Workshop on Foundation Models for Science.
[OpenReview NqZqClqtTK](https://openreview.net/forum?id=NqZqClqtTK).
Code: [recursionpharma/opentxfm](https://github.com/recursionpharma/opentxfm)
— "under construction"; as of 2026-04 it holds only the loss and activation.

Read: main text and appendix A. Appendices B–D (evaluation protocols,
ablation details, per-dataset results) not yet read in detail.

## Model

- Masked autoencoder. The encoder sees K = 2048 genes chosen uniformly
  from those observed in a sample; each token is `[E_i ; x_i]`, a learned
  (d-1)-dim gene embedding concatenated with the gene's value, plus a
  learned CLS token. No positional encoding.
- A 4-layer residual MLP decodes the CLS embedding to all G genes through
  a rectified tanh, log(L+1)·ReLU(tanh(z/4e)) (Eq. 1).
- Poisson loss e^x̂ − x̂·e^x on every gene, masked and visible (Eq. 2).
- Preprocessing: library-size normalization to L = 1e5, then log1p.
- Backbones (Table 6): S 6 blocks / 6 heads / 384-d (57M params), B 12 /
  12 / 768 (159M), L 24 / 16 / 1024 (403M). Stochastic depth 0.1, 0.1, 0.3.
- Training (§A.1): 200 epochs, global batch 1536, AdamW max lr 1e-3,
  betas (0.9, 0.999), eps 1e-6, weight decay 1 / (lr · total steps),
  one-cycle cosine schedule with 10% warmup, bf16, LayerScale. About
  1,000 H100 GPU-hours for B.
- No released weights.

## Pretraining data

DiverseRNA-1.4M (Table 1): 1,434,299 bulk and single-cell samples over
44,349 genes. Mostly single cell: K562 CRISPRi (502,080, curated to
distinct perturbation "phenoprints"), glioblastoma (504,929), MixSeq
(102,205), sci-Plex (99,300), gastric metaplasia (88,399), tumour
microenvironment (71,585), breast cancer (31,542). Bulk: **TCGA (23,733
samples, 19,594 genes)** and GTEx (10,526 samples, 36,695 genes).
Single-cell datasets keep genes expressed in ≥ 1,000 cells and cells
expressing ≥ 2,000 genes.

## Evaluations as published

None is on bulk data: TCGA and GTEx appear only in pretraining, and every
evaluation below is of single cells or perturbations. The paper therefore
says nothing direct about the quality of TxFM's bulk specimen embeddings,
which is what the shared probes measure here.

1. **Zero-shot perturbation representation** — the Bendidi et al. 2024
   benchmark ([Tx-Evaluation](https://github.com/valence-labs/Tx-Evaluation)),
   on CRISPRi screens in cell lines unseen in training: RPE1 (Replogle
   2022), HepG2 and Jurkat (Nadig 2024). Six scores: batch mixing (iLISI),
   linear probe and KNN on perturbation identity, perturbation
   consistency, recall of known biological relationships, and
   invertibility back to counts. Table 2 averages, HepG2 / Jurkat:
   TxFM-B 38.63 / 36.52; PCA 34.00 / 30.91; (Lib+Log)Norm 34.12 / 33.37;
   STATE-SE, the best other foundation model, 35.10 / 33.81. On
   invertibility alone the count baselines win: (Lib+Log)Norm 60.09
   against TxFM-B 44.40 on HepG2.
2. **Cell-type representation** — the Kedzierska et al. 2025 benchmark,
   five single-cell datasets. Clustering (ASW, NMI, ARI), batch
   integration (ASW over batch labels), and linear and 2-layer-MLP
   classification probes (accuracy, F1). Table 4, TxFM-B [DiverseRNA]
   vs PCA: ASW 54.4 vs 50.0, NMI 64.2 vs 55.3, ARI 46.0 vs 36.6, batch ASW
   85.7 vs 91.2, linear probe accuracy 84.7 vs 82.8, MLP accuracy 85.3 vs
   83.9.
3. **SSL fine-tuning on the evaluation data** (Table 3), perturbation
   score on RPE1 / HepG2 / Jurkat: TxFM-B 44.85 / 40.78 / 38.01; PCA 42.64
   / 38.63 / 34.89; scVI 37.65 / 35.63 / 32.62.
4. **Gene-gene relationship recall from model parameters** (Kraus et al.
   2024, Celik et al. 2024): cosine similarities between genes' encoder
   embeddings, and separately between their decoder weights, scored
   against CORUM, hu.MAP, Signor, StringDB and Reactome
   ("relationship recall @ 5-95", Figure 2). This is the gene-level score
   in the ablations (Table 5) and the epoch-wise analysis (Figure 2).
5. **Ablations** (Table 5), each scored by perturbation consistency and
   the two recalls: loss, preprocessing, K, masking temperature, decoder
   depth, output activation, backbone size, data curation. Library-size
   normalization, K = 2048, uniform masking, the rectified tanh and the
   Poisson loss each win or tie.
6. **Layer-wise analysis** (Figure 3): CLS representations from every
   encoder and decoder layer, scored by perturbation consistency; the last
   encoder block is best.

## For reimp

**Method-defining**, kept: the masked-autoencoder objective, the token
construction, CLS readout, the rectified tanh, the Poisson loss on all
genes, library-size normalization + log1p, uniform masking, K.

**Incidental**, standardized: the pretraining corpus becomes TCGA training
patients only; the gene universe becomes GENCODE v36 protein-coding genes
(19,944, so K = 2048 masks ~90%, the paper's average ratio); the batch size
and epoch count are scaled down to TCGA's ~9,200 training samples.

**Evaluation ideas to adapt**, with the published scores as rough context
only:

- Linear probe → already the shared `classification_probe`.
- KNN and clustering scores (NMI, ARI, ASW) on TCGA labels — a
  complement to linear probes that rewards compact geometry.
- Invertibility → already shared; note that here too simple count
  baselines are the ones to beat.
- Batch mixing → TCGA's technical confounders (sequencing plate and
  centre, parseable from aliquot barcodes): a representation should
  predict them poorly.
- Gene-gene relationship recall → a model-level evaluation of gene
  representations, for models that have them (TxFM's embedding table and
  decoder weights). Needs the external relationship databases.
