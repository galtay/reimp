# Tybalt — what the paper did

Way, Greene. *Extracting a biologically relevant latent space from cancer
transcriptomes with variational autoencoders*. Pacific Symposium on
Biocomputing 23:80–91 (2018).
[doi 10.1142/9789813235533_0008](https://doi.org/10.1142/9789813235533_0008),
[bioRxiv 10.1101/174474](https://doi.org/10.1101/174474), PMID 29218871,
PMC5728678. Code, processed data and trained weights:
[greenelab/tybalt](https://github.com/greenelab/tybalt) (BSD-3-Clause;
Keras 2 on TensorFlow 1.x; last commit Jan 2019).

Follow-up read for context: Way, Zietz, Rubinetti, Himmelstein, Greene.
*Compressing gene expression data using multiple latent space
dimensionalities learns complementary biological representations* ("BioBombe").
Genome Biology 21:109 (2020).
[doi 10.1186/s13059-020-02021-3](https://doi.org/10.1186/s13059-020-02021-3),
PMC7212571, [greenelab/BioBombe](https://github.com/greenelab/BioBombe).

Read: bioRxiv v2 through a web-page text extraction (the PDF was blocked, and
no open-access PSB full text was reachable), so section and figure numbers
below are approximate. Also read: the repo's README, `parameter_sweep.md`,
`process_data.ipynb`, `tybalt_vae.ipynb`, `scripts/vae_pancancer.py`,
`tybalt/models.py`, and issue #99. BioBombe was read in full from the Europe
PMC XML, plus the repo's `1.initial-k-sweep/config`. Where the paper and the
code disagree, the code is what produced the released model.

## Model

- A VAE with a single encoding layer: 5,000 genes → 100 latent dimensions →
  5,000 genes. It has about 1.5M parameters. The mean and the log-variance
  each get their own `Dense(100)` → BatchNorm → **ReLU**, so the posterior
  means are non-negative. The log-variances are also non-negative, which makes
  every posterior variance at least 1. This is a quirk of the code, not
  something the paper discusses. Sampling is `z = μ + exp(logvar / 2) · ε`.
- Decoder: one `Dense(5000, sigmoid)`. Weights are Glorot-uniform.
- Objective: `5000 · BCE(x, x̂) + β · KL`, with BCE averaged over genes, so
  in effect summed. It relies on inputs being min-max scaled to [0, 1].
- KL warm-up (Sønderby et al. 2016): β starts at 0 and is raised by κ at the
  end of each epoch until it reaches 1. The released notebook uses κ = 1,
  which means β = 0 for the first epoch and a full VAE from then on. The sweep
  over κ ∈ {0.01, 0.05, 0.1, 1} made "little to no difference" for this
  architecture. For the two-hidden-layer variants, κ < 1 was worse.
- Adam, lr 0.0005, batch 50. The sweep chose 100 epochs by validation loss,
  and the paper text says 100. The README and notebook train the released
  model for **50 epochs**, because "training did not improve much between 50
  and 100".
- Sweep grid: lr {0.0005–0.0025}, batch {50, 100, 128, 200}, epochs {10, 25,
  50, 100}, κ as above. They also tried two-hidden-layer encoders
  (5000→100→100 and 5000→300→100), which did "not improve performance as much
  as initially thought". Compute: the paper reports 8 GTX 1080 Ti GPUs on
  Penn's PMACS cluster for the sweep. Time per model is not stated.
- Sample embedding: `encoder = Model(input, z_mean_encoded)`, i.e. the ReLU'd
  posterior mean, 100 dimensions.
- Also in the repo but not in the paper: an ADAGE (tied-weight denoising AE)
  comparison, and a conditional VAE class (`tybalt/models.py`, `label_dim`).

## Training data

- UCSC Xena TCGA Pan-Cancer RNA-seq (`HiSeqV2`), downloaded 8 March 2016 and
  archived as Zenodo 56735. The paper describes it as "log2(FPKM + 1)
  transformed RSEM values". There are 10,459 samples: 9,732 tumours and 727
  adjacent normals, across 33 cancer types. The Xena file's own unit
  definition was not checked.
- Genes: the 5,000 genes with the highest absolute deviation. The paper says
  *median* absolute deviation. The code uses pandas `DataFrame.mad()`, which
  is the *mean* absolute deviation (issue #99). The author estimated that
  about 85% of the genes are the same either way.
- Scaling: `MinMaxScaler` per gene to [0, 1].
- **Every fitted statistic is fit on all 10,459 samples**: the gene ranking
  and the per-gene min and max, before any split.
- Split: a random 10% of *samples* is held out for the validation loss only.
  The embeddings are then written for all samples, and nothing downstream is
  evaluated on held-out samples.

## Evaluations as published

Almost entirely qualitative. The paper states: "We do not compare our
approach to alternate dimensionality reduction algorithms". There is no
PCA/ICA/NMF comparison and no downstream prediction task.

1. **Latent layout**: a t-SNE of the 100-dimensional encodings against a
   t-SNE of the raw 5,000 genes, coloured by cancer type (Fig. 2). Also
   distributions of node activations, which are right-skewed with some
   bimodal nodes.
2. **Single features**: encoding 82 "nearly perfectly" separates sex. Its
   high-weight genes are XIST and TSIX (positive) and Y-chromosome genes such
   as EIF1AY, UTY and KDM5D (negative). Encodings 53 and 66 pick out SKCM.
   Encoding 66 is enriched for lipid/ethanol-metabolism GO terms (e.g.
   ethanol oxidation, adj. p = 0.04); encoding 53 has no enriched GO term.
3. **HGSC subtypes** (TCGA OV; subtype labels are in the repo as
   `ov_subtype_info.tsv`): encodings 87 (mesenchymal, collagen/ECM), 56
   (immunoreactive, immune response), 79 and 38 (proliferative vs
   differentiated), and 77. Also "latent arithmetic", which subtracts subtype
   mean vectors to find the features that separate two subtypes.
4. The released model's reconstruction loss is shown only as a curve
   (Fig. 1D); no number is reported.

### BioBombe, the follow-up that does the comparison

- Data: the batch-corrected TCGA PanCanAtlas RSEM matrix, 11,060 samples ×
  16,148 protein-coding genes. Also GTEx v7 (TPM, 11,688 samples) and TARGET
  (734 samples). Each dataset is cut to its top 8,000 genes by median
  absolute deviation and min-max scaled to [0, 1] per gene. The scaling is
  applied "independently for the testing and training partitions", so test
  data is scaled with its own minimum and maximum.
- Split: 90/10, balanced by cancer or tissue type. Compared: PCA, ICA and NMF
  (scikit-learn), plus DAE and VAE (the Tybalt code). 28 values of k from 2
  to 200, 5 seeds each, on real and gene-wise permuted data: 4,200 models.
  VAE hyperparameters were swept per k over lr {0.0005–0.0025}, batch {50,
  100, 150}, epochs {50, 100}, κ {0, 0.5, 1}, at k ∈ {5, 25, 50, 75, 100,
  125}.
- Findings relevant to reimp:
  - "All the compression algorithms had similar reconstruction costs", with
    the most variability at low k.
  - PCA, ICA and NMF are highly stable across seeds (SVCCA). The VAE is
    "largely stable, with some decay in higher latent dimensionalities". The
    DAE is unstable. The VAE is closer to PCA, ICA and NMF than the DAE is,
    especially at low k.
  - With elastic-net logistic regression on the features, all algorithms
    predict cancer type with high AUPR at small k. Mutation status (TP53,
    PTEN, PIK3CA, KRAS; TTN as a negative control) needs larger k.
  - Ensembles of seeds or algorithms reach the signal at smaller k. A
    classifier on all 30,850 features is "comparable to" one on raw genes for
    TP53 (317 nonzero weights).
  - Conclusion: "There is no single best latent dimensionality or compression
    algorithm". AUPR values are in Fig. 7 and Additional files and are not
    transcribed here.

## For reimp

**Method-defining**, kept:

- A VAE with one dense encoding layer to a latent of k = 100, and BatchNorm
  and ReLU on both the mean and log-variance heads, as in the code that
  produced every published feature. A textbook linear-head VAE is a one-flag
  variant.
- The sigmoid decoder with a per-gene BCE reconstruction loss.
- KL warm-up (κ = 1 by default; the sweep found it irrelevant).
- Adam at lr 5e-4, batch 50, 50 epochs.
- Input: top-5,000 genes by absolute deviation, min-max scaled per gene.
- Embedding: the posterior mean.
- A log-transformed FPKM input (`fpkm_unstranded` + `log1p`) is the closest
  match. Because each gene is then min-max scaled, the log base does not
  matter.

**Incidental**, standardized:

- Training on the fold's training patients only (about 8,300 samples; the
  paper had about 9,400 in its 90%). Tumours and normals are both kept, as in
  the paper.
- The gene ranking, per-gene minimum and maximum fit on training samples.
  Val and test values are clipped to [0, 1] after scaling, since BCE and the
  sigmoid output cannot represent values outside it.
- The ranking uses *median* absolute deviation, as the paper and BioBombe
  describe it, over our default 19,944 protein-coding genes.
- The fold's val patients replace the random 10% sample hold-out, for loss
  monitoring and early stopping.
- A PyTorch/Lightning port replaces Keras/TF1. The released weights are not
  used: they were fit on all TCGA samples, including every test patient.

**Leakage in the original**: gene selection and scaling were fit on all
samples, and the hold-out was split by sample, not by patient. Neither
matters for the paper, which has no held-out evaluation. Both matter for us.

**Compute**: tiny. About 1.5M parameters, about 8,300 × 5,000 inputs, and
about 8,300 optimizer steps per fold. My estimate, not measured: minutes per
fold on a laptop CPU, and seconds per epoch on a GPU. A k-sweep with several
seeds is affordable.

**Evaluation ideas to adapt:**

- **The matched-k comparison with PCA** is the main reason to include
  Tybalt. Train Tybalt and PCA at the same k values (e.g. 16, 32, 64, 100,
  128, 256), on the same genes and scaling, and run every shared probe as a
  function of k. BioBombe predicts the two will look similar on cancer type
  and reconstruction, and that differences will show up at larger k and on
  subtler labels. It also matters that the shared PCA baseline is fit on
  lognorm counts at 256 components: Tybalt's 5,000-gene min-max input is a
  different input, so both inputs should be probed.
- **Mutation status** (TP53, PTEN, PIK3CA, KRAS, with TTN as a negative
  control), the BioBombe task. This is EVALS candidate 1. TTN is a useful
  built-in negative control for any genomic-label probe.
- **Gene-wise permuted training data** as a floor. BioBombe trains every
  algorithm on data where each gene is shuffled independently, which
  destroys correlation structure. A generic "what a model learns from no
  structure" baseline for any learner.
- **Stability across seeds** (SVCCA, or CCA between two seeds' embeddings of
  the same fold's test samples). Cheap for Tybalt; possibly generic later.
- **Molecular subtypes**: OV (TCGA 2011 HGSC subtypes; Tybalt ships them)
  alongside BRCA PAM50, per EVALS candidate 2. Needs an external label file.
- Sex separation by a single latent unit: a sanity check only. Any linear
  probe gets sex trivially from XIST and the Y-chromosome genes, and those
  genes are in the default gene set.
