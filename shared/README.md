# reimp-shared

What every reimplementation shares: the dataset, the folds, the data
loaders, and the yardstick.

| module | |
|---|---|
| `hub` | the [dataset][ds]'s `samples`, `genes` and value configs as pandas / numpy |
| `splits` | patient-level 5-fold cross-validation: train / val / test per fold |
| `preprocess` | gene selection and value transforms |
| `data` | `load_expression` (numpy), `ExpressionDataset` (torch), `ExpressionDataModule` (Lightning) |
| `ranking` | per-sample gene rankings for rank-based models (`GeneRanker`: expression, z, tf-idf with count / smooth / entropy weights, cohort L2) |
| `tokens` | binned expression tokens for masked language models (`BinTokenizer`: one training-set maximum, bin 0 for exact zeros; BulkRNABert, MOJO) and BERT's 80/10/10 masking (`mask_tokens`) |
| `genesets` | gene sets: GMT files, and MSigDB 2026.1 collections fetched by name into `$REIMP_CACHE`, md5-pinned (`load_gene_sets`; PLIER's prior). KEGG_LEGACY and BioCarta are not offered: KEGG and BioCarta license them to the Broad alone |
| `labels` | evaluation labels: survival endpoints (per case) and ssGSEA pathway scores (per aliquot) from `tcga-patients-open`, technical covariates from the expression dataset |
| `eval` | the embeddings file format; classification, invertibility, pathway, survival, geometry and confounder probes; patient bootstrap and paired differences; PCA and top-variance-gene baselines |
| `foldcli` | `FoldCLI`, the LightningCLI every Lightning model's command line uses: fold k of a model runs in `<trainer.default_root_dir>/fold<k>/`, checkpoints in its `checkpoints/`, and a rerun replaces it |
| `testing` | a miniature copy of the datasets for offline tests; `scramble_held_out` and `assert_embedding_ignores_held_out`, which check a model never fits or embeds with held-out rows |

## Folds

Every model is evaluated by patient-level 5-fold cross-validation,
stratified by project. Two steps take a patient to its fold:

1. **Patient → bucket, 0–99, in 5 blocks of 20.** Each project's patients
   are ranked by a salted hash of their case ID (`case_submitter_id`); rank
   r of n goes to block ⌊5 · (r + ½) / n⌋, so every project fills the
   blocks evenly, to within one patient. The hash picks the bucket within
   the block: bucket = 20 · block + hash mod 20.
2. **Bucket → fold.** Bucket b is a test bucket of fold b // 20. In fold k,
   fold k's buckets are test, the last 2 buckets of every other fold are
   val, and the rest train: 72 / 8 / 20.

All aliquots of a patient land together, and folds are computed over the
whole dataset before any filter, so every model holds out the same
patients whatever quantification, genes or samples it loads. There is no
split file: the same dataset gives the same folds on any machine, and a
release that adds or drops a patient moves only the few others whose rank
crosses a block boundary. Test folds are stratified by project exactly;
val, 10% of every block by the hash, on average.

Val is for the model's early stopping, drawn evenly from every training
fold; a whole fold as val would leave only 60% to train on. Every patient
is a test patient in exactly one fold, so pooling the folds' test
predictions scores the whole cohort, which matters most for per-project
scores such as survival. One fold on its own is an ordinary train / val /
test split.

`data.samples["fold"]` is each sample's test fold — ready for
scikit-learn's `PredefinedSplit` when fitting a classical model.

## Choosing data

Defaults are raw `unstranded` counts over the 19,944 protein-coding genes
(GENCODE's 19,962 less 18 `_PAR_Y` copies, which are zero in every
sample), all samples, no transform.

```python
from reimp_shared.data import load_expression

data = load_expression()  # counts, protein coding
data = load_expression("tpm_unstranded", transform="log1p")  # log TPM
data = load_expression(transform="lognorm", library_size=1e5)  # TxFM's input
data = load_expression(gene_types=None)  # all genes
data = load_expression(gene_ids=vocab)  # a fixed vocabulary, in order
data = load_expression(projects=["TCGA-LUAD"], sample_types=["Primary Tumor"])
data = load_expression(fold=2)  # train / val / test of fold 2 (default: fold 0)

data.values  # (n_samples, n_genes) numpy
data.samples  # obs: sample_index, case, project, sample_type, ..., split, fold
data.genes  # var: gene_index, gene_id, gene_name, gene_type, ...
data.rows("train")  # row positions of the training samples
```

| argument | options |
|---|---|
| `quantification` | counts: `unstranded`, `stranded_first`, `stranded_second`; normalized: `tpm_unstranded`, `fpkm_unstranded`, `fpkm_uq_unstranded` |
| `gene_types` | GENCODE biotypes, default `("protein_coding",)`; `None` for all |
| `gene_ids` | Ensembl IDs, versioned or not; overrides `gene_types`, keeps the given order |
| `transform` | `none` (as stored), `log1p`, `lognorm` (library-size normalize over the selected genes, then log1p) |
| `projects`, `sample_types` | restrict samples; splits are unaffected |
| `fold` | cross-validation fold 0–4, default 0: which samples are `train`, `val` and `test` |

The Lightning `ExpressionDataModule` takes the same arguments, plus
`gene_ids_path`, `batch_size` and `num_workers`, and yields
`{"values": (B, G), "sample_index": (B,)}` batches. In a LightningCLI
config:

```yaml
data:
  quantification: tpm_unstranded
  gene_types: [protein_coding]
  transform: log1p
  batch_size: 64
  fold: 2  # train one model per fold, 0-4
```

For models that read a sample as a ranked list of genes, fit a
`GeneRanker` on the training rows and rank any sample by it:

```python
from reimp_shared.ranking import GeneRanker

data = load_expression("tpm_unstranded")
ranker = GeneRanker("tfidf", idf_scheme="entropy").fit(data.values[data.rows("train")])
order = ranker.order(data.values)  # (n_samples, n_genes) gene positions, highest score first
```

## Evaluations

[`EVALS.md`](EVALS.md) catalogues the evaluations — implemented and queued,
with the papers each is adapted from — and the rules they all follow.

## Command line

```bash
reimp-shared splits                        # samples per project in each fold's test set
reimp-shared splits --fold 2               # ... in fold 2's train / val / test
reimp-shared baseline-pca --out out/pca256  # one PCA per fold: pca256/fold{0..4}.parquet
reimp-shared baseline-hvg --out out/hvg5000 # the 5,000 most variable genes, per fold
reimp-shared probe out/pca256 out/other     # each a directory of a model's fold files
reimp-shared probe out/pca256 out/other --against pca256  # plus paired differences
```

A model writes one embeddings file per fold, from the model trained for
that fold (`write_embeddings(..., fold=k)`). `probe` reads a directory of
them as one set and fits each probe once per fold, on that fold's train
and val samples in that fold's embedding space; the folds' out-of-fold
predictions are pooled and scored once, with 95% patient-bootstrap
intervals shown as `[lo, hi]` (`--bootstrap 0` to skip them); `--json FILE`
also writes them, with a `meta` entry naming the dataset and label commits,
the salt, the replicates, the endpoint and the pathway collection. A
directory is read as its `*.parquet` files, and its name labels the scores. With all
five folds the scores cover every patient and are labelled `cv`; a single
fold's file — a quick train / val / test report — is scored on that
fold's test set and labelled `fold2`. C-index pairs are formed within a
fold, since each fold's model has its own risk scale, and the spectrum,
NMI and ARI, which describe a whole embedding space, are averaged over
folds. `alpha` is the median of the folds' choices. `--against NAME`
adds every other embeddings' scores minus NAME's: with the same seed, two
embeddings scored on the same patients share their bootstrap draws, so
the difference gets a paired interval and the share of replicates in
which it is positive. `--replicates` keeps the draws in `--json` for
comparisons later.

The evaluations:

- **Classification.** Logistic regression per task: `project_id` (cancer
  type, tumour samples), `tumor_vs_normal`, and four within-organ tasks
  on tumour samples — `lung` (LUAD / LUSC), `kidney` (KICH / KIRC / KIRP),
  `colorectal` (COAD / READ), `glioma` (GBM / LGG). Accuracy, balanced
  accuracy, macro-F1, weighted-F1.
- **Invertibility.** Ridge regression back onto a fixed target —
  log-normalized protein-coding counts — whatever the model was trained
  on. R² pooled over genes (high-variance genes dominate) and averaged per
  gene (every gene equal), and per-sample Pearson next to the Pearson of a
  constant train-mean profile. That baseline is high because genes differ
  in mean expression far more than samples do; the gap to it is what the
  embedding adds. PCA of the target is the best linear code of its size,
  so the PCA baseline is near the ceiling here rather than a bar to clear.
- **Pathways.** Ridge regression onto ssGSEA pathway scores (MSigDB
  Hallmark by default, `--pathways` for Reactome, PID, oncogenic or
  Cancer Cell Atlas), within projects on tumour samples: R² relative to
  each project's training mean, pooled over pathways and averaged per
  pathway. The scores are tcga2hf's per-sample `score_raw`.
- **Survival.** One primary tumour sample per patient (metastatic and
  recurrent samples are left out; EVALS.md says why), a ridge-penalized
  Cox model stratified by project, and Harrell's C-index
  over pairs of patients in the same project, plus the unweighted mean of
  per-project C-indexes and each project's own C-index with its interval
  (the `survival_projects` table). PFI by default (`--endpoint` for OS, DSS, DFI), as Liu et al.
  2018 recommend for most TCGA cancer types. Within a project, cancer type
  carries no information, so 0.5 is the baseline.
- **Geometry.** Against cancer type, on tumour samples: effective rank
  and top-eigenvalue share of the embeddings, precision@1 and @10 of
  cosine nearest training neighbours, NMI and ARI of k-means clusters, and
  silhouette.
- **Confounders.** Within projects, on tumour samples: R² of predicting
  library QC (log total reads, assigned-read fraction, strand balance)
  from the embedding, and how far nearest neighbours over-share a
  sample's sequencing plate or tissue source site (parsed from the
  barcodes). Lower is more invariant; read models against the PCA
  baseline and each other, since sites also differ in their patients.

Survival labels come from `tcga-patients-open`: only the label columns are
read, remotely — a few minutes the first time — then cached under
`$REIMP_CACHE` (default `~/.cache/reimp`) per dataset commit. That dataset
is one GDC release behind the expression data, so ~1% of expression cases
have no labels and are left out.

[ds]: https://huggingface.co/datasets/gabrielaltay/tcga-gene-expression-quantification-open

## Citations

BibTeX for the data (TCGA, distributed by the GDC), the gene sets (MSigDB)
and the per-sample pathway scores (ssGSEA, as implemented in GSVA).

```bibtex
@article{weinstein2013cancer,
    title     = {The Cancer Genome Atlas Pan-Cancer analysis project},
    author    = {Weinstein, John N and Collisson, Eric A and Mills, Gordon B and Shaw, Kenna R Mills and Ozenberger, Brad A and Ellrott, Kyle and Shmulevich, Ilya and Sander, Chris and Stuart, Joshua M},
    journal   = {Nature Genetics},
    volume    = {45},
    number    = {10},
    pages     = {1113--1120},
    year      = {2013},
    publisher = {Springer Science and Business Media LLC},
    doi       = {10.1038/ng.2764},
    url       = {https://doi.org/10.1038/ng.2764}
}
```

```bibtex
@article{grossman2016toward,
    title     = {Toward a Shared Vision for Cancer Genomic Data},
    author    = {Grossman, Robert L. and Heath, Allison P. and Ferretti, Vincent and Varmus, Harold E. and Lowy, Douglas R. and Kibbe, Warren A. and Staudt, Louis M.},
    journal   = {New England Journal of Medicine},
    volume    = {375},
    number    = {12},
    pages     = {1109--1112},
    year      = {2016},
    publisher = {Massachusetts Medical Society},
    doi       = {10.1056/nejmp1607591},
    url       = {https://doi.org/10.1056/nejmp1607591}
}
```

```bibtex
@article{subramanian2005gene,
    title     = {Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles},
    author    = {Subramanian, Aravind and Tamayo, Pablo and Mootha, Vamsi K. and Mukherjee, Sayan and Ebert, Benjamin L. and Gillette, Michael A. and Paulovich, Amanda and Pomeroy, Scott L. and Golub, Todd R. and Lander, Eric S. and others},
    journal   = {Proceedings of the National Academy of Sciences},
    volume    = {102},
    number    = {43},
    pages     = {15545--15550},
    year      = {2005},
    publisher = {National Academy of Sciences},
    doi       = {10.1073/pnas.0506580102},
    url       = {https://doi.org/10.1073/pnas.0506580102}
}
```

```bibtex
@article{liberzon2015molecular,
    title     = {The Molecular Signatures Database Hallmark Gene Set Collection},
    author    = {Liberzon, Arthur and Birger, Chet and Thorvaldsdóttir, Helga and Ghandi, Mahmoud and Mesirov, Jill P. and Tamayo, Pablo},
    journal   = {Cell Systems},
    volume    = {1},
    number    = {6},
    pages     = {417--425},
    year      = {2015},
    publisher = {Elsevier BV},
    doi       = {10.1016/j.cels.2015.12.004},
    url       = {https://doi.org/10.1016/j.cels.2015.12.004}
}
```

```bibtex
@article{barbie2009systematic,
    title     = {Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1},
    author    = {Barbie, David A. and Tamayo, Pablo and Boehm, Jesse S. and Kim, So Young and Moody, Susan E. and Dunn, Ian F. and Schinzel, Anna C. and Sandy, Peter and Meylan, Etienne and Scholl, Claudia and others},
    journal   = {Nature},
    volume    = {462},
    number    = {7269},
    pages     = {108--112},
    year      = {2009},
    publisher = {Springer Science and Business Media LLC},
    doi       = {10.1038/nature08460},
    url       = {https://doi.org/10.1038/nature08460}
}
```

```bibtex
@article{hanzelmann2013gsva,
    title     = {GSVA: gene set variation analysis for microarray and RNA-Seq data},
    author    = {Hänzelmann, Sonja and Castelo, Robert and Guinney, Justin},
    journal   = {BMC Bioinformatics},
    volume    = {14},
    number    = {1},
    year      = {2013},
    publisher = {Springer Science and Business Media LLC},
    doi       = {10.1186/1471-2105-14-7},
    url       = {https://doi.org/10.1186/1471-2105-14-7}
}
```
