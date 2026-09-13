# reimp-tabssl

Self-supervised objectives for tabular data — SCARF, VIME and BYOL — as
Dradjat, Hamidi, Bartet and Hanczar adapt them to bulk expression:
*Self-supervised representation learning on gene expression data*,
Bioinformatics 41(11):btaf533, 2025
([doi](https://doi.org/10.1093/bioinformatics/btaf533),
[kdradjat/SSRL_RNAseq](https://github.com/kdradjat/SSRL_RNAseq); the link
printed in the paper is dead, and the released code does not run as is).

One encoder, four objectives. A 4 × [Linear 256 → BatchNorm → ReLU →
Dropout 0.2] MLP reads a sample's z-scored log expression; its 256-d output
is the embedding. It is trained to match a corrupted copy of each sample
(SCARF), to recover corrupted genes and the corruption mask (VIME), or to
predict an averaged teacher network across a clean and a corrupted view
(BYOL). `none` keeps the same encoder at its random initialization — the
control without which "SSL helps" cannot be told from "random ReLU
features of 20k genes probe well". With the architecture and inputs fixed,
what differs is the objective alone.

```bash
# ~20 s per objective on real data: the pipeline end to end, fold 0
uv run tabssl fit --config tabssl/configs/debug.yaml --model.objective vime \
  --trainer.default_root_dir runs/tabssl_debug/vime
uv run tabssl-embed --ckpt runs/tabssl_debug/vime/fold0/checkpoints/last.ckpt \
  --out out/tabssl_debug/vime/fold0.parquet

# The paper's settings, one run per fold and objective (scarf, vime, byol, none)
uv run tabssl fit --config tabssl/configs/tcga_scarf.yaml --data.fold 0  # and so on for folds 1-4
uv run tabssl-embed --ckpt runs/tabssl_scarf/fold0/checkpoints/best.ckpt
uv run reimp-shared probe out/tabssl_scarf out/pca256 --against pca256
```

`tabssl-embed` reads the fold, the data settings and the objective from the
checkpoint and, without `--out`, writes `out/tabssl_<objective>/fold<k>.parquet`:
each objective is its own set of embeddings for `reimp-shared probe`, one
file per fold. Fold k of a run lands in `<trainer.default_root_dir>/fold<k>/`
— `runs/tabssl_scarf/fold0/` holds the config, metrics and TensorBoard
events, and `checkpoints/` below it — and rerunning a fold replaces it.
Each full config early-stops on the validation patients' loss and keeps the
best checkpoint, `best.ckpt`, beside `last.ckpt`; embed from `best.ckpt`.
The `none` config makes one pass over the training samples and writes only
`last.ckpt`: embed from `runs/tabssl_none/fold<k>/checkpoints/last.ckpt`.

[`paper.md`](paper.md) records what the paper did, including its
evaluations; below is how this reimplementation follows it.

## Objectives

| | SCARF | VIME | BYOL | none |
|---|---|---|---|---|
| corruption | exactly ⌊0.3 · G⌋ genes per sample | each gene with probability 0.3 | as VIME, on the second view | — |
| replacement | Uniform[min_j, max_j] of the training samples (or their empirical marginal) | the gene's value in a random training sample | as VIME | — |
| views | clean anchor + one corrupted copy | one corrupted copy | clean + corrupted | — |
| head | Linear-BN-ReLU-Linear, 256 → 256 → 256 | mask and feature decoders, 4 × [Linear 256, ReLU] → G each | projector and predictor, 256 → 4096 → 256; EMA target of encoder + projector | — |
| loss | NT-Xent, cosine, τ = 1.0, symmetric over 2N views | BCE(mask) + 2.0 · MSE(all genes) | 2 − 2·cos, symmetric | — |
| optimizer | Adam 1e-4 | RMSprop 1e-3 | Adam 1e-4; EMA decay 0.9, fixed | none; one pass sets BatchNorm statistics |
| batch, max epochs | 256, 1,000 | 32, 500 | 32, 50 | 256, 1 |
| config | `tcga_scarf.yaml` | `tcga_vime.yaml` | `tcga_byol.yaml` | `tcga_none.yaml` |

Embeddings are the encoder's output on clean, scaled inputs in eval mode:
no corruption, no head, no dropout.

## Inputs and fitted statistics

Batches come from the shared `ExpressionDataModule`: protein-coding
`unstranded` counts, `lognorm` (library size 1e5, then log1p) by default,
or `log1p` of `tpm_unstranded`. At the start of `fit`, on the fold's
**training samples only**, the model fits:

- a per-gene z-score (`GeneScaler`: mean and standard deviation; a gene
  constant over training gets std 1), saved in the checkpoint and applied
  to every input, at training and at embedding;
- each gene's scaled [min, max] over the training samples, SCARF's
  uniform replacement range, also saved; every scaled input, at training
  and at embedding, is clipped to it (below);
- the pool of scaled training samples every marginal replacement (VIME,
  BYOL, SCARF's `marginal` option) is drawn from — rebuilt from the data
  at each fit rather than saved;
- BatchNorm running statistics, from training batches as usual (and for
  `none`, an exact average over one pass).

Validation samples are scaled by the training scaler and corrupted from the
training pool; they are used for the loss, early stopping and the best
checkpoint, nothing else. Test samples are touched only at embedding.
`tests/test_lit.py` rewrites every validation and test sample and checks
that no fitted statistic moves; `tests/test_cli.py` does the same around
`tabssl-embed` and checks that no training sample's embedding moves.

**Clipping to the training range** is ours; the paper's `StandardScaler`
does not clip. A gene nearly constant over the training samples (expressed
in only one of them, say) has a tiny standard deviation, and a held-out
sample that expresses it more scales to thousands of them. On fold 0,
1,017 genes have 0 < std < 0.01, training z-scores peak at |z| = 91
(√(n_train − 1), one expressing sample) and held-out ones at 1,398, with
118 entries in 52 held-out samples beyond 100. One such gene swamps the
first layer and throws the sample's embedding outside every training
sample's. Clipped to each gene's scaled training [min, max], the training
samples are unchanged and held-out |z| peaks at 88; 0.017% of held-out
entries are clipped. A gene constant over training clips to its one value,
so it carries no signal at embedding either.

Training corruptions are redrawn every batch from the global RNG
(`seed_everything`); validation corruptions come from a generator reset to
`seed` before each validation pass, so the validation loss compares like
with like across epochs.

## Config fields

`model:` (`LitTabSSL`); `n_genes` is linked from the DataModule.

| field | default | |
|---|---|---|
| `objective` | `scarf` | `scarf`, `vime`, `byol` or `none` (the untrained encoder) |
| `hidden_dim`, `n_layers`, `dropout` | 256, 4, 0.2 | the encoder (paper Table S4) |
| `corruption_rate` | 0.3 | every objective (Table S3) |
| `scarf_replacement` | `uniform` | `uniform` on [min, max], or `marginal`: the original SCARF, which the paper's Fig. S6 finds no better than training from scratch |
| `temperature` | 1.0 | SCARF's NT-Xent τ |
| `decoder_layers`, `vime_alpha` | 4, 2.0 | VIME's decoders and reconstruction weight |
| `byol_hidden_dim`, `byol_ema` | 4096, 0.9 | BYOL's projector / predictor width and target decay |
| `lr` | `null` | `null` takes the objective's rate: SCARF 1e-4, VIME 1e-3, BYOL 1e-4 |
| `seed` | 0 | validation corruptions |

`data:` takes every `ExpressionDataModule` argument (`quantification`,
`gene_types`, `transform`, `batch_size`, …); `data.fold` picks the
cross-validation fold. `trainer:` holds epochs, early stopping and
checkpoints.

## Deviations from the paper

The *For reimp* section of [`paper.md`](paper.md) sets these out; in brief:

- **Data.** TCGA only (no ARCHS4), the shared patient-level folds instead
  of a split by sample, and pretraining on training patients only. 19,944
  protein-coding genes instead of all 56,902. `lognorm` input and one
  z-score per fold fit on training samples, saved with the model; the
  paper quantile-normalized the whole cohort before splitting and refit a
  `StandardScaler` on each file it loaded, test samples included.
- **Scaled inputs are clipped** to each gene's training range, at training
  and at embedding, so a gene nearly constant over training cannot give a
  held-out sample z-scores in the thousands (see *Inputs and fitted
  statistics*).
- **Early stopping** on the validation patients' pretext loss (patience 30,
  the paper's only stated patience), embedding the best checkpoint. The
  paper trains for a fixed number of epochs, and the epoch it reports is
  not stated.
- **VIME's corruption is redrawn every batch.** The paper's code, like the
  original VIME's, draws it once before training and reuses it every
  epoch — an artefact of the reference code, not of the method.
- **VIME's replacement** is drawn per entry, with replacement, from the
  training samples: each gene's empirical marginal, as a column-wise
  permutation gives, but not tied to a batch or a fixed shuffle.
- **VIME's decoders end in a linear layer.** The paper's end in a sigmoid;
  the mask decoder's is folded into `BCEWithLogits` (the same loss), and
  the feature decoder drops it, since a sigmoid cannot output the
  negative half of its z-scored targets (a wrinkle in the paper's code).
  RMSprop takes Keras's defaults (ρ 0.9, ε 1e-7), as the paper's VIME is
  Keras.
- **BYOL's second view** replaces genes from the training pool, as VIME
  does, instead of a permutation within the mini-batch (their loader does
  not shuffle, so a batch's marginal depended on file order). Adam 1e-4 as
  in the code, not SGD at 1e-4 as in Table S3, which is implausibly slow.
  The code's Dropout 0.2 in the projector and predictor is left out: the
  paper's text gives Linear 4096 → BN → ReLU → Linear. The EMA moves
  parameters only; each network keeps its own BatchNorm statistics.
- **Frozen embeddings only.** reimp scores embeddings with the shared
  linear probes — the paper's "frozen" setting, where all three methods did
  worst. Unfrozen fine-tuning is out of scope.
- **`none`** is ours: the untrained-encoder control the paper lacks.

## Citations

BibTeX for the study this directory follows (Dradjat et al.) and for the
three methods it compares: SCARF, VIME and BYOL.

```bibtex
@article{dradjat2025selfsupervised,
    title     = {Self-supervised representation learning on gene expression data},
    author    = {Dradjat, Kevin and Hamidi, Massinissa and Bartet, Pierre and Hanczar, Blaise},
    journal   = {Bioinformatics},
    volume    = {41},
    number    = {11},
    year      = {2025},
    publisher = {Oxford University Press (OUP)},
    doi       = {10.1093/bioinformatics/btaf533},
    url       = {https://doi.org/10.1093/bioinformatics/btaf533}
}
```

```bibtex
@misc{bahri2021scarf,
    title     = {SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption},
    author    = {Bahri, Dara and Jiang, Heinrich and Tay, Yi and Metzler, Donald},
    year      = {2021},
    publisher = {arXiv},
    doi       = {10.48550/arXiv.2106.15147},
    url       = {https://arxiv.org/abs/2106.15147}
}
```

```bibtex
@inproceedings{yoon2020vime,
    title     = {VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain},
    author    = {Yoon, Jinsung and Zhang, Yao and Jordon, James and van der Schaar, Mihaela},
    booktitle = {Advances in Neural Information Processing Systems},
    volume    = {33},
    pages     = {11033--11043},
    year      = {2020},
    publisher = {Curran Associates, Inc.},
    url       = {https://proceedings.neurips.cc/paper_files/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf}
}
```

```bibtex
@misc{grill2020bootstrap,
    title     = {Bootstrap your own latent: A new approach to self-supervised Learning},
    author    = {Grill, Jean-Bastien and Strub, Florian and Altché, Florent and Tallec, Corentin and Richemond, Pierre H. and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
    year      = {2020},
    publisher = {arXiv},
    doi       = {10.48550/arXiv.2006.07733},
    url       = {https://arxiv.org/abs/2006.07733}
}
```
