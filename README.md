# reimp

Reimplementations of RNA-seq representation learners on one shared footing:
TCGA bulk RNA-seq from [`gabrielaltay/tcga-gene-expression-quantification-open`][ds]
(11,505 samples × 60,660 genes, 33 projects), the same patient-level
5-fold cross-validation, and the same linear probes for every model.

| directory | package | contents |
|---|---|---|
| [`shared/`](shared/) | `reimp-shared` | dataset access, the split, configurable data loaders, probes, PCA baseline |
| [`txfm/`](txfm/) | `reimp-txfm` | TxFM, a transformer masked autoencoder (Kenyon-Dean et al., 2026) |
| [`bulkrnabert/`](bulkrnabert/) | `reimp-bulkrnabert` | BulkRNABert, a BERT-style masked language model over 64-bin log-TPM tokens with learned gene embeddings (Gélard et al., 2024) |
| [`tifbert/`](tifbert/) | `reimp-tifbert` | TifBERT, a BERT doing masked gene modelling over each sample's genes ranked by tf-idf, in overlapping windows (Hosseini and Sharma, 2026) |
| [`bulkformer/`](bulkformer/) | `reimp-bulkformer` | BulkFormer, a GCN over a per-fold gene co-expression graph plus FAVOR+ Performer over all ~20k genes, masked-value regression (Kang et al., 2026) |
| [`tabssl/`](tabssl/) | `reimp-tabssl` | SCARF, VIME and BYOL on one MLP encoder, plus an untrained-encoder control (Dradjat et al., 2025) |
| [`vae/`](vae/) | `reimp-vae` | Tybalt (Way and Greene, 2018) and a tissue-supervised MMD autoencoder (Pande et al., 2026), the latter with `none` / `organ` / `project` supervision |
| [`compass/`](compass/) | `reimp-compass` | COMPASS, a gene-set concept bottleneck trained contrastively (Shen et al., 2026): 43 concept and 132 set scores, no cancer-type token |
| [`plier/`](plier/) | `reimp-plier` | PLIER / MultiPLIER, pathway-informed matrix factorization with a cell-marker + canonical-pathway prior, numpy port (Mao et al.; Taroni et al., 2019) |
| [`mojo/`](mojo/) | `reimp-mojo` | MOJO's RNA half: BulkRNABert's tokens on a convolutional U-Net with attention over 78 pooled positions (Gélard et al., 2025) |
| [`reports/`](reports/) | — | self-contained HTML reports on the data and the baselines, rebuilt from them |

Each model directory has a `paper.md` recording what the original work did —
model, data, evaluations, published numbers. The shared evaluations in
`reimp_shared.eval` are our standardized adaptations of those: one dataset,
one split, one set of labels for every model. [`shared/EVALS.md`](shared/EVALS.md)
catalogues them — what exists, where each came from, what is queued, and
the leakage rules every evaluation follows.

Each model's README ends with a Citations section giving the BibTeX for
the papers it implements. The sources of the evaluations are cited in
[`shared/EVALS.md`](shared/EVALS.md), and those of the data and gene sets
in [`shared/README.md`](shared/README.md). [`CITATION.cff`](CITATION.cff)
gives the citation for this repository.

## Develop

A [uv](https://docs.astral.sh/uv/) workspace; one `uv sync` at the root
installs every package and the dev tools.

```bash
uv sync
uv run pytest               # offline, against a miniature copy of the dataset
uv run pytest -m network    # against the published dataset (~0.8 GB download)
```

## Pipeline

Every model is trained once per fold, so every patient is scored once — by
the fold that never saw it:

```bash
uv run reimp-shared splits                                        # the five folds, per project
uv run txfm fit --config txfm/configs/tcga_s.yaml --data.fold 0   # and so on for folds 1-4
uv run txfm-embed --ckpt runs/txfm_s/fold0/checkpoints/best.ckpt --out out/txfm_s/fold0.parquet
uv run reimp-shared baseline-pca --out out/pca256                 # (Lib+Log)Norm + PCA, per fold
uv run reimp-shared baseline-hvg --out out/hvg5000                # the 5,000 most variable genes
uv run reimp-shared probe out/pca256 out/txfm_s --against pca256  # same probes, paired gaps
```

One fold on its own is a train / val / test split: probing a single fold's
file scores that fold's test set.

## Adding a reimplementation

0. Start with a stub: `<name>/README.md` and `<name>/paper.md` (what the
   paper did, and which of its choices are method-defining versus
   incidental to its data setup). The aim is the spirit of each method on
   this one dataset, not an exact reproduction: a method whose only new
   component is data we lack (another modality, a larger corpus) is
   skipped, and one that is still distinct on bulk RNA-seq alone gets an
   RNA-only version.
1. Create `<name>/pyproject.toml` depending on `reimp-shared`, with code in
   `<name>/src/reimp_<name>/` and tests in `<name>/tests/`.
2. Add `<name>` to `members`, `reimp-<name>` to `[tool.uv.sources]`, and
   `<name>/tests` to `testpaths` in the root `pyproject.toml`.
3. Load data through `reimp_shared.data` (choose quantification, genes and
   transform there; `fold` picks the cross-validation fold), train one
   model per fold, and write each fold's embeddings with
   `reimp_shared.eval.write_embeddings`, so `reimp-shared probe` can score
   the model next to every other one.
4. Test against the miniature dataset (`reimp_shared.testing`'s
   `write_fake_dataset` / `use_fake_dataset`, as `txfm/tests` does): a
   one-fold fit, an embeddings file that `read_embeddings` accepts, and the
   method-defining pieces (loss, masking, tokenizer) unit-tested on their
   own. Keep a `configs/debug.yaml` that runs end to end in a minute on a
   laptop, beside a full config. Every statistic the model fits (scalers,
   gene rankings, graphs, tokenizer maxima) comes from the fold's training
   samples only (`shared/EVALS.md`, rules 2–3).
5. Score it: write `out/<name>/fold{0..4}.parquet`, then
   `reimp-shared probe out/<name> out/pca256 --against pca256`, which prints
   every score beside the PCA baseline with paired intervals.

## License

MIT ([`LICENSE`](LICENSE)). Two third-party pieces carry their own terms:

- `compass/src/reimp_compass/data/conception_processed.tsv`, COMPASS's
  gene-set table, copied under its MIT licence
  ([`LICENSE-COMPASS`](compass/src/reimp_compass/data/LICENSE-COMPASS)).
- MSigDB gene sets are not in the repository: `reimp_shared.genesets`
  downloads them at run time, under MSigDB's terms (CC BY 4.0;
  `kegg_medicus` CC BY-SA 4.0).

[ds]: https://huggingface.co/datasets/gabrielaltay/tcga-gene-expression-quantification-open
