# reimp-bulkformer

BulkFormer — Kang, Fan, Yi, Cui, Cui, *BulkFormer: A large-scale foundation
model for bulk transcriptomes*, Cell Systems 17(7):101657, 2026
([doi](https://doi.org/10.1016/j.cels.2026.101657),
[KangBoming/BulkFormer](https://github.com/KangBoming/BulkFormer)).

Every gene of a sample is a token: a fixed sinusoid of its log1p TPM, plus a
learned gene embedding, plus an MLP summary of the whole sample. A graph
convolution over a gene co-expression network and Performer (FAVOR+) linear
attention mix all ~20k tokens, and the model regresses the values of the
~15% of genes it masked. A sample's embedding is its final gene tokens,
max-pooled over genes.

```bash
uv run bulkformer fit --config bulkformer/configs/debug.yaml               # ~15 s on real data
uv run bulkformer fit --config bulkformer/configs/tcga.yaml --data.fold 0  # and so on for folds 1-4
uv run bulkformer-embed --ckpt runs/bulkformer/fold0/checkpoints/last.ckpt \
    --out out/bulkformer/fold0.parquet
uv run bulkformer-embed --ckpt runs/bulkformer/fold0/checkpoints/last.ckpt --pooling mean \
    --out out/bulkformer_mean/fold0.parquet
uv run reimp-shared probe out/bulkformer out/pca256 --against pca256
```

Fold k of a run lands in `<trainer.default_root_dir>/fold<k>/`
(`reimp_shared.foldcli`): `config.yaml`, the CSV and TensorBoard logs, and
`checkpoints/best.ckpt` (lowest `val/loss`) beside `checkpoints/last.ckpt`
(the end of the cosine schedule, which is what we embed). Rerunning a fold
replaces its run.

The debug config trains a 1.3M-parameter model for 40 steps of 4 samples
on fold 0 and saves only `runs/bulkformer_debug/fold0/checkpoints/last.ckpt`;
on an M4 Max (MPS) the whole fit, data loading and graph included, takes ~15 s,
and embedding all 11,505 samples from it ~2.5 min. `tcga.yaml`'s 14.8M
model measured 2.2 s per step of 8 samples on the same machine, with a
31 GiB peak MPS allocation: ~37 min per epoch, ~12 h for its 20 epochs per
fold.
The graph ablation from `paper.md` is one flag:

```bash
uv run bulkformer fit --config bulkformer/configs/tcga.yaml --model.graph random  # degree-matched control
uv run bulkformer fit --config bulkformer/configs/tcga.yaml --model.graph none    # Performer only
```

[`paper.md`](paper.md) records what the paper did, including its
evaluations; below is how this reimplementation follows it.

## From the paper

| | |
|---|---|
| input | log1p TPM (`tpm_unstranded`, `transform: log1p`); every gene is a token |
| expression embedding | fixed `[sin(xθ), cos(xθ)]`, θ_i = 100^(−2i/d); zero at masked genes |
| gene identity | `nn.Embedding` with Xavier init, then an MLP d→4d→d (the Dec 2025 code, not ESM2) |
| sample context | an MLP over the whole masked input vector (masked genes at −10), added to every token |
| token | the sum of the three, then an MLP d→4d→d |
| block | LayerNorm → x + GCN(x) → K pre-norm Performer layers (8 heads, FFN ×4, GELU) |
| gene graph | \|Pearson r\| co-expression, top 20 per gene with its self-edge, \|r\| ≥ 0.4; GCN symmetric normalization, no added self-loops |
| head | LayerNorm, MLP d→4d→1, ReLU (v1's head, without the current code's appended scalars) |
| objective | ~15% of genes masked per sample, MSE on those; no 80/10/10 |
| optimizer | AdamW, peak lr 1e-4, linear warmup over 5% of steps |
| embedding | max over the final Performer layer's gene tokens; mean as a variant |

## The gene graph, per fold

The released graph is TCGA co-expression over every patient, test patients
included (`paper.md`, "Leakage"), so it is not used. `LitBulkFormer.setup`
builds the graph from the fold's **training samples only**, before the
first step (`graph.coexpression_graph`):

- **Linear TPM.** Correlations are over `tpm_unstranded` as stored (the
  `expm1` of the model's log1p inputs), not over log1p: linear TPM
  reproduced the released graph better (50% of its edges among our top-20
  neighbours, against 33% for log1p). `graph_space: log1p` is the
  alternative.
- Each gene keeps its 20 highest-|r| genes, counting itself (r = 1) as the
  released graphs do, then drops those with |r| < 0.4. A gene constant over
  the training samples keeps only its self-edge.
- The top-20 lists are directed; the graph is their union, made
  undirected, so `D^-1/2 A D^-1/2` is well defined. The normalized
  adjacency is one fixed sparse COO matrix and the GCN one
  `torch.sparse.mm` per layer (COO works on MPS, CSR does not); no
  torch_geometric.
- On fold 0 (8,304 training samples × 19,944 genes) it takes ~2 s on CPU,
  so it is rebuilt at every fit rather than cached. It is saved in the
  checkpoint (`model.graph_index`, `model.graph_weight`), so embedding
  never refits it.
- Fold 0's graph: 251,333 undirected edges between distinct genes (median
  |r| 0.57) plus 19,944 self-edges; a median degree of 22 counting the
  self-edge, a maximum of 465; 1,806 genes keep only their self-edge,
  367 of them because they are constant in training.

`graph: random` shuffles the gene labels of that same graph — every degree
and weight survives, which genes they join does not — and `graph: none`
drops the GCN.

## Fitted statistics

All in `LitBulkFormer.fit_statistics`, from `data.rows("train")` only
(`shared/EVALS.md`, rule 3), and all saved in the checkpoint:

- the gene graph (above);
- each gene's training mean log1p TPM, used only for the validation
  baseline `val/loss_gene_mean`;
- the head's output bias, started at the training mean log1p TPM.

`tests/test_lit.py::test_fitted_statistics_come_from_training_rows_only`
scrambles every validation and test sample as loaded
(`reimp_shared.testing.scramble_held_out`) and checks that none of them
moves; scrambling the training samples does.
`tests/test_cli.py::test_embedding_ignores_held_out_rows` does the same at
embed time: after scrambling, `bulkformer-embed` gives every training
sample of the checkpoint's fold the same embedding as before.

## Configs and fields

| config | size |
|---|---|
| `debug.yaml` | d 32, 1 Performer layer, 4 heads: 1.3M parameters; 40 steps at batch 4 |
| `tcga.yaml` | reimp's default, d 256, N 1, K 4, 8 heads, sample MLP d wide: 14.8M (5.1M gene table, 5.2M sample MLP, 4.5M the rest); 20 epochs at batch 8 |
| `tcga_147m.yaml` | the released "147M" configuration, d 640, K 12, sample MLP 2,560 wide: 133.2M on our genes; activation checkpointing |

`data` is the shared `ExpressionDataModule` (`shared/README.md`): the
model needs `quantification: tpm_unstranded` and `transform: log1p`, and
`fold` picks the cross-validation fold. `model` fields (`LitBulkFormer`):

| field | default | |
|---|---|---|
| `d_model`, `n_blocks`, `n_layers`, `n_heads` | 256, 1, 4, 8 | width, N graph blocks, K Performer layers per block, heads |
| `n_features` | null | FAVOR+ random features per head; null = d_head · ln d_head |
| `mlp_ratio` | 4.0 | hidden width of the FFNs and per-token MLPs, × d |
| `sample_hidden` | null | the sample-context MLP's hidden width; null = d |
| `dropout` | 0.1 | in the FFNs, attention output and the sample and token MLPs |
| `mask_ratio` | 0.15 | fraction of genes masked per sample |
| `graph` | `coexpression` | `coexpression`, `random` (degree-matched control) or `none` (Performer only) |
| `graph_k`, `graph_threshold` | 20, 0.4 | edges per gene counting its self-edge; minimum \|r\| |
| `graph_space` | `linear` | correlate linear TPM, or the log1p values |
| `feature_redraw_interval` | 1000 | optimizer steps between FAVOR+ feature redraws; null never |
| `activation_checkpointing` | false | recompute Performer layers in the backward pass |
| `pooling` | `max` | how `predict_step` pools the gene tokens; `bulkformer-embed --pooling` overrides it |
| `lr`, `weight_decay`, `warmup_frac` | 1e-4, 0.01, 0.05 | AdamW; linear warmup, then cosine decay to 0 |
| `seed` | 0 | validation masks and the random graph |

Validation logs the masked MSE (`val/loss`), the same MSE for predicting
each gene's training mean (`val/loss_gene_mean`, the baseline the paper's
pooled PCC hides), and per-sample Pearson on the masked genes
(`val/pearson`).

## Deviations

From `paper.md`'s "For reimp" decisions (method kept, incidentals
standardized):

- **Graph fit per fold on training samples**, from linear TPM, instead of
  the released `G_tcga.pt`.
- **Smaller model**: d 256, N 1, K 4, sample MLP d wide, against the
  released 132M configuration, which is kept as `tcga_147m.yaml`.
- **Pretraining on one fold's ~8,300 TCGA training samples**, not 0.5M
  ARCHS4 profiles, and our 19,944 protein-coding genes (their list maps to
  19,907 GENCODE v36 IDs; pass it with `data.gene_ids_path`).
- **Embedding**: max over all genes (mean as a variant), without the three
  appended per-sample scalars and without the 2,000-gene pooling list.

Where the paper or its code is silent, or we differ:

- **FAVOR+ is our own** (`favor.py`), since performer-pytorch is
  unmaintained: its feature map and stabilizers, orthogonal random
  features (d_head · ln d_head per head), one projection shared across
  heads, redrawn every 1,000 steps, and its `eps` of 1e-4. Keys are scaled
  by their largest feature over the sequence, so over thousands of genes a
  typical key feature is comparable to 1e-4, which shrinks the estimate
  towards uniform attention. At the default 110 features per 32-d head
  that trades a little bias for much less variance: on 2,048-4,096
  random tokens, eps 1e-6 was never closer to softmax attention, and at
  query and key scales 0.7-1.0 it was 1.4-2.8 times as far
  (`tests/test_favor.py`).
- **Undirected graph**: the union of the top-20 lists. The paper passes its
  graph to GCNConv without saying which direction an edge runs.
- The GCN's bias is added after aggregation, as in GCNConv.
- **Masking** takes exactly round(0.15 · G) genes per sample, which
  "about 15%" allows and which makes every sample's masked set the same
  size.
- **One dropout rate**, 0.1 (the current code: 0.05 in attention, 0.1 in
  the FFN), and GELU in every MLP.
- **Output-bias init**: the head's final bias starts at the training mean
  log1p TPM. With ReLU on the output and one offset shared by every gene, a
  default init can clamp every prediction to zero and stall training.
- Not implemented: the current code's shift of masked predictions so their
  mean matches the observed genes' (a prediction-head detail that uses the
  sample's observed mean); the STRING PPI and `G_gtex.pt` graph variants
  (external data); bf16.
- **Schedule and batch**: cosine decay after the paper's warmup, weight
  decay 0.01 on matrices and embeddings, gradient clipping at 1.0, batch 8
  for 20 epochs — ~21k optimizer steps, near the paper's ~30k at batch
  512.
