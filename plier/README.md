# reimp-plier

PLIER — Mao et al., *Pathway-level information extractor (PLIER) for gene
expression data*, Nature Methods 2019
([doi 10.1038/s41592-019-0456-1](https://doi.org/10.1038/s41592-019-0456-1)) —
and its transfer use, MultiPLIER — Taroni et al., Cell Systems 2019
([doi 10.1016/j.cels.2019.04.003](https://doi.org/10.1016/j.cels.2019.04.003)).

A matrix factorization whose gene loadings are pulled toward sparse,
non-negative combinations of curated gene sets. A sample's embedding is a
fixed ridge projection onto those loadings, the same for training and
held-out samples, which is how MultiPLIER projects new data. It is linear,
runs on a CPU, and is the only model here built on prior knowledge.

**The solver is written from the two papers**: PLIER's Methods and main
text, and MultiPLIER's STAR Methods for the projection. It is not derived
from any implementation of PLIER. Where the papers are silent, the
choices are reimp's own, and they are listed below under "Our decisions".
It is numpy and scipy, with scikit-learn's lasso for U. There is no
neural net, so no Lightning and no GPU.

```bash
uv run plier fit --config plier/configs/debug.yaml                     # ~15 s on real data
uv run plier-embed --model runs/plier_debug/fold0 --out out/plier_debug/fold0.parquet
uv run plier fit --config plier/configs/tcga.yaml --data.fold 0        # and so on for folds 1-4
uv run plier-embed --model runs/plier/fold0 --out out/plier/fold0.parquet
uv run plier fit --config plier/configs/tcga.yaml --prior null --out_dir runs/plier_noprior
```

[`paper.md`](paper.md) records what the papers did and, under "For
reimp", which of their choices are kept.

## From the papers

Y is genes × samples, z-scored per gene; C is the binary genes × gene-sets
prior; Z (genes × k) are the loadings, B (k × samples) the latent
variables (LVs) and U (gene sets × k) the prior's coefficients.

| | |
|---|---|
| objective | ‖Y − ZB‖² + λ1‖Z − CU‖² + λ2‖B‖² + λ3‖U‖₁ with Z ≥ 0, U ≥ 0 |
| solver | block coordinate descent on Z, U and B, from the SVD of Y |
| start | Y = UDVᵀ, so Z ≈ UD^½ and B ≈ D^½Vᵀ: the loop starts from B = D^½Vᵀ for the top k |
| Z step | (YBᵀ + λ1CU)(BBᵀ + λ1I)⁻¹, the minimizer of the first two terms, with its negative part set to 0. The paper prints the inverse as (Bᵀ + λ1I)⁻¹; we read it as the minimizer's BBᵀ |
| U step | argmin over U ≥ 0 of ‖Z − CU‖² + λ3‖U‖₁, one LV (column) at a time |
| B step | (ZᵀZ + λ2I)⁻¹ZᵀY |
| λ1, λ2 | d_k / 2 and d_k, from the k-th singular value of Y |
| λ3 | adjusted periodically so that 70% of the LVs use a gene set (`frac`) |
| k | twice the number of significant principal components, found by an elbow or by significance |
| stopping | the relative change in B below 5e-6, or once it levels off |
| annotations | a random fifth of each gene set's genes is held out of C. Each positive U[s, i] is scored by an AUC and p-value: set s's held-out genes against the genes not in s, ranked by Z[:, i]. "High confidence" is AUC > 0.7 and FDR < 0.05. This labels LVs; it does not change B |
| embedding | B = (ZᵀZ + λ2I)⁻¹Zᵀy for every sample (MultiPLIER) |

## Our decisions

Where the papers leave something open:

- **k.** By default, the elbow of the fold's training singular values,
  times 2. The elbow is found by the chord rule. The index and the
  singular values are both scaled to [0, 1], and the elbow is the point
  farthest below the straight line from the first value to the last. The
  components before it are kept. The spectrum is every singular value of
  Y, from the eigenvalues of its smaller Gram matrix (27 s on fold 0).
  - On fold 0 the rule keeps 246 components, so k = 492, λ2 = d_492 = 154
    and λ1 = 77.
  - `--model.k_rule gavish_donoho` counts instead the singular values
    above Gavish and Donoho's (2014) optimal hard threshold for unknown
    noise, ω(β) × the median singular value. This stands in for the
    paper's significance route; its permutation test (Leek 2007) is not
    implemented. On fold 0 it keeps 1,537 components (k = 3,074): with
    8,304 samples nearly everything is significant, so it is not the
    default.
  - k is capped at the rank of Y; a fixed `--model.k` skips the rule.
- **SVD.** An exact SVD when Y has at most 1,000 samples or genes, else
  scikit-learn's `randomized_svd` (10 oversamples, seeded). Each
  component's sign is chosen so that its left singular vector's positive
  part is the larger, since that is the part a non-negative Z keeps.
- **When the prior enters.** The paper's loop does not say when U starts.
  U stays 0 until that factorization stops by the stopping rule; then the
  prior enters, and the fit runs again to the same rule. The no-prior
  ablation is therefore exactly the first phase of the prior fit: the
  same k, λ1, λ2 and iterates. A test checks it. With the prior, each
  iteration runs U, then Z, then B, a cyclic reordering of the paper's
  Z, U, B, so that the first iteration with the prior already uses U.
- **Stopping.** The relative change is ‖ΔB‖_F / ‖B‖_F, not squared.
  "Levels off" means no new low in 20 iterations (`patience`). Each phase
  runs for at most 300 iterations (`max_iter`).
- **The U step.** A pure L1 penalty, as the paper writes it, with no
  intercept, over every gene set with no preselection. It is solved by
  scikit-learn's coordinate descent (`lasso_path`, `positive=True`) on the
  precomputed Gram matrix CᵀC, warm-started from the last U, with a
  relative duality-gap tolerance of 1e-8. scikit-learn scales the squared
  error by 1/2n, so its α is λ3 / 2n, and λ3 is reported on the paper's
  scale.
- **λ3.** The paper's binary search has a closed form. With U ≥ 0, an
  LV's column of U is 0 exactly when λ3 ≥ 2 maxₛ(Cᵀz)ₛ (the KKT
  conditions). So λ3 is the midpoint between the round(0.7k)-th and the
  next largest of those values, which meets the 70% target exactly for
  the current Z. LVs below that bound are not fit at all.
  - λ3 is set when the prior enters and again every 10 iterations
    (`l3_every`). Between settings the share drifts as Z moves toward CU:
    on fold 0 at k = 492 it went from 344 to 393 LVs in five iterations.
  - `--model.l3` fixes it instead.
- **Gene sets.** Sets with fewer than 5 genes among the modelled ones are
  left out of C (`min_genes`), so every set used has at least one
  held-out gene. From each other set, ⌊size / 5⌋ genes are held out, drawn
  with `seed`.
- **Annotation statistics.** The AUC is the Mann–Whitney U over the number
  of pairs, with ties counting a half; this matters because a clipped Z
  has many zeros. p is scipy's one-sided Mann–Whitney test (normal
  approximation with tie correction for samples this size). FDR is
  Benjamini–Hochberg over every positive U entry. `--model.holdout 0`
  holds nothing out and annotates nothing.
- **The objective** is recorded at every iteration. The paper's Z step
  clips an unconstrained minimizer rather than solving the constrained
  problem, so the objective can rise slightly in some iterations. This
  happened in the U = 0 phase on the simulated test data. With the prior
  and a fixed λ3 it fell at every iteration there. The U and B steps are
  exact minimizers, and a test checks that neither raises it.

## Ours, for reimp

Following paper.md's "For reimp":

- **Input**: log1p library-normalized unstranded counts (`lognorm`, library
  1e5) on the 19,944 protein-coding genes, as for the PCA baseline. The
  closer-to-paper z-scored RPKM is
  `--data.quantification fpkm_unstranded --data.transform none`.
- **One model per fold, trained on that fold's training samples** (8,304 on
  fold 0), not on recount2. Every statistic is fit on those samples alone
  (`shared/EVALS.md`, rule 3), and a test checks it:
  - per-gene means and SDs (n − 1 denominator), and each gene's range;
  - which genes are dropped as constant (367 on fold 0);
  - the SVD, the spectrum and k;
  - λ1, λ2 and λ3;
  - which genes are held out.

  Validation and test samples are clipped to each gene's training range
  (below), z-scored with the training means and SDs, then projected.
  MultiPLIER instead z-scores each target dataset on itself, which would
  recentre every test fold.
- **Held-out values are clipped to the training range.** Before z-scoring,
  each value is clipped to its gene's minimum and maximum over the
  training samples, so no held-out z-score leaves the range the training
  ones span. Training samples are unchanged, and so is the fit. Without
  the clip, genes nearly constant over training dominate some held-out
  projections. Measured on fold 0 (2026-09-13):
  - training z-scores reach at most 91 (√(n − 1)), held-out ones 1,398
    (KRTAP20-1, SD 8e-4);
  - 57% of test and 62% of validation samples have a gene outside its
    training range, though only 0.02% of values are;
  - for the worst 1% of samples, the excess beyond the range is over 11%
    (test) and 26% (validation) of ‖z‖², and at most 89%.
- **Genes constant over the training samples are dropped**; their z-score
  is undefined.
- **All genes**: every protein-coding gene that varies over the training
  samples is modelled. Genes in no set get empty rows of C.
  `--all_genes false` models only the genes in some set.
- **k**: the elbow rule above by default. The fixed-k variants are
  `--model.k 64` and `--model.k 256`, to sit beside PCA at equal dimension.
- **The no-prior ablation**, `--prior null`, runs the same solver with
  U = 0, so λ1‖Z‖² replaces λ1‖Z − CU‖². The PLIER preprint makes the same
  comparison by setting λ3 high. It keeps the same k, λ1 and λ2, so the
  difference between the two runs is what the prior buys.

### Prior

`prior` lists gene sets as `reimp_shared.genesets.load_gene_sets` reads
them: MSigDB 2026.1 collections by name, each downloaded once into
`$REIMP_CACHE/msigdb/` and md5-checked, and GMT files of your own by path.
Whole collections, with no per-set filtering. The default is MultiPLIER's
recipe, canonical pathways plus cell-type markers, as far as MSigDB's
licences allow:

| name | MSigDB collection | licence |
|---|---|---|
| `reactome` | C2:CP:REACTOME | CC BY 4.0 |
| `pid` | C2:CP:PID | CC BY 4.0 |
| `wikipathways` | C2:CP:WIKIPATHWAYS | CC BY 4.0 |
| `kegg_medicus` | C2:CP:KEGG_MEDICUS | CC BY-SA 4.0 |
| `cell_type` | C8, cell type signatures from single-cell studies | CC BY 4.0 |

Left out, and why:
- **KEGG_LEGACY and BioCarta**, the rest of C2:CP: KEGG and BioCarta
  license them to the Broad Institute alone, so `genesets` does not offer
  them. MultiPLIER's prior had 122 KEGG and 25 BioCarta sets.
- **The PLIER package's cell-type sets.** CIBERSORT's LM22 is free to
  academic users on registration, and the IRIS and DMAP marker sets are
  journal supplements with no stated licence. C8 stands in for them.
  Anyone with LM22 can add it by path:
  `--prior '[reactome, pid, wikipathways, kegg_medicus, cell_type, lm22.gmt]'`.
- **C2:CGP**, chemical and genetic perturbations: some of its sets were
  defined on TCGA patients, test patients included (paper.md).

**Overlap with the pathway probe is accepted, and must be reported.**
PLIER is a baseline here, not a model to pursue, so the prior is not
trimmed around the probes. `reactome` and `pid` are two of the collections
the pathway probe scores, and WikiPathways shares much of their content:
PLIER's Reactome and PID pathway scores are circular, and Hallmark,
oncogenic and Cancer Cell Atlas are the collections it did not read.

**Mapping symbols to genes.** Symbols are matched exactly to GENCODE v36
`gene_name`, with no alias rescue. The symbols that name none of the
modelled genes — renamed symbols, non-coding genes, genes constant over
the training samples — are counted in the log on every fit and saved in
`model.npz` as `unmapped`. The default prior leaves 4,876 unmapped on
fold 0: `cell_type` (C8) alone names 20,573 genes, more than the 19,577
protein-coding genes modelled.

## Configs

| field | |
|---|---|
| `out_dir` | fold k's model is written to `<out_dir>/fold<k>/` |
| `prior` | a list of MSigDB collection names and GMT file paths; `null` for the no-prior ablation |
| `all_genes` | `true`: every varying gene; `false`: the prior's genes only |
| `data.*` | `load_expression` arguments: `quantification`, `gene_types`, `transform`, `library_size`, `projects`, `fold`, `revision` |
| `model.*` | `PLIER` arguments: `k` (null: the rule), `k_rule`, `k_factor`, `l1`, `l2`, `l3` (null: the rules above), `frac`, `l3_every`, `holdout`, `min_genes`, `max_iter` (per phase), `tol`, `patience`, `seed` |

- **`debug.yaml`** uses fold 0's whole training set with the default prior
  (4,484 sets, 4,472 with at least 5 genes), but k = 32 and at most 30
  iterations per phase. The U = 0 phase reaches the cap, the prior enters
  at iteration 31 and λ3 is set three times. The fit takes ~6 s on an M4
  Max (~11 s with loading). On fold 0, 23 of 32 LVs use a gene set and 13
  are annotated.
- **`tcga.yaml`** uses the defaults above. Its full-length runtime is
  unmeasured. On fold 0 (k = 492), the spectrum takes 27 s, the SVD 5 s
  and each iteration ~0.7 s, the U step included. At the cap of 300
  iterations per phase that is under 10 minutes per fold.

Any field can be overridden on the command line, e.g. `--data.fold 3`.

A fit writes three files:
- `model.npz`:
  - gene IDs and names, and the training means, SDs and ranges;
  - Z, B and U, and C with and without the held-out genes;
  - the singular values (all of them when the rule chose k), the number
    of components the rule kept and the Gavish–Donoho threshold;
  - the λs, the iteration at which the prior entered, and the ‖ΔB‖/‖B‖
    and objective traces;
  - the unmapped symbols and the hyperparameters.
- `annotations.tsv`: `gene_set`, `lv`, `u`, `auc`, `p_value`, `fdr`.
- `config.yaml`, which `plier-embed` reads to reload the same data.

## Tests

`uv run pytest plier/tests` runs in a few seconds:

- **Solver** (`test_solver.py`), from the paper's definitions, on
  synthetic data. Some tests use a smaller version of the PLIER preprint's
  simulation: Gamma loadings and Beta LVs, with gene sets from each LV's
  top genes plus random sets.
  - the B step is the ridge minimizer, and the Z step the unconstrained
    minimizer with its negatives set to 0;
  - the U step meets the non-negative lasso's KKT conditions and agrees
    with scikit-learn's `Lasso` fit on C itself;
  - an LV uses a gene set exactly below its λmax;
  - λ3 meets `frac`, and a larger λ3 gives a sparser U;
  - the Gavish–Donoho constants (λ*(1) = 4/√3, ω(1) ≈ 2.858), the elbow,
    and k = 2 × the components on a low-rank matrix, by either rule;
  - k is capped at the rank;
  - λ1 and λ2 come from d_k;
  - Z ≥ 0 and U ≥ 0, the objective falls, and the U and B steps never
    raise it;
  - the prior enters after the no-prior fit converges, whose iterates it
    repeats;
  - the simulated gene sets are recovered with AUC > 0.9;
  - the projection formula;
  - a state round trip;
  - the held-out fifth and the annotations' AUC, FDR and threshold.
- **Scaler** (`test_scaler.py`): training statistics, dropped constant
  genes, and clipping.
- **Prior** (`test_prior.py`): symbol mapping and the unmapped count. GMT
  reading and MSigDB fetching are tested in `shared/tests/test_genesets.py`.
- **Pipeline** (`test_pipeline.py`), on the miniature dataset, with the
  prior entering:
  - the fitted statistics, λ3, U, the held-out genes and the annotations
    among them, are unchanged when validation and test rows are rewritten
    (`scramble_held_out`);
  - genes constant over the training rows are dropped;
  - one projection embeds every split;
  - held-out values are clipped to the training range;
  - a save/load round trip, U and the annotations included.
- **CLI** (`test_cli.py`):
  - every config names only MSigDB collections `genesets` knows, fits with
    the prior entering, and embeds to a file `read_embeddings` accepts;
  - `--data.fold`: the model's means are that fold's training means;
  - `plier-embed` gives training samples the same embedding when held-out
    rows are rewritten (`assert_embedding_ignores_held_out`);
  - the no-prior variant.

Sources read for the solver: the PLIER author manuscript
([PMC7262669](https://pmc.ncbi.nlm.nih.gov/articles/PMC7262669/)), its
Methods and main text; the PLIER preprint, bioRxiv
[10.1101/116061](https://doi.org/10.1101/116061) v2, for the simulation
and the no-prior comparison; and the MultiPLIER author manuscript
([PMC6538307](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538307/)) for the
projection. The PLIER supplement was not read: PMC served a CAPTCHA.

## Citations

BibTeX for PLIER and MultiPLIER, the papers the solver is written from.

```bibtex
@article{mao2019pathwaylevel,
    title     = {Pathway-level information extractor (PLIER) for gene expression data},
    author    = {Mao, Weiguang and Zaslavsky, Elena and Hartmann, Boris M. and Sealfon, Stuart C. and Chikina, Maria},
    journal   = {Nature Methods},
    volume    = {16},
    number    = {7},
    pages     = {607--610},
    year      = {2019},
    publisher = {Springer Science and Business Media LLC},
    doi       = {10.1038/s41592-019-0456-1},
    url       = {https://doi.org/10.1038/s41592-019-0456-1}
}
```

```bibtex
@article{taroni2019multiplier,
    title     = {MultiPLIER: A Transfer Learning Framework for Transcriptomics Reveals Systemic Features of Rare Disease},
    author    = {Taroni, Jaclyn N. and Grayson, Peter C. and Hu, Qiwen and Eddy, Sean and Kretzler, Matthias and Merkel, Peter A. and Greene, Casey S.},
    journal   = {Cell Systems},
    volume    = {8},
    number    = {5},
    pages     = {380--394.e4},
    year      = {2019},
    publisher = {Elsevier BV},
    doi       = {10.1016/j.cels.2019.04.003},
    url       = {https://doi.org/10.1016/j.cels.2019.04.003}
}
```
