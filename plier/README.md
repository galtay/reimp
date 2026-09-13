# reimp-plier

PLIER — Mao et al., *Pathway-level information extractor (PLIER) for gene
expression data*, Nature Methods 2019
([wgmao/PLIER](https://github.com/wgmao/PLIER)) — and its transfer use,
MultiPLIER — Taroni et al., Cell Systems 2019
([greenelab/multi-plier](https://github.com/greenelab/multi-plier)).

A matrix factorization whose gene loadings are pulled toward sparse,
non-negative combinations of curated gene sets. A sample's embedding is a
fixed ridge projection onto those loadings, the same for training and
held-out samples, which is how MultiPLIER projects new data. It is linear,
runs on a CPU, and is the only model here built on prior knowledge. The
code is a numpy port of the R package's `PLIER()` (`R/Allfuncs.R` at
`fe4e9b2`), with scikit-learn's elastic net for U. There is no neural net,
so no Lightning and no GPU.

```bash
uv run plier fit --config plier/configs/debug.yaml                     # ~15 s on real data
uv run plier-embed --model runs/plier_debug/fold0 --out out/plier_debug/fold0.parquet
uv run plier fit --config plier/configs/tcga.yaml --data.fold 0        # and so on for folds 1-4
uv run plier-embed --model runs/plier/fold0 --out out/plier/fold0.parquet
uv run plier fit --config plier/configs/tcga.yaml --prior null --out_dir runs/plier_noprior
```

Run from the repository root: the configs name the prior as
`plier/priors/recommended.gmt`. [`paper.md`](paper.md) records what the
papers did and, under "For reimp", which of their choices are kept. Below
is how this reimplementation follows it.

## From the paper and the package

Y is genes × samples, z-scored per gene; C is the binary genes × gene-sets
prior.

| | |
|---|---|
| objective | ‖Y − ZB‖² + λ1‖Z − CU‖² + λ2‖B‖² + λ3‖U‖₁ with Z ≥ 0, U ≥ 0 |
| start | SVD of Y, randomized at rank max(200, n/4) with 3 power iterations (full SVD at ≤ 500 samples); B = (VD)ᵀ[:k], Z its ridge fit; a Z column with no positive entry is sign-flipped, then Z is clipped at 0 |
| Z step | (YBᵀ + λ1 CU)(BBᵀ + λ1 I)⁻¹, negatives set to 0: a projection, not a constrained solve |
| B step | (ZᵀZ + λ2 I)⁻¹ZᵀY |
| U step | from iteration 20 (U = 0 before). Per LV, a non-negative elastic net (α = 0.9, with intercept, not standardized) of Z[:, j] on its candidate sets. Candidates: the top `maxPath` = 10 sets per LV, ranked by the ridge regression Ĉ Z (pseudo-inverse of CᵀC, α = 5), pooled over LVs (`pathwaySelection = "complete"`) |
| λ3 | every 20 iterations, the value on glmnet's path exp(−4) … exp(−12) (steps of 0.125) at which the share of LVs using a gene set is nearest `frac` = 0.7 |
| k | 2 × `num.pc`, capped at 0.9 × samples. `num.pc` is the elbow of the Tukey-smoothed second differences of the singular values (R's `smooth`, ported and checked against R) |
| λ1, λ2 | d_k / 2 and d_k, from the k-th singular value |
| stopping | ‖ΔB‖²/‖B‖² < 1e-6, or once that ratio stops falling (the package's rule); at most 350 iterations |
| gene sets | sets with fewer than 10 genes are zeroed |
| annotations | a fifth of each set's genes are held out of C. Each positive U[j, i] is scored by a Wilcoxon AUC: set j's held-out genes against the genes in none of LV i's sets, on Z[:, i], with BH FDR. "Annotated" means AUC > 0.7 and FDR < 0.05. This labels LVs; it does not change B |
| embedding | B = (ZᵀZ + λ2 I)⁻¹Zᵀy for every sample |

## Ours

Following paper.md's "For reimp":

- **Input**: log1p library-normalized unstranded counts (`lognorm`, library
  1e5) on the 19,944 protein-coding genes, as for the PCA baseline. The
  closer-to-paper z-scored RPKM is
  `--data.quantification fpkm_unstranded --data.transform none`.
- **One model per fold, trained on that fold's training samples** (8,304 on
  fold 0), not on recount2. Every statistic is fit on those samples alone
  (`shared/EVALS.md`, rule 3), and a test checks it:
  - per-gene means and SDs (n − 1, as R's `sd`);
  - which genes are dropped as constant (367 on fold 0);
  - the SVD, `num.pc` and k;
  - λ1, λ2 and λ3;
  - which genes are held out.

  Validation and test samples are z-scored with the training means and SDs,
  then projected. MultiPLIER instead z-scores each target dataset on itself,
  which would recentre every test fold.
- **`allGenes = TRUE`**: every protein-coding gene that varies over the
  training samples is modelled. Genes in no set get empty rows of C.
  `--all_genes false` is the package default, prior genes only.
- **k**: the package rule by default. On fold 0 it gives `num.pc` = 323,
  so k = 646 and λ2 = d_646 ≈ 141. The fixed-k variants are
  `--model.k 64` and `--model.k 256`, to sit beside PCA at equal dimension.
- **The no-prior ablation**, `--prior null`, runs the same solver with
  U = 0 (λ1‖Z‖² in place of λ1‖Z − CU‖², PLIERv2's "PLIERbase"). It keeps
  the same k, λ1 and λ2, so the difference between the two runs is what the
  prior buys.

### Prior

`priors/recommended.gmt` holds 260 sets on 4,932 HGNC symbols. It is
exported by `scripts/export_prior.R` (`Rscript plier/scripts/export_prior.R`,
base R and a network connection) from three matrices bundled with the
PLIER package at `wgmao/PLIER@fe4e9b2`, md5-checked:

| source object | sets |
|---|---|
| `bloodCellMarkersIRISDMAP`, IRIS and DMAP cell types | 61 |
| `svmMarkers`, CIBERSORT LM22 | 22 |
| `canonicalPathways` (MSigDB C2:CP) without the 252 REACTOME and 116 PID sets: 122 KEGG, 25 BioCarta, 30 others | 177 |

What the prior leaves out, and why:
- **Every collection the pathway probe scores**: Reactome, PID, oncogenic
  (C6), Hallmark and Cancer Cell Atlas. The probe then scores held-out
  collections, as MultiPLIER held out C6.
- **MSigDB's chemical and genetic perturbation sets (C2:CGP)**, some of
  which were derived from TCGA patients.

Some gene overlap with the probed collections remains. The prior's genes
cover 43% of Hallmark's, 30% of Reactome's and 53% of PID's (paper.md), so
say so beside pathway scores.

The canonical pathways are MSigDB content, KEGG included, redistributed by
the GPL package under MSigDB's terms.

**Mapping symbols to genes.** Symbols are matched exactly to GENCODE v36
`gene_name`, with no alias rescue. Of the 4,932 symbols:
- 231 name no protein-coding gene. 204 of those name no gene at all,
  mostly symbols renamed since the package was built (e.g. `ATP5A1`, now
  `ATP5F1A`).
- On fold 0, 5 more name genes that are constant over the training
  samples.

So 236 symbols are unmapped. After mapping, 256 of the 260 sets keep at
least 10 genes. The count is logged on every fit and saved in `model.npz`
as `unmapped`.

## Configs

| field | |
|---|---|
| `out_dir` | fold k's model is written to `<out_dir>/fold<k>/` |
| `prior` | a GMT file, relative to the repo root; `null` for the no-prior ablation |
| `all_genes` | `true`: every varying gene; `false`: the prior's genes only |
| `data.*` | `load_expression` arguments: `quantification`, `gene_types`, `transform`, `library_size`, `projects`, `fold`, `revision` |
| `model.*` | `PLIER` arguments: `k` (null: the rule), `k_multiplier`, `svd_rank` (null: the package's), `l1`, `l2`, `l3` (null: the rules above), `frac`, `max_iter`, `tol`, `prior_start`, `max_path`, `pathway_selection`, `glm_alpha`, `min_genes`, `seed` |

- **`debug.yaml`** uses fold 0's whole training set with the prior, but k = 32,
  SVD rank 200 and 60 iterations. That is enough for λ3 to be tuned at
  iterations 20, 40 and 60. It fits in ~12 s on an M4 Max. On fold 0,
  22 of 32 LVs use a gene set and 16 are annotated.
- **`tcga.yaml`** uses the package's defaults. Its full-length runtime is
  unmeasured. After iteration 20 every iteration fits 646 elastic nets, and
  every 20th iteration fits them along a 65-value path, so expect tens of
  minutes per fold.

Any field can be overridden on the command line, e.g. `--data.fold 3`.

A fit writes three files:
- `model.npz`: gene IDs and names, the training means and SDs, Z, B, U,
  C with and without the held-out genes, the singular values, the λs, the
  ‖ΔB‖²/‖B‖² trace, the unmapped symbols and the hyperparameters.
- `annotations.tsv`: `gene_set`, `lv`, `u`, `auc`, `p_value`, `fdr`.
- `config.yaml`, which `plier-embed` reads to reload the same data.

## Deviations from the R code

- **SVD**: scikit-learn's `randomized_svd` (10 oversamples, 3 power
  iterations, like `rsvd(q = 3)`), seeded by `seed` rather than R's
  `set.seed(123456)`. The leading singular values match. The tail that
  `num.pc` reads can differ slightly.
- **Elastic net**: scikit-learn's coordinate descent (`enet_path`, the
  solver behind `ElasticNet(l1_ratio = 0.9, positive = True)`), fit along
  the path with warm starts as glmnet does.
  - For the Gaussian family, glmnet divides y by its SD s and λ by s, which
    turns its ridge term into λ(1 − α)/(2s)‖β‖², not the λ(1 − α)/2‖β‖² its
    documentation states. `nonnegative_enet` does the same, so λ3 is on
    glmnet's scale. A test checks it against glmnet 5.0 to 1e-7; without the
    scaling, scikit-learn is 0.07 away on that problem.
  - Convergence is scikit-learn's duality gap at `tol` = 1e-7, where glmnet
    thresholds coefficient changes at 1e-7: close, not identical.
- **Wilcoxon p-values**: scipy's Mann–Whitney U, with R's switch between
  exact and normal approximation, checked against `wilcox.test`.
- **Guards the R code lacks**:
  - Genes constant over the training samples are dropped; R would produce
    NaNs.
  - k is capped at the rank of Y.
  - A fixed k raises the SVD rank to at least k, and a rule-chosen k above
    the SVD's rank triggers a recomputed SVD. R indexes past `d` and fails.
- **Not ported**:
  - `doCrossval = FALSE`'s pseudo-cross-validation (`getAUC`): annotations
    always use held-out genes, the package default.
  - `penalty.factor` (all ones, the default) and the random start (`rseed`).
- **Convergence before the prior enters**: as in R, the loop stops as soon
  as ‖ΔB‖²/‖B‖² < `tol`, even before iteration 20, which leaves U = 0. On
  the debug run the ratio was 1e-3 at iteration 20, so this did not happen.
  The log reports the number of LVs with a gene set every 20 iterations.
- **No end-to-end parity test against R PLIER** (paper.md's
  `dataWholeBlood` plan) in this pass. The pieces are tested against R
  instead: the smoother, `num.pc`, `wilcox.test`, `p.adjust` and glmnet's
  λ scale.

## Tests

`uv run pytest plier/tests` runs in a few seconds:

- **Solver steps** (`test_model.py`) on synthetic data:
  - the Z and B steps are the objective's minimizers, with Z clipped at 0;
  - the SVD start;
  - `pinv_ridge`;
  - `maxPath` candidate pools;
  - U ≥ 0 on candidate sets only;
  - λ3 from the path;
  - the prior entering at iteration 20;
  - the λ and k rules;
  - the held-out fifth and the annotations;
  - the no-prior ablation at the same k and λs;
  - the projection formula.
- **R reference values** (`test_smooth.py`): R's smoother and `num.pc`.
- **Prior** (`test_prior.py`): GMT I/O, symbol mapping and the unmapped
  count, and the exported prior's composition.
- **Pipeline** (`test_pipeline.py`), on the miniature dataset:
  - the fitted statistics are unchanged when validation and test values
    are replaced;
  - genes constant over the training rows are dropped;
  - one projection embeds every split;
  - a save/load round trip.
- **CLI** (`test_cli.py`): every config fits and embeds to a file
  `read_embeddings` accepts; `--data.fold`; the no-prior variant.
