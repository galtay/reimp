# plier (stub)

PLIER (Mao et al., Nature Methods 2019;
[wgmao/PLIER](https://github.com/wgmao/PLIER)) and its transfer use,
MultiPLIER (Taroni et al., Cell Systems 2019;
[greenelab/multi-plier](https://github.com/greenelab/multi-plier)).

Not yet reimplemented. [`paper.md`](paper.md) records what the papers did —
model, data, evaluations and published numbers — as the reference for the
reimplementation and for which evaluations `reimp_shared.eval` adopts.
When code lands, this directory becomes a workspace member like `txfm/`.

A matrix factorization whose gene loadings are pulled toward sparse
combinations of curated gene sets; a sample's embedding is a fixed ridge
projection onto those loadings, the same for training and held-out
samples. It is linear and cheap (minutes per fold on a CPU), and it is the
only model here built on prior knowledge. Three decisions carry into the
reimplementation, a numpy port of the R code:

- The prior is cell-type markers plus KEGG, BioCarta and other canonical
  pathways. It leaves out every collection the pathway probe scores
  (Hallmark, Reactome, PID, oncogenic, Cancer Cell Atlas), so the probe
  scores held-out collections, and it leaves out MSigDB's chemical and
  genetic perturbation sets, some of which were derived from TCGA patients.
- Held-out samples are z-scored with training means and SDs, not on
  themselves as MultiPLIER does.
- The same solver without the prior, at the same k, is run beside it: the
  difference is what the prior buys.
