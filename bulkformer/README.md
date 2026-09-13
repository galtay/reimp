# bulkformer (stub)

BulkFormer, from Kang et al. (Cell Systems 2026):
[KangBoming/BulkFormer](https://github.com/KangBoming/BulkFormer).

Not yet reimplemented. [`paper.md`](paper.md) records what the paper did —
model, data, evaluations and published numbers — as the reference for the
reimplementation and for which evaluations `reimp_shared.eval` adopts.
When code lands, this directory becomes a workspace member like `txfm/`.

What sets it apart: a graph layer over a gene co-expression network ahead
of Performer attention, over all ~20k genes, trained by masked-value
regression. Three decisions carry into the reimplementation:

- The co-expression graph is built per fold from training samples. The
  released graph matches TCGA co-expression computed over every patient,
  test patients included, so neither it nor the published weights are used.
- A scaled-down model (d = 256, one graph layer, four Performer layers):
  the 132M-parameter release is sized for 0.5M profiles, not ~8,000.
- The embedding pools the final gene tokens (max, as in the paper), without
  the per-sample depth and detection statistics the current code appends.

The experiment it makes cheap: a graph ablation — no graph, a random
graph, training-fold co-expression, protein interactions — scored by the
shared probes.
