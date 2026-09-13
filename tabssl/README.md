# tabssl (stub)

Self-supervised objectives for tabular data — SCARF, VIME and BYOL — as
Dradjat et al. (Bioinformatics 2025) adapt them to bulk expression:
[kdradjat/SSRL_RNAseq](https://github.com/kdradjat/SSRL_RNAseq). The link
printed in the paper is dead, and the released code does not run as is.

Not yet reimplemented. [`paper.md`](paper.md) records what the paper did —
model, data, evaluations and published numbers — as the reference for the
reimplementation and for which evaluations `reimp_shared.eval` adopts.
When code lands, this directory becomes a workspace member like `txfm/`.

One encoder, three objectives: a 4 × 256 MLP whose output is the
embedding, trained to match a corrupted copy of each sample (SCARF), to
recover corrupted genes and the corruption mask (VIME), or to predict an
averaged teacher network (BYOL). With the architecture fixed, what differs
is the objective alone. Planned beside them: the same encoder untrained,
since random features of 20k genes can already probe well.

In the paper, frozen embeddings — the setting reimp scores — trailed a
supervised MLP trained from scratch, and PCA was never tried.
