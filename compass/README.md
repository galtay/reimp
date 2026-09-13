# compass (stub)

COMPASS, from Shen et al. (Nature Medicine 2026):
[mims-harvard/COMPASS](https://github.com/mims-harvard/COMPASS).

Not yet reimplemented. [`paper.md`](paper.md) records what the paper did —
model, data, evaluations and published numbers — as the reference for the
reimplementation and for which evaluations `reimp_shared.eval` adopts.
When code lands, this directory becomes a workspace member like `txfm/`.

A structured bottleneck: gene tokens pass one transformer layer and are
read out through 132 literature gene sets into 43 immune and
tumour-microenvironment concepts, trained self-supervised with a triplet
loss on the concept vector. The pretraining is TCGA-only and
reimplementable; the immunotherapy-response evaluations need cohorts we do
not have. Two changes carry into the reimplementation:

- The cancer-type input token is dropped: it would hand the cancer-type
  probes their label.
- The released weights saw every TCGA tumour patient, so the model is
  retrained per fold.

Read it beside a 43-component PCA and a training-free score of the same
gene sets, which separate what training adds from what the prior alone
gives. About half its concept genes are also in MSigDB Hallmark sets, so
part of any advantage on the pathway probe is overlap.
