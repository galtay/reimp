# mojo (stub)

MOJO, from InstaDeep (Gélard et al., bioRxiv 2025/2026):
[instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research),
the same repository as BulkRNABert.

Not yet reimplemented. [`paper.md`](paper.md) records what the paper did —
model, data, evaluations and published numbers — as the reference for the
reimplementation. When code lands, this directory becomes a workspace
member like `txfm/`.

MOJO models RNA and DNA methylation together; the dataset has no
methylation, so only its RNA half is reimplemented here. That half is
still distinct: BulkRNABert's tokens and masked-token objective, but a
U-Net backbone that convolves ~20k gene tokens down to a few dozen
positions before any attention and decodes back up to predict the masked
genes. Scored beside `bulkrnabert/`, it shows what that backbone changes
with the tokens and objective held fixed.
