# vae (stub)

Autoencoders with a regularized latent, two published configurations of
one small package:

- **Tybalt** (Way and Greene, PSB 2018;
  [greenelab/tybalt](https://github.com/greenelab/tybalt)): one dense layer
  to a 100-d latent, KL-regularized, over the 5,000 most variable genes.
  [`paper.md`](paper.md), with the BioBombe follow-up that compares it with
  PCA, ICA and NMF across latent sizes.
- **Tissue-supervised autoencoder** (Pande, Uyar and Akalin, bioRxiv 2026;
  [BIMSBbioinfo/flexynesis_tissue_vae_manuscript](https://github.com/BIMSBbioinfo/flexynesis_tissue_vae_manuscript)):
  MMD-regularized rather than KL, a 121-d latent, and a tissue classifier
  trained jointly on the latent. [`tissue_vae.md`](tissue_vae.md).

Not yet reimplemented. The two summaries record what each paper did —
model, data, evaluations and published numbers — as the reference for the
reimplementation and for which evaluations `reimp_shared.eval` adopts.
When code lands, this directory becomes a workspace member like `txfm/`.

Tybalt's main use is a like-for-like comparison with PCA at matched latent
sizes. On TCGA alone, tissue supervision becomes cancer-type supervision,
which the cancer-type probes would then read back, so the supervised
configuration comes in variants: `none` (the unsupervised reference),
`organ` (closest to the paper; the within-organ probes stay an honest test)
and `project`. Supervised variants are reported as such and never ranked
with unsupervised models.
