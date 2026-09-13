# MOJO — what the paper did

Gélard, Benkirane, Pierrot, Richard, Cournède. *Bimodal masked language
modeling for bulk RNA-seq and DNA methylation representation learning*.
bioRxiv [10.1101/2025.06.25.661237](https://doi.org/10.1101/2025.06.25.661237):
v1 27 June 2025 (the repo gives its venue as the ICML 2025 GenBio
workshop), v2 29 May 2026. Only third-party pages claim ICML 2026
acceptance (unverified). Code:
[instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research),
`mojo/` (CC BY-NC-SA 4.0), the same repository as BulkRNABert. Weights:
[InstaDeepAI/MOJO](https://huggingface.co/InstaDeepAI/MOJO) (52.3M
parameters).

Read: v1 and v2 in full, the repo's MOJO code and the released
`config.json`. bioRxiv serves every table as an image, which could not be
downloaded, so the only numbers below are those the prose restates.

## Model

- **Tokens**: one RNA token and one methylation token per gene, each
  embedded (d = 256) and summed with a gene embedding initialized from
  Gene2Vec. RNA uses BulkRNABert's tokenizer: 64 linear bins of
  log10(1 + TPM) divided by a dataset maximum (5.528 here, 5.547 in
  BulkRNABert). Methylation is the mean 450k beta over CpGs within
  ±1.5 kb of the TSS or in the gene body.
- **Genes**: BulkRNABert's list less the genes without methylation
  probes, 17,116, padded to 17,152.
- **Backbone** (a U-Net, from `config.json`):
  - a convolutional stem (kernel 15);
  - 8 convolution blocks, each halving the length: 17,152 positions down
    to 67, at 512 channels;
  - 8 transformer layers over those 67 positions (16 heads, pre-LN,
    rotary position embeddings, SwiGLU FFN of width 1,024);
  - a deconvolution tower back up to one position per gene, with skip
    connections from the downsampling blocks;
  - one masked-token head per modality.
- **Objective**: masked language modelling, 15% of tokens selected and
  replaced 80/10/10 mask/random/kept; one cross-entropy per modality,
  summed; 192B tokens. v1 adds a mutual-information loss that makes the
  representation robust to a missing modality.
- **Missing modality**: fed as all `<MASK>` tokens. An extended
  pretraining adds 2,022 RNA-only and 560 methylation-only samples.
- **Sample embedding**: the mean over the 67 pooled positions of the last
  transformer layer, 512-d.
- **Cost**: attention runs over 67 positions, not 17k genes — about 100×
  faster per step than BulkRNABert-style attention over every gene, and
  300× faster than a pure-transformer MOJO (v2). The convolutions run over
  the gene list in an arbitrary order; ordering genes by genome position
  changed nothing within error (v2 appendix A.5).

## Pretraining data

TCGA samples with both RNA-seq and 450k methylation (9,252 paired
samples). v2's limitations section: "pre-trained on the entirety of the
paired TCGA cohort", so, as with BulkRNABert, the test samples were seen
in pretraining. The "5% kept for testing" does not state its unit.

## Evaluations as published

Downstream tasks use only the paired samples, 80/20 × 5 seeds, stratified
by cohort (survival: by cohort and event).

1. **Pan-cancer classification**, weighted-F1 (v1 prose): both
   modalities 0.952; methylation dropped at test time 0.854, or 0.937 with
   the mutual-information loss. BulkRNABert on RNA alone: 0.943.
2. **Survival**: a pooled pan-cancer C-index (about 0.77, derived from
   v2's "recovers 97%"), and per-cohort C-indexes weighted by test-set
   size. Endpoint "diagnosis until death", source not stated. External
   cohorts: ICGC 0.823 ± 0.019 against BulkRNABert 0.805 ± 0.036; TARGET
   0.601 ± 0.023, level with BulkRNABert.
3. **v2 additions**: TARGET and ICGC external cohorts; transfer to the
   27k methylation array; scGPT, SeNMo, GAT and MultiSurv baselines;
   pure-transformer, gene-order and tokenization ablations; ovarian
   subtyping fine-tuned on RNA alone, where MOJO beats BulkRNABert (a
   figure only, numbers unverified).

## For reimp

The multimodal part is out of scope: the dataset has no methylation.
What remains on RNA alone is still unlike anything else here, which is why
this directory exists: **attention over a few dozen learned, pooled
positions rather than over genes**. BulkRNABert attends over every gene
token, TxFM over a random subset of 2,048, BulkFormer over all of them
through linear attention and a gene graph; MOJO convolves the gene
sequence down to 67 positions first and decodes back up for the masked
tokens.

**Method-defining**, kept:
- BulkRNABert's tokenizer on TPM (64 linear bins of normalized log TPM),
  so the two differ in backbone alone.
- The U-Net: convolutional stem, halving convolution blocks, transformer
  layers over the pooled positions, a deconvolution tower with skips.
- Masked language modelling on the RNA tokens, 15%, 80/10/10.
- The embedding: the mean over pooled positions of the last transformer
  layer.

**Incidental**, standardized:
- Methylation, its token embedding and the mutual-information loss: no
  methylation in the data.
- Gene2Vec: gene embeddings learned from scratch, as in `bulkrnabert/`.
- Genes: the default 19,944 protein-coding genes, padded to a multiple of
  2^(number of halving blocks); dataset order, with genome order as a
  variant (their ablation found no difference).
- Pretraining on each fold's training patients only, and the tokenizer's
  maximum from training samples; the released weights, transductive and
  bimodal, are not used.
- Size scaled to ~8,000 training samples per fold; 52M parameters is a
  starting point, not a target.

**Evaluation ideas**: none new beyond BulkRNABert's. Scored beside
`bulkrnabert/`, it isolates what the pooled-attention backbone changes
under the same tokens and objective.
