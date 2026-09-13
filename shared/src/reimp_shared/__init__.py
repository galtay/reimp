"""Shared ground for the reimp models: one dataset, one split, one yardstick.

- `hub`         the TCGA expression dataset on the HF Hub, as numpy / pandas
- `splits`      patient-level 5-fold cross-validation: train / val / test per fold
- `preprocess`  library-size normalization and gene selection
- `eval`        embeddings file format, linear probes, PCA baseline
"""
