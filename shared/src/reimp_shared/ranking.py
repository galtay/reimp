"""Per-sample gene rankings, for models that read a sample as an ordered list of genes.

TifBERT tokenizes each sample as its genes sorted by a score and compares
several scores; rank-based models such as Geneformer and Cell2Sentence do
the same with others. A `GeneRanker` computes one. Scores that compare a
gene against the cohort need per-gene statistics, which `fit` measures on
the samples it is given — the training samples — so a test sample is
ranked by statistics it did not help set.

Scores, for a sample's value x_g of gene g, applied to values as given
(choose the quantification and transform in `reimp_shared.data`):

  expression   x_g
  z            (x_g − μ_g) / σ_g
  tfidf        x_g · idf_g, with the rarity weight idf_g from `idf_scheme`:
                 count    log(N / n_g), n_g the training samples with x > threshold
                 smooth   log((1 + N) / (1 + n_g)) + 1
                 entropy  log N − H_g, H_g the entropy of gene g's share
                          of its own total across the training samples
  cohort_l2    x_g / ||x_·g||₂ — what TifBERT's released code computes
               (sklearn's TfidfTransformer fit with genes as documents);
               its paper describes tfidf

The count form is text retrieval's, and on dense bulk data it degenerates:
a gene detected in every sample gets weight 0, whatever its expression.
The smoothed form never reaches 0. The entropy form needs no threshold,
is 0 only for a gene expressed identically everywhere, and equals the
count form for a gene uniform on n_g samples and absent from the rest.
A gene never expressed in the training samples gets weight 0 under every
scheme: there is nothing to say how rare it is.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

RankScore = Literal["expression", "z", "tfidf", "cohort_l2"]
IdfScheme = Literal["count", "smooth", "entropy"]
RANK_SCORES: tuple[str, ...] = ("expression", "z", "tfidf", "cohort_l2")
IDF_SCHEMES: tuple[str, ...] = ("count", "smooth", "entropy")


def _reciprocal(x: np.ndarray) -> np.ndarray:
    return np.divide(1.0, x, out=np.zeros_like(x), where=x > 0)


def inverse_document_frequency(
    values: np.ndarray, scheme: IdfScheme = "entropy", detection_threshold: float = 0.0
) -> np.ndarray:
    """Per-gene rarity weight across the samples in `values` (samples x genes)."""
    values = np.asarray(values, dtype=np.float64)
    n_samples = values.shape[0]
    total = values.sum(axis=0)
    if scheme == "entropy":
        share = values / np.where(total > 0, total, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.where(share > 0, share * np.log(share), 0.0).sum(axis=0)
        idf = np.maximum(np.log(n_samples) - entropy, 0.0)
    else:
        detected = (values > detection_threshold).sum(axis=0).astype(np.float64)
        if scheme == "count":
            idf = np.log(n_samples / np.maximum(detected, 1.0))
        elif scheme == "smooth":
            idf = np.log((1.0 + n_samples) / (1.0 + detected)) + 1.0
        else:
            raise ValueError(f"unknown idf scheme {scheme!r}; expected one of {IDF_SCHEMES}")
    return np.where(total > 0, idf, 0.0)


class GeneRanker:
    """A per-gene score fit on training samples; `order` ranks each sample's genes by it."""

    def __init__(
        self,
        score: RankScore = "tfidf",
        idf_scheme: IdfScheme = "entropy",
        detection_threshold: float = 0.0,
    ) -> None:
        if score not in RANK_SCORES:
            raise ValueError(f"unknown score {score!r}; expected one of {RANK_SCORES}")
        if idf_scheme not in IDF_SCHEMES:
            raise ValueError(f"unknown idf scheme {idf_scheme!r}; expected one of {IDF_SCHEMES}")
        self.score = score
        self.idf_scheme = idf_scheme
        self.detection_threshold = detection_threshold
        self.offset_: np.ndarray | None = None
        self.weight_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> GeneRanker:
        """Measure the per-gene statistics the score needs from `values` (samples x genes)."""
        values = np.asarray(values, dtype=np.float64)
        n_genes = values.shape[1]
        self.offset_, self.weight_ = np.zeros(n_genes), np.ones(n_genes)
        if self.score == "z":
            self.offset_ = values.mean(axis=0)
            self.weight_ = _reciprocal(values.std(axis=0))
        elif self.score == "tfidf":
            self.weight_ = inverse_document_frequency(
                values, self.idf_scheme, self.detection_threshold
            )
        elif self.score == "cohort_l2":
            self.weight_ = _reciprocal(np.sqrt((values**2).sum(axis=0)))
        return self

    def scores(self, values: np.ndarray) -> np.ndarray:
        """Each gene's score in each sample; genes with no weight score 0."""
        if self.weight_ is None:
            raise RuntimeError("call fit before scoring")
        return (np.asarray(values, dtype=np.float64) - self.offset_) * self.weight_

    def order(self, values: np.ndarray) -> np.ndarray:
        """Gene positions per sample, highest score first; ties keep gene order."""
        return np.argsort(-self.scores(values), axis=-1, kind="stable")
