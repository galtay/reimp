"""Evaluation of sample embeddings: one yardstick for every model.

A model is trained once per cross-validation fold, and each fold's model
writes its embeddings of every sample as a parquet of `sample_index`,
`embedding` and `fold` (see `write_embeddings`). Every evaluation joins
them to the dataset by `sample_index` and, as `protocol` plans it, fits
once per fold on that fold's train and val samples and pools the folds'
test predictions into one out-of-fold score — `cv`, over every patient,
when all five folds are there. Scores carry patient-bootstrap intervals,
so any two models — or a model and a baseline — are compared on the same
patients with the same readout.

- `classification_probe`  logistic regression onto sample labels
- `invertibility`         ridge regression back onto expression
- `pathway_probe`         ridge regression onto ssGSEA pathway scores, within cancer type
- `survival_probe`        stratified Cox regression onto a survival endpoint
                          (`survival_scores` adds each project's own C-index)
- `geometry_probe`        spectrum, neighbours and clusters against cancer type
- `confounder_probe`      library QC and batch structure within cancer type
- `paired_differences`    one embeddings' scores minus another's, paired intervals
"""

from reimp_shared.eval.baselines import hvg_embeddings, pca_embeddings
from reimp_shared.eval.bootstrap import bootstrap_weights, summarize, weighted_mean
from reimp_shared.eval.classification import (
    ORGAN_TASKS,
    TASKS,
    classification_probe,
    classification_scores,
    fit_linear_probe,
    task_labels,
)
from reimp_shared.eval.compare import paired_differences
from reimp_shared.eval.confounders import (
    batch_enrichment,
    centre_by_project,
    confounder_probe,
    within_project_r2,
)
from reimp_shared.eval.embeddings import read_embeddings, write_embeddings
from reimp_shared.eval.geometry import (
    clustering_agreement,
    geometry_probe,
    neighbour_precision,
    spectrum,
)
from reimp_shared.eval.invertibility import (
    INVERTIBILITY_TARGET,
    invertibility,
    invertibility_target,
    reconstruction_scores,
)
from reimp_shared.eval.pathways import pathway_probe, pathway_scores
from reimp_shared.eval.survival import (
    concordance,
    fit_cox,
    stratified_concordance,
    survival_predictions,
    survival_probe,
    survival_samples,
    survival_scores,
)

__all__ = [
    "INVERTIBILITY_TARGET",
    "ORGAN_TASKS",
    "TASKS",
    "batch_enrichment",
    "bootstrap_weights",
    "centre_by_project",
    "classification_probe",
    "classification_scores",
    "clustering_agreement",
    "concordance",
    "confounder_probe",
    "fit_cox",
    "fit_linear_probe",
    "geometry_probe",
    "hvg_embeddings",
    "invertibility",
    "invertibility_target",
    "neighbour_precision",
    "paired_differences",
    "pathway_probe",
    "pathway_scores",
    "pca_embeddings",
    "read_embeddings",
    "reconstruction_scores",
    "spectrum",
    "stratified_concordance",
    "summarize",
    "survival_predictions",
    "survival_probe",
    "survival_samples",
    "survival_scores",
    "task_labels",
    "weighted_mean",
    "within_project_r2",
    "write_embeddings",
]
