"""Labels for the supervised MMD-AE: a fixed TCGA project -> organ mapping.

Pande et al. supervise the latent with a tissue (a UBERON class), and TCGA
tumours sit in organ classes there: LUAD and LUSC are both lung; KICH, KIRC
and KIRP all kidney; GBM and LGG both brain. On TCGA alone the closest
label is a fixed map from each project to the organ it was sampled from.
It depends on no cohort statistic, so it is the same in every fold, and a
normal sample gets its project's organ, as GTEx normals got theirs in the
paper, which leaves tumour vs normal unsupervised.

Beyond the four within-organ groups the shared probes test (lung, kidney,
colorectal, glioma), two more pairs share an organ: UCEC and UCS (both
corpus uteri) and ACC and PCPG (adrenal cortex and, for most PCPG tumours,
adrenal medulla). Every other project is its own organ: 26 classes.

  none      no labels; the head is removed (the unsupervised reference)
  organ     one of the 26 organs, closest to the paper
  project   the 33 projects, i.e. cancer-type supervision
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np

Supervision = Literal["none", "organ", "project"]
SUPERVISIONS: tuple[str, ...] = ("none", "organ", "project")

PROJECT_ORGAN: dict[str, str] = {
    "TCGA-ACC": "adrenal_gland",
    "TCGA-BLCA": "bladder",
    "TCGA-BRCA": "breast",
    "TCGA-CESC": "cervix",
    "TCGA-CHOL": "bile_duct",
    "TCGA-COAD": "colorectal",
    "TCGA-DLBC": "lymph_node",
    "TCGA-ESCA": "esophagus",
    "TCGA-GBM": "brain",
    "TCGA-HNSC": "head_and_neck",
    "TCGA-KICH": "kidney",
    "TCGA-KIRC": "kidney",
    "TCGA-KIRP": "kidney",
    # Sampled as peripheral blood ("Primary Blood Derived Cancer").
    "TCGA-LAML": "blood",
    "TCGA-LGG": "brain",
    "TCGA-LIHC": "liver",
    "TCGA-LUAD": "lung",
    "TCGA-LUSC": "lung",
    "TCGA-MESO": "pleura",
    "TCGA-OV": "ovary",
    "TCGA-PAAD": "pancreas",
    "TCGA-PCPG": "adrenal_gland",
    "TCGA-PRAD": "prostate",
    "TCGA-READ": "colorectal",
    "TCGA-SARC": "soft_tissue",
    "TCGA-SKCM": "skin",
    "TCGA-STAD": "stomach",
    "TCGA-TGCT": "testis",
    "TCGA-THCA": "thyroid",
    "TCGA-THYM": "thymus",
    "TCGA-UCEC": "uterus",
    "TCGA-UCS": "uterus",
    "TCGA-UVM": "eye",
}


def check_supervision(supervision: str) -> None:
    """Raise unless `supervision` is one of `SUPERVISIONS`."""
    if supervision not in SUPERVISIONS:
        raise ValueError(f"unknown supervision {supervision!r}; expected one of {SUPERVISIONS}")


def label_classes(
    supervision: Supervision, project_organ: Mapping[str, str] | None = None
) -> list[str]:
    """The class names a supervised head predicts, in label order; empty for `none`.

    `project_organ` replaces `PROJECT_ORGAN`, e.g. for a dataset with other
    projects; `project` supervision takes its keys as the classes.
    """
    check_supervision(supervision)
    mapping = PROJECT_ORGAN if project_organ is None else project_organ
    if supervision == "organ":
        return sorted(set(mapping.values()))
    if supervision == "project":
        return sorted(mapping)
    return []


def sample_labels(
    projects: Iterable[str],
    supervision: Supervision,
    project_organ: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Each sample's class index (int64) from its project, whatever its sample type."""
    classes = label_classes(supervision, project_organ)
    if not classes:
        raise ValueError("no labels without supervision")
    mapping = PROJECT_ORGAN if project_organ is None else project_organ
    projects = list(projects)
    missing = sorted(set(projects) - set(mapping))
    if missing:
        raise ValueError(f"projects missing from the project -> organ mapping: {missing}")
    index = {name: i for i, name in enumerate(classes)}
    names = [mapping[p] for p in projects] if supervision == "organ" else projects
    return np.array([index[name] for name in names], dtype=np.int64)
