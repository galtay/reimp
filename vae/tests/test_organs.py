import pytest

from reimp_shared.eval import ORGAN_TASKS
from reimp_vae.organs import PROJECT_ORGAN, label_classes, sample_labels

TCGA_PROJECTS = (
    "TCGA-ACC",
    "TCGA-BLCA",
    "TCGA-BRCA",
    "TCGA-CESC",
    "TCGA-CHOL",
    "TCGA-COAD",
    "TCGA-DLBC",
    "TCGA-ESCA",
    "TCGA-GBM",
    "TCGA-HNSC",
    "TCGA-KICH",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-LAML",
    "TCGA-LGG",
    "TCGA-LIHC",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-MESO",
    "TCGA-OV",
    "TCGA-PAAD",
    "TCGA-PCPG",
    "TCGA-PRAD",
    "TCGA-READ",
    "TCGA-SARC",
    "TCGA-SKCM",
    "TCGA-STAD",
    "TCGA-TGCT",
    "TCGA-THCA",
    "TCGA-THYM",
    "TCGA-UCEC",
    "TCGA-UCS",
    "TCGA-UVM",
)


def test_every_tcga_project_has_an_organ() -> None:
    assert len(TCGA_PROJECTS) == 33
    assert sorted(PROJECT_ORGAN) == sorted(TCGA_PROJECTS)


def test_within_organ_probe_groups_each_share_one_organ() -> None:
    """The supervised label ties inside each within-organ task, keeping those probes honest."""
    organs = [{PROJECT_ORGAN[p] for p in projects} for projects in ORGAN_TASKS.values()]
    assert all(len(group) == 1 for group in organs)
    assert len(set.union(*organs)) == len(ORGAN_TASKS)


def test_organs_merge_only_projects_from_one_organ() -> None:
    classes = label_classes("organ")
    assert classes == sorted(classes)
    assert len(classes) == 26
    shared = {o for o in classes if list(PROJECT_ORGAN.values()).count(o) > 1}
    assert shared == {"lung", "kidney", "colorectal", "brain", "uterus", "adrenal_gland"}


def test_classes_per_supervision() -> None:
    assert label_classes("none") == []
    assert label_classes("project") == sorted(TCGA_PROJECTS)


def test_labels_follow_the_project_alone() -> None:
    """A normal sample gets its project's organ: labels see nothing but the project."""
    organ = sample_labels(["TCGA-LUAD", "TCGA-LUSC", "TCGA-KIRC", "TCGA-LUAD"], "organ")
    classes = label_classes("organ")
    assert [classes[i] for i in organ] == ["lung", "lung", "kidney", "lung"]
    project = sample_labels(["TCGA-LUAD", "TCGA-LUSC"], "project")
    assert project[0] != project[1]
    assert organ.dtype.name == "int64"


def test_a_replacement_mapping(project_organ) -> None:
    assert label_classes("organ", project_organ) == ["kidney", "lung"]
    labels = sample_labels(["TCGA-CCC", "TCGA-AAA", "TCGA-BBB"], "organ", project_organ)
    assert labels.tolist() == [0, 1, 1]


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="missing"):
        sample_labels(["TCGA-LUAD", "TCGA-XXX"], "organ")
    with pytest.raises(ValueError, match="without supervision"):
        sample_labels(["TCGA-LUAD"], "none")
    with pytest.raises(ValueError, match="unknown supervision"):
        label_classes("tissue")


@pytest.mark.network
def test_every_project_in_the_dataset_has_an_organ() -> None:
    from reimp_shared import hub

    assert set(hub.load_samples()["project_id"]) == set(PROJECT_ORGAN)
