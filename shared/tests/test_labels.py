import numpy as np
import pandas as pd
import pytest

from reimp_shared import hub, labels
from reimp_shared.testing import PROJECTS


def test_load_survival_reads_every_project(fake_dataset) -> None:
    survival = labels.load_survival()
    for endpoint in labels.SURVIVAL_ENDPOINTS:
        assert {f"{endpoint}_event", f"{endpoint}_time"} <= set(survival.columns)
    assert set(survival["project_id"]) == set(PROJECTS)
    assert survival["case_submitter_id"].is_unique


def test_load_survival_is_cached(fake_dataset, monkeypatch) -> None:
    first = labels.load_survival()

    def read_again(commit):
        pytest.fail("survival labels were read again instead of from the cache")

    monkeypatch.setattr(labels, "_patient_parquets", read_again)
    pd.testing.assert_frame_equal(labels.load_survival(), first)


def test_some_expression_cases_lack_labels_as_in_the_real_data(fake_dataset) -> None:
    """The fake dataset mimics the release gap between the two datasets."""
    expression_cases = set(hub.load_samples()["case_submitter_id"])
    labelled = set(labels.load_survival().dropna(subset=["pfi_event"])["case_submitter_id"])
    assert 0 < len(expression_cases - labelled) < 5


def test_load_ssgsea_is_a_table_of_aliquots_by_pathways(fake_dataset) -> None:
    scores = labels.load_ssgsea("hallmark")
    pathways = [c for c in scores.columns if c != "aliquot_id"]
    assert pathways == [f"HALLMARK_FAKE_{j}" for j in range(6)]
    assert scores["aliquot_id"].is_unique
    assert np.isfinite(scores[pathways].to_numpy()).all()
    # Every aliquot of the case missing from the patients dataset is missing here too.
    aliquots = set(hub.load_samples()["aliquot_id"])
    assert set(scores["aliquot_id"]) < aliquots
    assert 0 < len(aliquots - set(scores["aliquot_id"])) < 3


def test_load_ssgsea_rejects_unknown_collections() -> None:
    with pytest.raises(ValueError, match="unknown collection"):
        labels.load_ssgsea("kegg")


def test_technical_covariates_from_barcodes_and_read_tallies(fake_dataset) -> None:
    samples = hub.load_samples()
    covariates = labels.technical_covariates()
    assert covariates["sample_index"].tolist() == samples["sample_index"].tolist()
    assert covariates["tss"].tolist() == samples["case_submitter_id"].str[5:7].tolist()
    assert covariates["plate"].str.fullmatch(r"A\d{3}").all()
    assigned = hub.load_values().sum(axis=1)
    reads = assigned + samples[list(labels.UNASSIGNED)].sum(axis=1).to_numpy()
    np.testing.assert_allclose(10 ** covariates["log_reads"], reads)
    np.testing.assert_allclose(covariates["assigned_fraction"], assigned / reads)


def test_a_full_commit_hash_needs_no_lookup() -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    assert labels._resolve(commit) == commit
