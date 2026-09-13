"""The data report builds from a dataset and states every number it promises."""

import json
import re

import pytest
import shared_data as report

from reimp_shared import hub
from reimp_shared.splits import N_FOLDS, SPLITS
from reimp_shared.testing import use_fake_dataset, write_fake_dataset


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    root = write_fake_dataset(tmp_path / "dataset")
    use_fake_dataset(monkeypatch, root)
    return root


def test_report_builds_from_the_dataset(fake_dataset, tmp_path) -> None:
    page = report.build(tmp_path / "report.html").read_text()
    assert "{{" not in page
    assert "__REPORT_DATA__" not in page
    payload = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S
    )
    data = json.loads(payload.group(1))

    samples = hub.load_samples()
    assert sum(p["samples"] for p in data["cohort"]["projects"]) == len(samples)
    assert len(data["folds"]["layout"]) == N_FOLDS
    for size in data["folds"]["sizes"]:
        assert sum(size[f"{split}_samples"] for split in SPLITS) == len(samples)
    patients = sum(sum(p["folds"]) for p in data["balance"]["projects"])
    assert patients == samples["case_submitter_id"].nunique()


def test_every_placeholder_needs_a_value() -> None:
    with pytest.raises(KeyError, match="missing_number"):
        report.render({}, {}, "<p>{{missing_number}}</p>")
