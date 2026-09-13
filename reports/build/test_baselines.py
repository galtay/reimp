"""The baselines report fits, scores and builds on a dataset, and reuses what it computed."""

import json
import re

import baselines as report
import pytest

from reimp_shared.splits import N_FOLDS
from reimp_shared.testing import use_fake_dataset, write_fake_dataset


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    root = write_fake_dataset(tmp_path / "dataset")
    use_fake_dataset(monkeypatch, root)
    return root


def _payload(page: str) -> dict:
    found = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S
    )
    return json.loads(found.group(1))


def test_report_builds_and_reuses_its_scores(fake_dataset, tmp_path) -> None:
    work = tmp_path / "work"
    options = {"work": work, "dims": (2, 4), "hvg_genes": 8, "salts": ("alt",), "bootstrap": 10}
    page = report.build(tmp_path / "report.html", **options).read_text()
    assert "{{" not in page
    assert "__REPORT_" not in page
    data = _payload(page)
    assert (data["dims"], data["head"], data["hvg_genes"]) == ([2, 4], 4, 8)
    groups = {m["group"] for m in data["metrics"]}
    assert {"classification", "expression", "geometry", "confounders"} <= groups
    for metric in data["metrics"]:
        assert [p["dim"] for p in metric["sweep"]] == [2, 4]
        assert metric["sweep"][-1]["value"] == metric["value"]
        assert [f["label"] for f in metric["folds"]] == [f"fold {k}" for k in range(N_FOLDS)]
        assert [s["label"] for s in metric["salts"]] == ["salt alt"]
    assert any(f["value"] is not None for m in data["metrics"] for f in m["folds"])
    assert any(m["hvg"]["value"] is not None for m in data["metrics"])
    # Genes minus components, with paired intervals, wherever both have replicates.
    deltas = [m["delta"] for m in data["metrics"] if m["delta"] is not None]
    assert deltas and all(d["lo"] <= d["hi"] for d in deltas)

    scores = sorted((work / "scores").glob("*.json"))
    expected = ["hvg8.json", "pca2.json", "pca4.json", "pca4_folds.json", "pca4_salt-alt.json"]
    assert [p.name for p in scores] == expected
    written = {p: p.stat().st_mtime_ns for p in scores}
    report.build(tmp_path / "again.html", **options)
    assert {p: p.stat().st_mtime_ns for p in scores} == written


def test_score_files_scored_differently_do_not_mix(tmp_path) -> None:
    for name, salt in [("a", "one"), ("b", "two")]:
        (tmp_path / f"{name}.json").write_text(json.dumps({"meta": {"salt": salt}}))
    with pytest.raises(ValueError, match="salt"):
        report.Scores([tmp_path / "a.json", tmp_path / "b.json"])
