import json

import pytest

from reimp_shared import hub
from reimp_shared.cli import main
from reimp_shared.eval import read_embeddings


def test_splits(fake_dataset, capsys) -> None:
    main(["splits"])
    out = capsys.readouterr().out
    for fold in range(5):
        assert f"fold{fold}" in out
    main(["splits", "--fold", "2"])
    out = capsys.readouterr().out
    for split in ["train", "val", "test"]:
        assert split in out


def test_baseline_then_probe(fake_dataset, tmp_path, capsys) -> None:
    embeddings = tmp_path / "pca"
    scores = tmp_path / "scores.json"
    main(["baseline-pca", "--out", str(embeddings), "--n-components", "4"])
    assert sorted(p.name for p in embeddings.iterdir()) == [f"fold{k}.parquet" for k in range(5)]
    main(["probe", str(embeddings), "--json", str(scores), "--bootstrap", "20"])
    out = capsys.readouterr().out
    for column in [
        "tumor_vs_normal",
        "weighted_f1",
        "r2_pooled",
        "c_index_macro",
        "precision_at_1",
        "plate_enrichment",
        "r2_pathway",
    ]:
        assert column in out
    written = json.loads(scores.read_text())
    meta = written.pop("meta")
    assert meta["dataset"].endswith(f"@{hub.commit()}")
    assert (meta["bootstrap"], meta["endpoint"], meta["pathways"]) == (20, "pfi", "hallmark")
    tables = {"classification", "invertibility", "pathways", "survival", "geometry", "confounders"}
    assert set(written) == tables | {"survival_projects"}
    assert "accuracy_lo" in written["classification"][0]
    # Replicates stay out of the file unless asked for.
    assert not any(k.endswith("_boot") for k in written["classification"][0])
    assert {row["project"] for row in written["survival_projects"]} <= set(
        hub.load_samples()["project_id"]
    )
    for table in written.values():
        assert {row["split"] for row in table} == {"cv"}
        assert {row["embeddings"] for row in table} == {"pca"}
    assert {row["endpoint"] for row in written["survival"]} == {"pfi"}


def test_two_baselines_differenced_with_paired_intervals(fake_dataset, tmp_path, capsys) -> None:
    pca, hvg, scores = tmp_path / "pca", tmp_path / "hvg", tmp_path / "scores.json"
    main(["baseline-pca", "--out", str(pca), "--n-components", "4"])
    main(["baseline-hvg", "--out", str(hvg), "--n-genes", "8"])
    _, embedding, _ = read_embeddings(hvg / "fold0.parquet")
    assert embedding.shape[1] == 8
    main(
        ["probe", str(pca), str(hvg), "--bootstrap", "20", "--against", "pca"]
        + ["--json", str(scores), "--replicates"]
    )
    assert "differences from pca" in capsys.readouterr().out
    written = json.loads(scores.read_text())
    assert len(written["classification"][0]["accuracy_boot"]) == 20
    differences = written["differences"]
    assert {row["embeddings"] for row in differences} == {"hvg"}
    assert {row["against"] for row in differences} == {"pca"}
    assert {"classification", "survival"} <= {row["table"] for row in differences}
    for row in differences:
        assert row["delta_lo"] <= row["delta_hi"]
    with pytest.raises(SystemExit, match="none of the embeddings"):
        main(["probe", str(pca), "--against", "hvg"])


def test_one_fold_baseline_then_probe(fake_dataset, tmp_path) -> None:
    """One fold on its own: a train / val / test report on that fold's test set."""
    embeddings = tmp_path / "pca"
    scores = tmp_path / "scores.json"
    main(["baseline-pca", "--out", str(embeddings), "--folds", "1", "--n-components", "4"])
    main(["probe", str(embeddings / "fold1.parquet"), "--json", str(scores), "--bootstrap", "0"])
    written = json.loads(scores.read_text())
    del written["meta"]
    for table in written.values():
        assert {row["split"] for row in table} == {"fold1"}
        # A fold file is labelled with its model's directory.
        assert {row["embeddings"] for row in table} == {"pca"}
