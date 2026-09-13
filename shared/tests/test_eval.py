import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from reimp_shared import hub, labels
from reimp_shared.data import load_expression
from reimp_shared.eval import (
    classification_probe,
    classification_scores,
    fit_linear_probe,
    pca_embeddings,
    read_embeddings,
    survival_probe,
    task_labels,
    write_embeddings,
)
from reimp_shared.splits import N_FOLDS, split_samples


def test_embeddings_round_trip(tmp_path) -> None:
    sample_index = np.array([4, 0, 7])
    embeddings = np.random.default_rng(0).normal(size=(3, 5)).astype(np.float32)
    path = write_embeddings(tmp_path / "sub" / "emb.parquet", sample_index, embeddings, 1)
    got_index, got, fold = read_embeddings(path)
    np.testing.assert_array_equal(got_index, sample_index)
    np.testing.assert_array_equal(got, embeddings)
    assert fold.tolist() == [1, 1, 1]


def test_embeddings_without_a_fold_are_rejected(tmp_path) -> None:
    path = tmp_path / "e.parquet"
    pq.write_table(pa.table({"sample_index": [0, 1]}), path)
    with pytest.raises(ValueError, match="no `fold` column"):
        read_embeddings(path)


def test_fold_embeddings_read_back_from_a_directory(tmp_path) -> None:
    rng = np.random.default_rng(0)
    for k in range(N_FOLDS):
        write_embeddings(tmp_path / f"fold{k}.parquet", np.arange(4), rng.normal(size=(4, 3)), k)
    (tmp_path / "scores.json").write_text("{}")  # anything else in the directory is ignored
    sample_index, embeddings, fold = read_embeddings(tmp_path)
    assert embeddings.shape == (4 * N_FOLDS, 3)
    assert sorted(fold.tolist()) == sorted(list(range(N_FOLDS)) * 4)
    assert sorted(sample_index.tolist()) == sorted(list(range(4)) * N_FOLDS)


def test_write_embeddings_checks_the_fold(tmp_path) -> None:
    with pytest.raises(ValueError, match="fold must be"):
        write_embeddings(tmp_path / "e.parquet", np.arange(2), np.zeros((2, 3)), fold=N_FOLDS)


def test_write_embeddings_checks_shape(tmp_path) -> None:
    with pytest.raises(ValueError, match="embeddings must be"):
        write_embeddings(tmp_path / "e.parquet", np.arange(3), np.zeros((2, 4)), 0)


def test_task_labels() -> None:
    samples = pd.DataFrame(
        {
            "project_id": ["A", "A", "B"],
            "sample_type": ["Primary Tumor", "Solid Tissue Normal", "Metastatic"],
        }
    )
    assert task_labels(samples, "project_id").tolist()[::2] == ["A", "B"]
    assert pd.isna(task_labels(samples, "project_id")[1])
    assert task_labels(samples, "tumor_vs_normal").tolist() == ["tumor", "normal", "tumor"]
    with pytest.raises(ValueError, match="unknown task"):
        task_labels(samples, "stage")


def test_organ_tasks_keep_only_their_own_tumours() -> None:
    samples = pd.DataFrame(
        {
            "project_id": ["TCGA-LUAD", "TCGA-LUSC", "TCGA-LUAD", "TCGA-BRCA"],
            "sample_type": ["Primary Tumor", "Primary Tumor", "Solid Tissue Normal", "Metastatic"],
        }
    )
    lung = task_labels(samples, "lung")
    assert lung[:2].tolist() == ["TCGA-LUAD", "TCGA-LUSC"]
    assert lung[2:].isna().all()


def test_linear_probe_separates_separable_classes() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, 1, (50, 4)), rng.normal(3, 1, (50, 4))])
    y = np.array(["a"] * 50 + ["b"] * 50)
    clf = fit_linear_probe(x[::2], y[::2])
    assert classification_scores(y[1::2], clf.predict(x[1::2]))["accuracy"] == 1.0


def _predictions(n: int = 200, seed: int = 0):
    """Noisy predictions, including a class that is never the truth."""
    rng = np.random.default_rng(seed)
    y_true = rng.choice(list("abcd"), n)
    y_pred = np.where(rng.random(n) < 0.6, y_true, rng.choice(list("abcde"), n))
    return y_true, y_pred


def _sklearn(y_true, y_pred) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        }


def test_classification_scores_match_scikit_learn() -> None:
    y_true, y_pred = _predictions()
    assert classification_scores(y_true, y_pred) == pytest.approx(_sklearn(y_true, y_pred))


def test_classification_weights_are_duplicated_rows() -> None:
    y_true, y_pred = _predictions()
    weights = np.random.default_rng(1).integers(0, 3, len(y_true)).astype(float)
    repeated = np.repeat(np.arange(len(y_true)), weights.astype(int))
    weighted = classification_scores(y_true, y_pred, weights[None, :])
    expected = _sklearn(y_true[repeated], y_pred[repeated])
    assert {name: v[0] for name, v in weighted.items()} == pytest.approx(expected)


def test_pca_embeddings_shape() -> None:
    values = np.random.default_rng(0).normal(size=(20, 10)).astype(np.float32)
    out = pca_embeddings(values, np.arange(15), n_components=3)
    assert out.shape == (20, 3)
    assert out.dtype == np.float32


def test_one_fold_is_scored_on_its_own_test_set(fake_dataset) -> None:
    data = load_expression(transform="lognorm", fold=2)
    embeddings = pca_embeddings(data.values, data.rows("train"), n_components=8)
    scores = classification_probe(
        data.samples["sample_index"].to_numpy(),
        embeddings,
        hub.load_samples(),
        fold=np.full(len(embeddings), 2),
        n_bootstrap=100,
    )
    assert set(scores["split"]) == {"fold2"}
    assert scores.set_index("task").loc["tumor_vs_normal", "n"] == len(data.rows("test"))
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        assert (scores[f"{metric}_lo"] <= scores[metric]).all()
        assert (scores[metric] <= scores[f"{metric}_hi"]).all()


def test_cv_classification_scores_every_labelled_sample_once(cv_pca) -> None:
    sample_index, embeddings, fold = cv_pca
    samples = hub.load_samples()
    scores = classification_probe(sample_index, embeddings, samples, fold=fold, n_bootstrap=100)
    assert set(scores["task"]) == {"project_id", "tumor_vs_normal"}
    assert scores["split"].tolist() == ["cv", "cv"]
    by_task = scores.set_index("task")
    for task in by_task.index:
        assert by_task.loc[task, "n"] == task_labels(samples, task).notna().sum()
    assert by_task.loc["project_id", "balanced_accuracy"] > 0.9
    assert (scores["accuracy_lo"] <= scores["accuracy"]).all()
    assert (scores["accuracy"] <= scores["accuracy_hi"]).all()


def test_cv_classification_pools_each_folds_test_predictions(cv_pca) -> None:
    """The pooled score is the score of the five folds' own test predictions, concatenated."""
    sample_index, embeddings, fold = cv_pca
    samples = hub.load_samples()
    scores = classification_probe(
        sample_index, embeddings, samples, ["project_id"], fold=fold, n_bootstrap=0
    )
    rows = samples.set_index("sample_index").loc[sample_index].reset_index()
    y = task_labels(rows, "project_id").to_numpy()
    truth, pred = [], []
    for k in range(N_FOLDS):
        x_k, y_k = embeddings[fold == k], y[fold == k]
        roles = split_samples(samples, k).set_axis(samples["sample_index"])
        split = roles.loc[sample_index[fold == k]].to_numpy()
        labelled = pd.notna(y_k)
        fit, test = labelled & (split != "test"), labelled & (split == "test")
        truth.append(y_k[test])
        pred.append(fit_linear_probe(x_k[fit], y_k[fit]).predict(x_k[test]))
    expected = classification_scores(np.concatenate(truth), np.concatenate(pred))
    assert scores.iloc[0][list(expected)].to_dict() == pytest.approx(expected)


def test_survival_probe_runs_on_the_fake_dataset(cv_pca) -> None:
    sample_index, embeddings, fold = cv_pca
    scores = survival_probe(
        sample_index,
        embeddings,
        hub.load_samples(),
        labels.load_survival(),
        fold=fold,
        n_bootstrap=100,
    )
    assert scores["split"].tolist() == ["cv"]
    assert (scores["patients"] > 0).all()
    assert scores["c_index"].dropna().between(0, 1).all()
    assert {"c_index_macro", "c_index_lo", "c_index_hi", "projects"} <= set(scores.columns)
