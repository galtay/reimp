import numpy as np
import pytest

from reimp_plier.model import PLIER, b_step
from reimp_plier.pipeline import FoldModel, fit_fold
from reimp_plier.prior import read_gmt
from reimp_shared.data import ExpressionData, load_expression


def _model() -> PLIER:
    return PLIER(k=3, min_genes=3, max_iter=30)


def _load(fold: int = 0) -> ExpressionData:
    return load_expression(transform="lognorm", fold=fold)


def _with_values(data: ExpressionData, values: np.ndarray) -> ExpressionData:
    return ExpressionData(samples=data.samples, genes=data.genes, values=values)


def test_fitted_statistics_come_from_training_rows_only(fake_dataset, fake_prior) -> None:
    data = _load()
    gene_sets = read_gmt(fake_prior)
    fitted = fit_fold(data, _model(), gene_sets)
    train = data.values[data.rows("train")]
    np.testing.assert_allclose(fitted.scaler.mean, train.mean(axis=0, dtype=np.float64))
    np.testing.assert_allclose(fitted.scaler.sd, train.std(axis=0, ddof=1, dtype=np.float64))

    # Validation and test values, however wild, change nothing that is fit:
    # means, SDs, gene drops, the SVD, k, the lambdas, Z, U and B.
    wild = data.values.copy()
    held_out = np.flatnonzero(data.samples["split"].to_numpy() != "train")
    wild[held_out] = np.random.default_rng(1).uniform(0, 100, (len(held_out), wild.shape[1]))
    again = fit_fold(_with_values(data, wild), _model(), gene_sets)
    np.testing.assert_array_equal(again.gene_ids, fitted.gene_ids)
    np.testing.assert_array_equal(again.scaler.mean, fitted.scaler.mean)
    np.testing.assert_array_equal(again.scaler.sd, fitted.scaler.sd)
    for name in ("singular_values_", "z_", "u_", "b_"):
        np.testing.assert_array_equal(getattr(again.model, name), getattr(fitted.model, name))
    for name in ("k_", "l1_", "l2_", "l3_"):
        assert getattr(again.model, name) == getattr(fitted.model, name)


def test_genes_constant_over_training_rows_are_dropped(fake_dataset, fake_prior) -> None:
    data = _load()
    values = data.values.copy()
    values[data.rows("train"), 0] = 1.0  # varies only outside the training rows
    fitted = fit_fold(_with_values(data, values), _model(), read_gmt(fake_prior))
    assert data.genes["gene_id"].iloc[0] not in set(fitted.gene_ids)
    assert len(fitted.gene_ids) == len(data.genes) - 1
    assert np.isfinite(fitted.embed(_with_values(data, values))).all()


def test_embedding_is_one_projection_for_every_split(fake_dataset, fake_prior) -> None:
    data = _load()
    fitted = fit_fold(data, _model(), read_gmt(fake_prior))
    embeddings = fitted.embed(data)
    assert embeddings.shape == (len(data.samples), fitted.model.k_)
    # Training samples get the fit's own B ...
    train = data.rows("train")
    np.testing.assert_allclose(embeddings[train], fitted.model.b_.T, rtol=1e-5, atol=1e-5)
    # ... and every sample the same map, z-scored with the training statistics.
    columns = np.flatnonzero(np.isin(data.genes["gene_id"], fitted.gene_ids))
    y = ((data.values[:, columns] - fitted.scaler.mean) / fitted.scaler.sd).T
    z, l2 = fitted.model.z_, fitted.model.l2_
    expected = np.linalg.inv(z.T @ z + l2 * np.eye(z.shape[1])) @ z.T @ y
    np.testing.assert_allclose(embeddings, expected.T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b_step(y, z, l2).T, expected.T)


def test_prior_symbols_that_name_no_gene_are_counted(fake_dataset, fake_prior) -> None:
    fitted = fit_fold(_load(), _model(), read_gmt(fake_prior))
    assert fitted.unmapped == ["NOT_A_GENE"]
    assert fitted.model.prior_.shape == (len(fitted.gene_ids), 4)


def test_all_genes_false_models_only_the_priors_genes(fake_dataset, fake_prior) -> None:
    data = _load()
    everything = fit_fold(data, _model(), read_gmt(fake_prior))
    members = fit_fold(data, _model(), read_gmt(fake_prior), all_genes=False)
    assert len(members.gene_ids) < len(everything.gene_ids)
    assert (members.model.prior_.sum(axis=1) > 0).all()
    with pytest.raises(ValueError, match="needs a prior"):
        fit_fold(data, _model(), None, all_genes=False)


def test_save_and_load_round_trip(fake_dataset, fake_prior, tmp_path) -> None:
    data = _load()
    fitted = fit_fold(data, _model(), read_gmt(fake_prior))
    loaded = FoldModel.load(fitted.save(tmp_path / "fold0"))
    np.testing.assert_array_equal(loaded.embed(data), fitted.embed(data))
    assert loaded.model.params() == fitted.model.params()
    assert loaded.unmapped == fitted.unmapped
    assert loaded.model.names_ == fitted.model.names_
    assert len(loaded.model.annotations_) == len(fitted.model.annotations_)
