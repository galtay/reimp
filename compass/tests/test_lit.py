import lightning as L
import numpy as np
import pytest
import torch

from reimp_compass.lit import LitCompass
from reimp_shared.data import ExpressionDataModule

TINY = dict(d_model=8, n_heads=2, head_dim=4, dim_ff=16, dropout=0.0)


def _datamodule(**kwargs) -> ExpressionDataModule:
    dm = ExpressionDataModule("tpm_unstranded", transform="log1p", batch_size=8, **kwargs)
    dm.setup()
    return dm


def _model(dm: ExpressionDataModule, hierarchy_path: str, **kwargs) -> LitCompass:
    return LitCompass(n_genes=dm.n_genes, hierarchy_path=hierarchy_path, **TINY, **kwargs)


def _trainer(tmp_path, **kwargs) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        **kwargs,
    )


def test_the_scaler_is_fit_on_training_samples_only(fake_dataset, hierarchy_path) -> None:
    dm = _datamodule()
    data = dm.data
    model = _model(dm, hierarchy_path)
    with pytest.warns(UserWarning, match="2 concept genes are not selected"):
        model.prepare(data)
    train = data.values[data.rows("train")]
    scaler = model.model.scaler
    np.testing.assert_allclose(scaler.minimum.numpy(), train.min(0), rtol=1e-6)
    np.testing.assert_allclose(scaler.scale.numpy(), train.max(0) - train.min(0), rtol=1e-5)
    # The full cohort's range differs: the fit did not see val or test rows ...
    assert not np.allclose(scaler.scale.numpy(), data.values.max(0) - data.values.min(0))
    # ... and moving them moves nothing.
    before = {k: v.clone() for k, v in model.model.scaler.state_dict().items()}
    held_out = np.concatenate([data.rows("val"), data.rows("test")])
    data.values[held_out] = 1e3
    model.prepare(data)
    for name, value in model.model.scaler.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_prepare_binds_concept_genes_by_symbol(fake_dataset, hierarchy_path) -> None:
    dm = _datamodule()
    model = _model(dm, hierarchy_path)
    with pytest.warns(UserWarning, match="not selected"):
        model.prepare(dm.data)
    projector = model.model.projector
    names = dm.data.genes["gene_name"].to_numpy()
    present = projector.member_present.numpy()
    assert names[projector.member_gene.numpy()[present]].tolist() == [
        f"GENE{i}" for i in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 30)
    ]


def test_negatives_are_other_patients_of_the_same_split(fake_dataset, hierarchy_path) -> None:
    dm = _datamodule()
    model = _model(dm, hierarchy_path, negatives="same_project")
    with pytest.warns(UserWarning):
        model.prepare(dm.data)
    samples = dm.data.samples
    for split in ("train", "val"):
        anchors = torch.from_numpy(np.repeat(dm.data.rows(split), 20))
        negatives = model._samplers[split].draw(anchors).numpy()
        assert np.isin(negatives, dm.data.rows(split)).all()
        case = samples["case_submitter_id"].to_numpy()
        assert (case[negatives] != case[anchors.numpy()]).all()
    project = samples["project_id"].to_numpy()
    train = np.repeat(dm.data.rows("train"), 20)
    assert (
        project[model._samplers["train"].draw(torch.from_numpy(train)).numpy()] == project[train]
    ).all()


@pytest.mark.filterwarnings("ignore:.*concept genes")
def test_training_step_is_finite_and_differentiable(fake_dataset, hierarchy_path) -> None:
    dm = _datamodule()
    model = _model(dm, hierarchy_path)
    model.prepare(dm.data)
    loss = model.training_step(next(iter(dm.train_dataloader())), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for p in (model.model.tokenizer.weight, model.model.projector.gene_logits):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert model.model.projector.set_logits.grad is not None


@pytest.mark.filterwarnings("ignore:.*concept genes")
def test_validation_triplets_are_reproducible(fake_dataset, hierarchy_path) -> None:
    dm = _datamodule()
    model = _model(dm, hierarchy_path).eval()
    model.prepare(dm.data)
    batch = next(iter(dm.val_dataloader()))
    runs = []
    for _ in range(2):
        model.on_validation_epoch_start()
        runs.append(torch.cat(model._triplets(batch, "val", model._generator)))
    torch.testing.assert_close(runs[0], runs[1])


def test_raw_counts_are_rejected(fake_dataset, hierarchy_path) -> None:
    dm = ExpressionDataModule(batch_size=8)
    dm.setup()
    model = _model(dm, hierarchy_path)
    batch = next(iter(dm.train_dataloader()))
    with pytest.raises(TypeError, match="log1p"):
        model.training_step(batch, 0)


@pytest.mark.filterwarnings("ignore:.*concept genes")
@pytest.mark.parametrize("negatives", ["any", "same_project"])
def test_fit_logs_metrics_and_predicts_every_sample(
    negatives, fake_dataset, hierarchy_path, tmp_path
) -> None:
    dm = _datamodule()
    model = _model(dm, hierarchy_path, negatives=negatives)
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(model, datamodule=dm)
    for name in ["train/loss", "val/loss", "val/d_pos", "val/d_neg", "val/active"]:
        assert torch.isfinite(trainer.callback_metrics[name])
    train = dm.data.values[dm.data.rows("train")]
    np.testing.assert_allclose(model.model.scaler.minimum.numpy(), train.min(0), rtol=1e-6)

    first = trainer.predict(model, datamodule=dm)
    second = trainer.predict(model, datamodule=dm)
    concepts = torch.cat([o["concepts"] for o in first])
    torch.testing.assert_close(concepts, torch.cat([o["concepts"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    assert concepts.shape == (len(sample_index), 3)
    assert torch.cat([o["sets"] for o in first]).shape == (len(sample_index), 6)
