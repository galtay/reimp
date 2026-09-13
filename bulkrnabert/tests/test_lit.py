import lightning as L
import numpy as np
import pytest
import torch

from reimp_bulkrnabert.lit import LitBulkRNABert
from reimp_bulkrnabert.model import mlm_loss
from reimp_shared.data import ExpressionDataModule
from reimp_shared.testing import scramble_held_out
from reimp_shared.tokens import IGNORE_INDEX

TINY = dict(d_model=32, n_layers=1, n_heads=4, dim_ff=64)
DATA = dict(quantification="tpm_unstranded", transform="log1p", batch_size=8)


def _batch(n: int = 8, g: int = 40, seed: int = 0) -> dict[str, torch.Tensor]:
    """Log TPM from two expression profiles, with dropouts at zero."""
    rng = np.random.default_rng(seed)
    log_tpm = rng.gamma(2.0, 1.5, (2, g))[np.arange(n) % 2]
    log_tpm[:, : g // 5] = 0.0
    values = torch.from_numpy(log_tpm + rng.normal(0, 0.1, (n, g)).clip(0)).float()
    return {"values": values, "sample_index": torch.arange(n)}


def _fitted(g: int = 40, **kwargs) -> LitBulkRNABert:
    model = LitBulkRNABert(n_genes=g, **{**TINY, **kwargs})
    model.fit_tokenizer(_batch(g=g)["values"].numpy())
    return model


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


def test_training_step_is_finite_and_differentiable() -> None:
    model = _fitted()
    loss = model.training_step(_batch(), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.model.gene_embedding.weight.grad is not None


def test_tokenizing_needs_a_fitted_maximum() -> None:
    model = LitBulkRNABert(n_genes=40, **TINY)
    with pytest.raises(RuntimeError, match="not fit"):
        model.training_step(_batch(), 0)


def test_corruption_selects_about_mask_prob_of_the_true_tokens() -> None:
    model = _fitted(mask_prob=0.15)
    values = _batch(n=64)["values"]
    tokens = model.tokenizer.tokens(values)
    inputs, labels = model._corrupt(values, torch.Generator().manual_seed(0))
    selected = labels != IGNORE_INDEX
    assert selected.float().mean().item() == pytest.approx(0.15, abs=0.02)
    # Labels are the true bins, zeros included; untouched positions pass through.
    assert torch.equal(labels[selected], tokens[selected])
    assert (labels[selected] == 0).any()
    assert torch.equal(inputs[~selected], tokens[~selected])
    assert (inputs[selected] == model.model.mask_id).float().mean().item() > 0.7


def test_overfits_a_small_batch() -> None:
    """The masked-token loss on a fixed batch must at least halve."""
    torch.manual_seed(0)
    model = _fitted()
    values = _batch()["values"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        inputs, labels = model._corrupt(values)
        loss = mlm_loss(model.model(inputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])


def test_tokenizer_maximum_comes_from_training_rows_only(
    fake_dataset, monkeypatch, tmp_path
) -> None:
    fold = 1
    # Every val and test row is rewritten (times 3, plus 1), so its maximum
    # passes the training rows': a fit that reads any held-out split moves.
    scramble_held_out(monkeypatch, fold)
    dm = ExpressionDataModule(**DATA, fold=fold)
    dm.setup()
    data = dm.data
    train_max = float(data.values[data.rows("train")].max())
    for split in ["val", "test"]:
        assert data.values[data.rows(split)].max() > train_max + 0.5
    model = LitBulkRNABert(n_genes=dm.n_genes, **TINY)
    _trainer(tmp_path, max_epochs=1).fit(model, datamodule=dm)
    assert model.hparams.token_max == pytest.approx(train_max)
    assert model.tokenizer.max_ == model.hparams.token_max


def test_given_token_max_is_kept(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(**DATA)
    model = LitBulkRNABert(n_genes=dm.n_genes, token_max=12.5, **TINY)
    _trainer(tmp_path, max_epochs=1).fit(model, datamodule=dm)
    assert model.tokenizer.max_ == model.hparams.token_max == 12.5


def test_fit_logs_metrics_and_the_checkpoint_keeps_the_tokenizer(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(**DATA)
    model = LitBulkRNABert(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(model, datamodule=dm)
    for name in ["train/loss", "val/loss", "val/accuracy"]:
        assert torch.isfinite(trainer.callback_metrics[name])
    decay, no_decay = trainer.optimizers[0].param_groups
    assert decay["weight_decay"] == model.hparams.weight_decay
    assert no_decay["weight_decay"] == 0.0

    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)
    loaded = LitBulkRNABert.load_from_checkpoint(ckpt, map_location="cpu")
    assert loaded.tokenizer.max_ == model.tokenizer.max_
    values = torch.from_numpy(dm.data.values[:4])
    assert torch.equal(loaded.tokenizer.tokens(values), model.tokenizer.tokens(values))


def test_validation_masks_are_reproducible(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(**DATA)
    model = LitBulkRNABert(n_genes=dm.n_genes, token_max=10.0, **TINY)
    trainer = _trainer(tmp_path)
    first = trainer.validate(model, datamodule=dm, verbose=False)
    second = trainer.validate(model, datamodule=dm, verbose=False)
    assert first == second


def test_predictions_are_deterministic_and_cover_every_sample(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(**DATA)
    model = LitBulkRNABert(n_genes=dm.n_genes, token_max=10.0, **TINY)
    trainer = _trainer(tmp_path)
    first = trainer.predict(model, datamodule=dm)
    second = trainer.predict(model, datamodule=dm)
    embeddings = torch.cat([o["embedding"] for o in first])
    assert torch.equal(embeddings, torch.cat([o["embedding"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), TINY["d_model"])
