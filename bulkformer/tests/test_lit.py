import dataclasses

import lightning as L
import numpy as np
import pytest
import torch

from reimp_bulkformer.graph import coexpression_graph
from reimp_bulkformer.lit import LitBulkFormer, pearson, warmup_cosine
from reimp_bulkformer.model import mask_genes, masked_mse
from reimp_shared.data import ExpressionDataModule

TINY = dict(d_model=16, n_blocks=1, n_layers=1, n_heads=2, dropout=0.0, graph_k=5)


def _datamodule(**kwargs) -> ExpressionDataModule:
    return ExpressionDataModule(
        **{"quantification": "tpm_unstranded", "transform": "log1p", "batch_size": 8, **kwargs}
    )


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


def _fitted(fake_dataset, **kwargs) -> tuple[LitBulkFormer, ExpressionDataModule]:
    dm = _datamodule()
    dm.setup()
    model = LitBulkFormer(n_genes=dm.n_genes, **{**TINY, **kwargs})
    model.fit_statistics(dm.data)
    return model, dm


def _graph(model: LitBulkFormer) -> tuple[torch.Tensor, torch.Tensor]:
    return model.model.graph_index.clone(), model.model.graph_weight.clone()


# ---------- fitted statistics ----------


def test_fitted_statistics_come_from_training_rows_only(fake_dataset) -> None:
    model, dm = _fitted(fake_dataset)
    train = dm.data.values[dm.data.rows("train")]
    index, weight = coexpression_graph(np.expm1(train), k=TINY["graph_k"])
    assert torch.equal(model.model.graph_index, index)
    torch.testing.assert_close(model.model.graph_weight, weight)
    np.testing.assert_allclose(model.gene_mean.numpy(), train.mean(axis=0), rtol=1e-5)

    rng = np.random.default_rng(0)

    def refit(rows: np.ndarray) -> LitBulkFormer:
        values = dm.data.values.copy()
        values[rows] = rng.uniform(0, 10, (len(rows), values.shape[1])).astype(np.float32)
        other = LitBulkFormer(n_genes=dm.n_genes, **TINY)
        other.fit_statistics(dataclasses.replace(dm.data, values=values))
        return other

    # Scrambling every validation and test sample changes nothing...
    held_out = np.concatenate([dm.data.rows("val"), dm.data.rows("test")])
    unchanged = refit(held_out)
    assert torch.equal(unchanged.model.graph_index, index)
    torch.testing.assert_close(unchanged.model.graph_weight, weight)
    torch.testing.assert_close(unchanged.gene_mean, model.gene_mean)
    # ...while scrambling the training samples does.
    changed = refit(dm.data.rows("train"))
    assert not torch.allclose(changed.gene_mean, model.gene_mean)
    assert not torch.equal(changed.model.graph_index, index)


def test_graph_space_log1p_correlates_the_values_as_given(fake_dataset) -> None:
    model, dm = _fitted(fake_dataset, graph_space="log1p")
    train = dm.data.values[dm.data.rows("train")]
    assert torch.equal(model.model.graph_index, coexpression_graph(train, k=TINY["graph_k"])[0])


def test_random_graph_is_degree_matched_but_rewired(fake_dataset) -> None:
    coexpression, _ = _fitted(fake_dataset)
    control, _ = _fitted(fake_dataset, graph="random")
    a, b = _graph(coexpression)[0], _graph(control)[0]
    n = coexpression.hparams.n_genes
    degrees = torch.bincount(a[0], minlength=n).sort().values
    assert torch.equal(torch.bincount(b[0], minlength=n).sort().values, degrees)
    assert not torch.equal(a, b)


def test_no_graph_fits_gene_means_only(fake_dataset) -> None:
    model, _ = _fitted(fake_dataset, graph="none")
    assert not model.model.has_graph and bool(model.fitted)


def test_setup_needs_log1p_values(fake_dataset, tmp_path) -> None:
    dm = _datamodule(transform="none")
    model = LitBulkFormer(n_genes=dm.n_genes, **TINY)
    with pytest.raises(ValueError, match="log1p"):
        _trainer(tmp_path, max_steps=1).fit(model, datamodule=dm)


def test_unknown_graph_raises() -> None:
    with pytest.raises(ValueError, match="unknown graph"):
        LitBulkFormer(n_genes=10, graph="ppi")


# ---------- steps ----------


def test_training_step_is_finite_and_differentiable(fake_dataset) -> None:
    model, dm = _fitted(fake_dataset)
    batch = next(iter(dm.train_dataloader()))
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.model.gene_embedding.weight.grad is not None
    assert model.model.blocks[0].gcn.linear.weight.grad is not None


def test_overfits_a_small_batch(fake_dataset) -> None:
    """On a batch it trains on, masked MSE falls well below the training gene means'.

    Both are scored on the same held-out masks, averaged over several: one
    mask covers only a couple of genes per sample.
    """
    torch.manual_seed(0)
    model, dm = _fitted(fake_dataset)
    values = next(iter(dm.train_dataloader()))["values"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        mask = mask_genes(*values.shape, 0.15)
        loss = masked_mse(model.model(values, mask), values, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    fitted, baseline = 0.0, 0.0
    with torch.no_grad():
        for seed in range(5):
            mask = mask_genes(*values.shape, 0.15, generator=torch.Generator().manual_seed(seed))
            fitted += masked_mse(model.model(values, mask), values, mask).item()
            baseline += masked_mse(model.gene_mean.expand_as(values), values, mask).item()
    assert fitted < 0.75 * baseline


def test_fit_logs_metrics_and_uses_the_training_graph(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitBulkFormer(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(model, datamodule=dm)
    for name in ["train/loss", "val/loss", "val/loss_gene_mean", "val/pearson"]:
        assert torch.isfinite(trainer.callback_metrics[name])
    train = dm.data.values[dm.data.rows("train")]
    assert torch.equal(
        model.model.graph_index.cpu(), coexpression_graph(np.expm1(train), k=TINY["graph_k"])[0]
    )
    decay, no_decay = trainer.optimizers[0].param_groups
    assert decay["weight_decay"] == model.hparams.weight_decay
    assert no_decay["weight_decay"] == 0.0


def test_features_are_redrawn_every_interval(fake_dataset, tmp_path) -> None:
    def projection_after_fit(interval: int | None) -> torch.Tensor:
        torch.manual_seed(0)
        dm = _datamodule()
        model = LitBulkFormer(n_genes=dm.n_genes, feature_redraw_interval=interval, **TINY)
        start = model.model.blocks[0].layers[0].attn.projection.clone()
        _trainer(tmp_path, max_steps=3, limit_val_batches=0).fit(model, datamodule=dm)
        return start, model.model.blocks[0].layers[0].attn.projection

    start, end = projection_after_fit(None)
    assert torch.equal(start, end)
    start, end = projection_after_fit(1)
    assert not torch.equal(start, end)


def test_checkpoint_restores_the_graph_and_the_embeddings(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitBulkFormer(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_epochs=1)
    trainer.fit(model, datamodule=dm)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)

    restored = LitBulkFormer.load_from_checkpoint(ckpt, map_location="cpu")
    assert bool(restored.fitted)
    assert torch.equal(restored.model.graph_index, model.model.graph_index.cpu())
    torch.testing.assert_close(restored.gene_mean, model.gene_mean.cpu())
    first = _trainer(tmp_path).predict(model, datamodule=dm)
    second = _trainer(tmp_path).predict(restored, datamodule=dm)
    torch.testing.assert_close(
        torch.cat([o["embedding"] for o in first]), torch.cat([o["embedding"] for o in second])
    )


def test_predictions_are_reproducible_and_cover_every_sample(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitBulkFormer(n_genes=dm.n_genes, **TINY)
    first = _trainer(tmp_path).predict(model, datamodule=dm)
    second = _trainer(tmp_path).predict(model, datamodule=dm)
    embeddings = torch.cat([o["embedding"] for o in first])
    assert torch.equal(embeddings, torch.cat([o["embedding"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), TINY["d_model"])


# ---------- helpers ----------


def test_warmup_cosine_rises_linearly_then_decays_to_zero() -> None:
    lrs = [warmup_cosine(step, warmup=10, total=100) for step in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:], strict=False))
    assert warmup_cosine(100, warmup=10, total=100) == pytest.approx(0.0)


def test_pearson_is_per_row() -> None:
    a = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    b = torch.tensor([[2.0, 4.0, 6.0], [3.0, 1.0, 2.0]])
    torch.testing.assert_close(pearson(a, b), torch.tensor([1.0, 0.0]))
