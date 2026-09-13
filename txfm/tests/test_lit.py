import lightning as L
import numpy as np
import pytest
import torch

from reimp_shared.data import ExpressionDataModule
from reimp_shared.preprocess import log_normalize
from reimp_txfm.lit import LitTxFM
from reimp_txfm.model import poisson_loss

TINY = dict(
    n_unmasked=8,
    d_model=32,
    n_layers=1,
    n_heads=4,
    decoder_layers=2,
    drop_path_rate=0.0,
    layer_scale_init=None,
)


def _batch(n: int = 8, g: int = 40, seed: int = 0) -> dict[str, torch.Tensor]:
    """Log-normalized counts from two expression profiles."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.5, (2, g))[np.arange(n) % 2]
    proportions = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    counts = torch.from_numpy(rng.poisson(1e5 * proportions))
    return {"values": log_normalize(counts), "sample_index": torch.arange(n)}


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
    model = LitTxFM(n_genes=40, **TINY)
    loss = model.training_step(_batch(), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.model.encoder.gene_embedding.weight.grad is not None


def test_raw_counts_are_rejected() -> None:
    model = LitTxFM(n_genes=40, **TINY)
    batch = {"values": torch.ones(2, 40, dtype=torch.int32), "sample_index": torch.arange(2)}
    with pytest.raises(TypeError, match="lognorm"):
        model.training_step(batch, 0)


def test_overfits_a_small_batch() -> None:
    """The loss's excess over the oracle (x̂ = x) must at least halve."""
    torch.manual_seed(0)
    model = LitTxFM(n_genes=40, **TINY)
    target = _batch()["values"]
    oracle = poisson_loss(target, target).mean().item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(200):
        loss = poisson_loss(model.model(*model._mask(target)), target).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] - oracle < 0.5 * (losses[0] - oracle)


def test_fit_logs_metrics_and_applies_the_papers_weight_decay(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(transform="lognorm", batch_size=8)
    model = LitTxFM(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(model, datamodule=dm)
    for name in ["train/loss", "val/loss", "val/loss_visible", "val/loss_holdout"]:
        assert torch.isfinite(trainer.callback_metrics[name])
    for name in ["val/pearson_holdout", "val/r2_holdout"]:
        assert torch.isfinite(trainer.callback_metrics[name])
    decay, no_decay = trainer.optimizers[0].param_groups
    assert decay["weight_decay"] == pytest.approx(
        1 / (model.hparams.lr * trainer.estimated_stepping_batches)
    )
    assert no_decay["weight_decay"] == 0.0


def test_predictions_are_reproducible_and_cover_every_sample(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(transform="lognorm", batch_size=8)
    model = LitTxFM(n_genes=dm.n_genes, embed_draws=3, **TINY)
    trainer = _trainer(tmp_path)
    first = trainer.predict(model, datamodule=dm)
    second = trainer.predict(model, datamodule=dm)
    embeddings = torch.cat([o["embedding"] for o in first])
    assert torch.equal(embeddings, torch.cat([o["embedding"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), TINY["d_model"])
