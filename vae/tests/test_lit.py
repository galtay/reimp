import lightning as L
import numpy as np
import pytest
import torch

from reimp_vae.data import VAEDataModule
from reimp_vae.lit import LitVAE

TYBALT = dict(latent_dim=4)
MMDAE = dict(
    latent_dim=4,
    hidden_factor=0.5,
    heads="linear",
    reconstruction="mse",
    regularizer="mmd",
    glorot_init=False,
    lr=1.72e-3,
)


def _batch(n: int = 16, g: int = 12, seed: int = 0, labels: bool = False) -> dict:
    """Values in [0, 1] from two expression profiles."""
    rng = np.random.default_rng(seed)
    profiles = rng.uniform(0.1, 0.9, (2, g))[np.arange(n) % 2]
    values = np.clip(profiles + rng.normal(0.0, 0.05, (n, g)), 0.0, 1.0)
    batch = {"values": torch.tensor(values, dtype=torch.float32), "sample_index": torch.arange(n)}
    if labels:
        batch["label"] = torch.arange(n) % 2
    return batch


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


@pytest.mark.parametrize(
    ("config", "n_classes"), [(TYBALT, 0), (MMDAE, 0), (MMDAE, 2)], ids=["tybalt", "mmd", "sup"]
)
def test_training_step_is_finite_and_differentiable(config, n_classes) -> None:
    model = LitVAE(n_genes=12, n_classes=n_classes, **config)
    loss = model.training_step(_batch(labels=n_classes > 0), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_bce_rejects_values_outside_the_unit_interval() -> None:
    batch = _batch()
    batch["values"] = batch["values"] * 3 - 1
    with pytest.raises(ValueError, match="minmax"):
        LitVAE(n_genes=12, **TYBALT).training_step(batch, 0)


def test_a_supervised_model_needs_labels() -> None:
    with pytest.raises(KeyError, match="label"):
        LitVAE(n_genes=12, n_classes=2, **MMDAE).training_step(_batch(), 0)


def test_the_kl_is_off_in_epoch_zero_and_always_on_in_the_objective() -> None:
    model = LitVAE(n_genes=12, **TYBALT)
    losses = model.losses(_batch())
    assert model.regularizer_weight() == 0.0  # epoch 0 of the warm-up
    torch.testing.assert_close(model.objective(losses, model.regularizer_weight()), losses["recon"])
    torch.testing.assert_close(model.objective(losses, 1.0), losses["recon"] + losses["kl"])
    assert LitVAE(n_genes=12, **MMDAE).regularizer_weight() == 1.0


@pytest.mark.parametrize("config", [TYBALT, MMDAE], ids=["tybalt", "mmd"])
def test_the_objective_falls_on_a_small_batch(config) -> None:
    torch.manual_seed(0)
    model = LitVAE(n_genes=12, **config)
    batch = _batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    history = []
    for _ in range(200):
        loss = model.objective(model.losses(batch), 1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    assert np.mean(history[-10:]) < 0.8 * np.mean(history[:10])


def test_tybalt_fit_logs_each_term_and_warms_up_the_kl(fake_dataset, tmp_path) -> None:
    dm = VAEDataModule(
        quantification="unstranded",
        transform="lognorm",
        top_genes=10,
        scaling="minmax",
        batch_size=8,
    )
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(LitVAE(n_genes=dm.n_genes, **TYBALT), datamodule=dm)
    metrics = trainer.callback_metrics
    assert metrics["train/beta"] == 1.0  # the second epoch
    for name in ["train/loss", "train/recon", "train/kl", "val/recon", "val/kl"]:
        assert torch.isfinite(metrics[name])
    torch.testing.assert_close(metrics["val/loss"], metrics["val/recon"] + metrics["val/kl"])


def test_supervised_fit_trains_the_head(fake_dataset, project_organ, tmp_path) -> None:
    dm = VAEDataModule(
        quantification="tpm_unstranded",
        transform="log1p",
        scaling="zscore",
        supervision="organ",
        project_organ=project_organ,
        batch_size=8,
    )
    model = LitVAE(n_genes=dm.n_genes, n_classes=dm.n_classes, **MMDAE)
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(model, datamodule=dm)
    metrics = trainer.callback_metrics
    for name in ["val/recon", "val/mmd", "val/class", "val/accuracy"]:
        assert torch.isfinite(metrics[name])
    torch.testing.assert_close(
        metrics["val/loss"], metrics["val/recon"] + metrics["val/mmd"] + metrics["val/class"]
    )


def test_predictions_are_posterior_means_of_every_sample(fake_dataset, tmp_path) -> None:
    dm = VAEDataModule(quantification="tpm_unstranded", transform="log1p", scaling="zscore")
    model = LitVAE(n_genes=dm.n_genes, **MMDAE)
    trainer = _trainer(tmp_path)
    first = trainer.predict(model, datamodule=dm)
    second = trainer.predict(model, datamodule=dm)
    embeddings = torch.cat([o["embedding"] for o in first])
    assert torch.equal(embeddings, torch.cat([o["embedding"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    with torch.no_grad():
        mu, _ = model.model.eval().encoder(torch.from_numpy(dm.data.values))
    torch.testing.assert_close(embeddings, mu)
