import lightning as L
import numpy as np
import pytest
import torch

from reimp_shared.data import ExpressionDataModule
from reimp_shared.ranking import GeneRanker
from reimp_tifbert.lit import LitTifBERT
from reimp_tifbert.sequences import all_windows

TINY = dict(max_genes=None, window=8, stride=4, d_model=32, n_layers=1, n_heads=4, dropout=0.0)


def _values(n: int = 8, g: int = 40, seed: int = 0) -> torch.Tensor:
    """TPM-like values from two expression profiles, a few genes unexpressed per sample."""
    rng = np.random.default_rng(seed)
    profile = np.exp(rng.normal(0.0, 1.5, (2, g)))[np.arange(n) % 2]
    values = rng.gamma(20.0, profile / 20.0)
    values[rng.random((n, g)) < 0.1] = 0.0
    return torch.from_numpy(values.astype(np.float32))


def _batch(values: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"values": values, "sample_index": torch.arange(len(values))}


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


def _datamodule(**kwargs) -> ExpressionDataModule:
    return ExpressionDataModule(quantification="tpm_unstranded", batch_size=8, **kwargs)


def test_ranking_needs_a_fitted_ranker() -> None:
    model = LitTifBERT(n_genes=40, **TINY)
    with pytest.raises(RuntimeError, match="fit the gene ranker"):
        model.on_before_batch_transfer(_batch(_values()), 0)


def test_the_ranker_must_see_the_models_genes() -> None:
    with pytest.raises(ValueError, match="41 genes"):
        LitTifBERT(n_genes=40, **TINY).fit_ranker(np.ones((3, 41)))


def test_only_gene_tokens_leave_the_cpu() -> None:
    values = _values()
    model = LitTifBERT(n_genes=40, **TINY).fit_ranker(values.numpy())
    batch = model.on_before_batch_transfer(_batch(values), 0)
    assert set(batch) == {"genes", "lengths", "sample_index"}
    assert batch["genes"].dtype == torch.long
    assert batch["genes"].max().item() <= model.model.pad_id


def test_training_step_is_finite_and_differentiable() -> None:
    values = _values()
    model = LitTifBERT(n_genes=40, **TINY).fit_ranker(values.numpy())
    loss = model.training_step(model.on_before_batch_transfer(_batch(values), 0), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.model.token_embedding.weight.grad is not None


def test_learns_to_fill_in_masked_genes() -> None:
    """On two fixed sentences the masked-gene loss must at least halve."""
    torch.manual_seed(0)
    values = _values()
    model = LitTifBERT(n_genes=40, **{**TINY, "n_layers": 2}).fit_ranker(values.numpy())
    batch = model.on_before_batch_transfer(_batch(values.repeat(8, 1)), 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(300):
        loss = model._masked_step(batch)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert np.mean(losses[-30:]) < 0.5 * np.mean(losses[:30])


def test_fit_logs_metrics(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitTifBERT(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(model, datamodule=dm)
    for name in ["train/loss", "val/loss", "val/accuracy"]:
        assert torch.isfinite(trainer.callback_metrics[name])


def test_the_ranker_is_fit_on_training_samples_only(fake_dataset, tmp_path) -> None:
    def fitted_weights(scale_rows: str | None) -> tuple[np.ndarray, ExpressionDataModule]:
        dm = _datamodule()
        dm.setup()
        if scale_rows is not None:
            # In place: the loaders read the same array.
            rows = dm.data.rows(scale_rows)
            dm.data.values[rows] *= np.random.default_rng(0).uniform(0.1, 10, dm.data.values.shape)[
                rows
            ]
        model = LitTifBERT(n_genes=dm.n_genes, **TINY)
        _trainer(tmp_path, max_steps=1, limit_val_batches=0).fit(model, datamodule=dm)
        return model.ranker.weight_, dm

    weights, dm = fitted_weights(None)
    train = dm.data.values[dm.data.rows("train")]
    np.testing.assert_array_equal(weights, GeneRanker().fit(train).weight_)
    assert not np.allclose(weights, GeneRanker().fit(dm.data.values).weight_)
    # Val and test samples can be anything; training samples cannot.
    for split in ["val", "test"]:
        np.testing.assert_array_equal(fitted_weights(split)[0], weights)
    assert not np.allclose(fitted_weights("train")[0], weights)


def test_the_checkpoint_carries_the_ranker(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitTifBERT(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_steps=1, limit_val_batches=0)
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(tmp_path / "model.ckpt")
    loaded = LitTifBERT.load_from_checkpoint(tmp_path / "model.ckpt", map_location="cpu")
    np.testing.assert_array_equal(loaded.ranker.weight_, model.ranker.weight_)
    np.testing.assert_array_equal(loaded.ranker.offset_, model.ranker.offset_)
    assert loaded.ranker.score == "tfidf" and loaded.ranker.idf_scheme == "entropy"


def test_a_samples_embedding_is_the_mean_over_its_windows(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitTifBERT(n_genes=dm.n_genes, **{**TINY, "embed_chunk": 5})
    trainer = _trainer(tmp_path)
    first = trainer.predict(model, datamodule=dm)
    second = trainer.predict(model, datamodule=dm)
    embeddings = torch.cat([o["embedding"] for o in first])
    assert torch.equal(embeddings, torch.cat([o["embedding"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), TINY["d_model"])

    # By hand, for the first sample: every window, pooled, then averaged.
    batch = model.on_before_batch_transfer(_batch(torch.from_numpy(dm.data.values[:1])), 0)
    tokens, _ = all_windows(batch["genes"], batch["lengths"], 8, 4, model.model.pad_id)
    assert len(tokens) > 1
    with torch.no_grad():
        by_hand = torch.stack([model.model.embed(w.unsqueeze(0))[0] for w in tokens]).mean(0)
    torch.testing.assert_close(embeddings[0], by_hand, atol=1e-5, rtol=1e-5)
