import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch

from reimp_mojo.lit import LitMOJO, genome_order, warmup_cosine
from reimp_shared.data import ExpressionDataModule
from reimp_shared.testing import scramble_held_out
from reimp_shared.tokens import IGNORE_INDEX

TINY = dict(
    n_bins=16,
    embed_dim=16,
    conv_channels=16,
    d_model=32,
    n_down=3,
    n_layers=1,
    n_heads=4,
    stem_kernel=5,
    kernel_size=3,
)


def _datamodule(**kwargs) -> ExpressionDataModule:
    return ExpressionDataModule("tpm_unstranded", transform="log1p", batch_size=8, **kwargs)


def _batch(n: int = 8, g: int = 40, seed: int = 0) -> dict[str, torch.Tensor]:
    """Log TPM from two expression profiles, a fifth of it zero."""
    rng = np.random.default_rng(seed)
    means = rng.gamma(1.0, 3.0, (2, g))[np.arange(n) % 2]
    values = np.log1p(rng.gamma(2.0, means / 2.0)) * (rng.random((n, g)) > 0.2)
    return {"values": torch.from_numpy(values.astype(np.float32)), "sample_index": torch.arange(n)}


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


# ---------- tokens and masking ----------


def test_training_step_is_finite_and_differentiable() -> None:
    model = LitMOJO(n_genes=40, token_max=4.0, **TINY)
    loss = model.training_step(_batch(), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.model.gene_embedding.weight.grad is not None


def test_an_unfit_tokenizer_and_raw_counts_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="not fit"):
        LitMOJO(n_genes=40, **TINY).training_step(_batch(), 0)
    counts = {"values": torch.ones(2, 40, dtype=torch.int32), "sample_index": torch.arange(2)}
    with pytest.raises(TypeError, match="log1p"):
        LitMOJO(n_genes=40, token_max=4.0, **TINY).training_step(counts, 0)


def test_tokens_are_bins_of_the_values_over_token_max() -> None:
    model = LitMOJO(n_genes=4, token_max=3.0, **{**TINY, "n_bins": 4})
    values = torch.tensor([[0.0, 0.5, 2.0, 9.0]])
    assert model._tokens(values).tolist() == [[0, 1, 2, 3]]


def test_masking_selects_15_percent_of_genes_and_masks_80_percent_of_those() -> None:
    model = LitMOJO(n_genes=500, token_max=4.0, **TINY)
    tokens = torch.randint(16, (200, 500), generator=torch.Generator().manual_seed(0))
    inputs, labels = model._mask(tokens, torch.Generator().manual_seed(1))
    assert inputs.shape == labels.shape == tokens.shape
    selected = labels != IGNORE_INDEX
    assert selected.float().mean().item() == pytest.approx(0.15, abs=0.01)
    masked = inputs == model.model.mask_id
    assert (masked & ~selected).sum() == 0
    assert (masked.sum() / selected.sum()).item() == pytest.approx(0.8, abs=0.02)
    # The pad token is the model's alone: masking never writes it.
    assert (inputs != model.model.pad_id).all()


def test_seeded_masks_are_the_same_on_every_call() -> None:
    model = LitMOJO(n_genes=40, token_max=4.0, **TINY)
    tokens = model._tokens(_batch()["values"])
    a = model._mask(tokens, torch.Generator().manual_seed(0))
    b = model._mask(tokens, torch.Generator().manual_seed(0))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_loss_falls_on_a_small_batch() -> None:
    torch.manual_seed(0)
    model = LitMOJO(n_genes=40, token_max=float(_batch()["values"].max()), **TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss = model.training_step(_batch(), 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.8 * np.mean(losses[:10])


# ---------- gene order ----------


def test_genome_order_sorts_by_chromosome_then_start() -> None:
    genes = pd.DataFrame(
        {
            "chromosome": ["chr2", "chr1", "chrX", "chr1", None, "chr10"],
            "start": [5.0, 10.0, 1.0, 2.0, np.nan, 3.0],
        }
    )
    assert genome_order(genes).tolist() == [3, 1, 0, 5, 2, 4]


def test_genome_order_permutes_tokens_before_the_model(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitMOJO(n_genes=dm.n_genes, gene_order="genome", **TINY)
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    trainer.fit(model, datamodule=dm)
    assert sorted(model.gene_perm.tolist()) == list(range(dm.n_genes))
    model.gene_perm = torch.arange(dm.n_genes).flip(0)
    values = torch.from_numpy(dm.data.values[:4])
    torch.testing.assert_close(model._tokens(values), model.tokenizer.tokens(values).flip(1))


def test_unknown_gene_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="gene_order"):
        LitMOJO(n_genes=40, gene_order="random")


# ---------- fitted statistics ----------


@pytest.mark.parametrize("fold", [0, 3])
def test_tokenizer_maximum_comes_from_training_rows_only(
    fold, fake_dataset, monkeypatch, tmp_path
) -> None:
    clean = _datamodule(fold=fold)
    clean.setup()
    expected = float(clean.data.values[clean.data.rows("train")].max())
    # Scrambled val and test rows would set the maximum if they counted.
    scramble_held_out(monkeypatch, fold)
    dm = _datamodule(fold=fold)
    dm.setup()
    assert dm.data.values.max() > expected
    model = LitMOJO(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    trainer.fit(model, datamodule=dm)
    assert model.hparams.token_max == expected
    assert model.tokenizer.max_ == expected


def test_fit_logs_metrics(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitMOJO(n_genes=dm.n_genes, **TINY)
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.fit(model, datamodule=dm)
    for name in ["train/loss", "train/accuracy", "val/loss", "val/accuracy"]:
        assert torch.isfinite(trainer.callback_metrics[name])
    decay, no_decay = trainer.optimizers[0].param_groups
    assert decay["weight_decay"] == model.hparams.weight_decay
    assert no_decay["weight_decay"] == 0.0


@pytest.mark.parametrize("total_steps", [1, 2, 20, 1000])
def test_warmup_cosine_peaks_after_warmup_and_decays_without_reaching_zero(total_steps) -> None:
    factor = warmup_cosine(total_steps, 0.05)
    values = [factor(step) for step in range(total_steps)]
    warmup = max(1, round(0.05 * total_steps))
    assert values[warmup - 1] == 1.0
    assert values[:warmup] == sorted(values[:warmup])
    assert values[warmup - 1 :] == sorted(values[warmup - 1 :], reverse=True)
    assert all(0 < v <= 1 for v in values)


def test_validation_masks_repeat_across_loops(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitMOJO(n_genes=dm.n_genes, token_max=10.0, **TINY)
    trainer = _trainer(tmp_path)
    first = trainer.validate(model, datamodule=dm, verbose=False)
    assert trainer.validate(model, datamodule=dm, verbose=False) == first


def test_predictions_are_reproducible_and_cover_every_sample(fake_dataset, tmp_path) -> None:
    dm = _datamodule()
    model = LitMOJO(n_genes=dm.n_genes, token_max=10.0, **TINY)
    trainer = _trainer(tmp_path)
    first = trainer.predict(model, datamodule=dm)
    second = trainer.predict(model, datamodule=dm)
    embeddings = torch.cat([o["embedding"] for o in first])
    assert torch.equal(embeddings, torch.cat([o["embedding"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), TINY["d_model"])
