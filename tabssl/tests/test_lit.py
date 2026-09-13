import lightning as L
import numpy as np
import pytest
import torch

from reimp_shared.data import ExpressionDataModule, load_expression
from reimp_shared.preprocess import log_normalize
from reimp_shared.testing import scramble_held_out
from reimp_tabssl.lit import LitTabSSL

TINY = dict(hidden_dim=16, byol_hidden_dim=32, dropout=0.0)
G = 40


def _values(n: int = 32, g: int = G, seed: int = 0) -> torch.Tensor:
    """Log-normalized counts from two expression profiles."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.5, (2, g))[np.arange(n) % 2]
    proportions = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return log_normalize(torch.from_numpy(rng.poisson(1e5 * proportions))).float()


def _model(objective: str, **kwargs) -> LitTabSSL:
    torch.manual_seed(0)
    model = LitTabSSL(n_genes=G, objective=objective, **{**TINY, **kwargs})
    model.fit_statistics(_values(seed=1).numpy())
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


# ---------- objectives ----------


@pytest.mark.parametrize("objective", ["scarf", "vime", "byol"])
def test_training_step_is_finite_and_reaches_the_encoder(objective) -> None:
    model = _model(objective)
    loss = model.training_step({"values": _values(), "sample_index": torch.arange(32)}, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.encoder.blocks[0][0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("replacement", ["uniform", "marginal"])
def test_scarf_corrupts_a_fixed_count_per_sample_from_training_statistics(replacement) -> None:
    model = _model("scarf", scarf_replacement=replacement)
    x = model.scaler(_values())
    x_tilde = model._scarf_corrupt(x, torch.Generator().manual_seed(0))
    changed = x_tilde != x
    k = int(0.3 * G)
    if replacement == "uniform":
        # A continuous draw never lands on the clean value: exactly k change.
        assert (changed.sum(dim=1) == k).all()
        low, high = model.low.expand_as(x)[changed], model.high.expand_as(x)[changed]
        assert (x_tilde[changed] >= low).all() and (x_tilde[changed] <= high).all()
    else:
        # A draw from the pool can repeat the clean value, never add to k.
        assert (changed.sum(dim=1) <= k).all() and changed.sum(dim=1).float().mean() > k / 2
        in_pool = (x_tilde[:, None, :] == model.pool[None, :, :]).any(dim=1)
        assert in_pool[changed].all()


def test_vime_mask_target_is_where_the_corrupted_copy_differs() -> None:
    model = _model("vime")
    x = model.scaler(_values(n=256))
    x_tilde, changed = model._vime_corrupt(x, torch.Generator().manual_seed(0))
    assert torch.equal(changed, x_tilde != x)
    # Bernoulli(0.3), less the draws that repeat the clean value.
    assert 0.2 < changed.float().mean().item() <= 0.32
    # Every replaced value is the same gene's value in some training sample.
    in_pool = (x_tilde[:, None, :] == model.pool[None, :, :]).any(dim=1)
    assert in_pool[changed].all()


def test_vime_loss_is_bce_plus_alpha_mse() -> None:
    model = _model("vime", vime_alpha=2.0)
    losses = model.losses(_values(), torch.Generator().manual_seed(0))
    torch.testing.assert_close(losses["loss"], losses["mask_bce"] + 2.0 * losses["feature_mse"])


def test_byol_target_follows_the_online_network_by_ema() -> None:
    model = _model("byol", byol_ema=0.9)
    optimizer = model.configure_optimizers()
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    target_before = [p.clone() for p in model.target_encoder.parameters()]
    loss = model.training_step({"values": _values(), "sample_index": torch.arange(32)}, 0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.on_train_batch_end(None, None, 0)
    for t, b, o in zip(
        model.target_encoder.parameters(), target_before, model.encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(t, 0.9 * b + 0.1 * o)
    # The optimizer trains the online network only.
    trained = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not trained & {id(p) for p in model.target_encoder.parameters()}


@pytest.mark.parametrize(
    ("objective", "optimizer", "lr"),
    [
        ("scarf", torch.optim.Adam, 1e-4),
        ("vime", torch.optim.RMSprop, 1e-3),
        ("byol", torch.optim.Adam, 1e-4),
    ],
)
def test_each_objective_has_the_papers_optimizer(objective, optimizer, lr) -> None:
    opt = _model(objective).configure_optimizers()
    assert type(opt) is optimizer and opt.param_groups[0]["lr"] == lr
    assert _model(objective, lr=5e-4).configure_optimizers().param_groups[0]["lr"] == 5e-4


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="objective"):
        LitTabSSL(n_genes=G, objective="simclr")
    with pytest.raises(ValueError, match="scarf_replacement"):
        LitTabSSL(n_genes=G, scarf_replacement="normal")
    with pytest.raises(TypeError, match="lognorm"):
        LitTabSSL(n_genes=G).fit_statistics(np.ones((4, G), dtype=np.int32))


# ---------- statistics fit on training samples only ----------


@pytest.mark.parametrize("objective", ["scarf", "vime"])
def test_fitted_statistics_come_from_training_rows_only(
    objective, fake_dataset, monkeypatch, tmp_path
) -> None:
    fold = 1
    clean = load_expression(transform="lognorm", fold=fold)
    train = clean.rows("train")
    others = np.setdiff1d(np.arange(len(clean.values)), train)
    clean_train = clean.values[train]
    # Rewrite every validation and test sample: any statistic that saw one would move.
    scramble_held_out(monkeypatch, fold)

    dm = ExpressionDataModule(transform="lognorm", batch_size=8, fold=fold)
    model = LitTabSSL(n_genes=dm.n_genes, objective=objective, **TINY)
    _trainer(tmp_path, max_epochs=1, limit_train_batches=2, limit_val_batches=1).fit(
        model, datamodule=dm
    )
    assert not np.allclose(dm.data.values[others], clean.values[others])  # the scramble took

    torch.testing.assert_close(model.scaler.mean, torch.from_numpy(clean_train.mean(axis=0)))
    torch.testing.assert_close(model.scaler.std, torch.from_numpy(clean_train.std(axis=0)))
    scaled = model.scaler(torch.from_numpy(clean_train))
    torch.testing.assert_close(model.low, scaled.min(dim=0).values)
    torch.testing.assert_close(model.high, scaled.max(dim=0).values)
    if objective == "vime":
        # The replacement pool is the training samples, and nothing else.
        torch.testing.assert_close(model.pool, scaled)
    else:
        assert model.pool.numel() == 0  # uniform replacement needs only the ranges


def test_statistics_are_saved_with_the_model(tmp_path) -> None:
    model = _model("scarf")
    torch.save(model.state_dict(), tmp_path / "state.pt")
    loaded = LitTabSSL(n_genes=G, **TINY)
    loaded.load_state_dict(torch.load(tmp_path / "state.pt"))
    for name in ["scaler.mean", "scaler.std", "low", "high"]:
        assert torch.equal(loaded.state_dict()[name], model.state_dict()[name])
    assert "pool" not in model.state_dict()  # rebuilt from the data at every fit


def test_inputs_are_clipped_to_the_training_range() -> None:
    model = _model("scarf").eval()
    train = _values(seed=1)  # the samples `_model` fit on
    # Training samples lie within their own range: the clip leaves them as they are.
    torch.testing.assert_close(model.scale(train), model.scaler(train), rtol=0, atol=0)
    # Far outside the range, as a held-out sample can be for a gene nearly
    # constant over training: clipped to the training extreme.
    held_out = _values(n=4, seed=2)
    held_out[:, 0], held_out[:, 1] = 1e4, -1e4
    x = model.scale(held_out)
    assert (x >= model.low).all() and (x <= model.high).all()
    torch.testing.assert_close(x[:, 0], model.high[0].expand(4))
    torch.testing.assert_close(x[:, 1], model.low[1].expand(4))
    # So it embeds as the training maximum (minimum) would.
    at_extremes = held_out.clone()
    at_extremes[:, 0], at_extremes[:, 1] = train[:, 0].max(), train[:, 1].min()
    with torch.no_grad():
        embed = [
            model.predict_step({"values": v, "sample_index": torch.arange(4)}, 0)["embedding"]
            for v in (held_out, at_extremes)
        ]
    torch.testing.assert_close(embed[0], embed[1])


# ---------- validation, embedding and the untrained control ----------


def test_validation_loss_is_the_same_every_epoch(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(transform="lognorm", batch_size=8)
    model = LitTabSSL(n_genes=dm.n_genes, objective="vime", **TINY)
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=1)
    trainer.fit(model, datamodule=dm)
    first = trainer.validate(model, datamodule=dm, verbose=False)[0]["val/loss"]
    second = trainer.validate(model, datamodule=dm, verbose=False)[0]["val/loss"]
    assert first == second


def test_embeddings_are_the_encoder_on_clean_scaled_inputs(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(transform="lognorm", batch_size=8)
    model = LitTabSSL(n_genes=dm.n_genes, objective="scarf", **{**TINY, "dropout": 0.2})
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=2, limit_val_batches=1)
    trainer.fit(model, datamodule=dm)
    first = trainer.predict(model, datamodule=dm)
    second = trainer.predict(model, datamodule=dm)
    embeddings = torch.cat([o["embedding"] for o in first])
    assert torch.equal(embeddings, torch.cat([o["embedding"] for o in second]))
    sample_index = torch.cat([o["sample_index"] for o in first])
    assert sample_index.tolist() == dm.data.samples["sample_index"].tolist()
    with torch.no_grad():
        scaled = model.scaler(torch.from_numpy(dm.data.values))
        expected = model.eval().encoder(torch.clamp(scaled, model.low, model.high))
    torch.testing.assert_close(embeddings, expected)
    assert embeddings.shape == (len(sample_index), TINY["hidden_dim"])


def test_untrained_control_only_sets_batchnorm_statistics(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(transform="lognorm", batch_size=8)
    model = LitTabSSL(n_genes=dm.n_genes, objective="none", **TINY)
    before = {k: v.clone() for k, v in model.named_parameters()}
    trainer = _trainer(tmp_path, max_epochs=1)
    trainer.fit(model, datamodule=dm)
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name]), name
    bn = model.encoder.blocks[0][1]
    n_batches = len(dm.data.rows("train")) // 8
    assert bn.num_batches_tracked.item() == n_batches
    assert bn.momentum is None  # a cumulative average of the batches, not an exponential one
    assert not torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean))
    assert model.scaler.fitted
