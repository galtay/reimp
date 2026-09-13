import lightning as L
import numpy as np
import pytest

from reimp_shared.data import load_expression
from reimp_shared.testing import scramble_held_out
from reimp_vae.data import VAEDataModule
from reimp_vae.lit import LitVAE
from reimp_vae.scaling import GeneScaler

TYBALT = dict(quantification="unstranded", transform="lognorm", top_genes=10, scaling="minmax")
MMDAE = dict(quantification="tpm_unstranded", transform="log1p", top_genes=None, scaling="zscore")
STATISTICS = ["genes_", "offset_", "scale_", "low_", "high_"]


def test_width_and_classes_are_known_before_setup(fake_dataset, project_organ) -> None:
    dm = VAEDataModule(**TYBALT, supervision="organ", project_organ=project_organ)
    assert dm.data is None
    assert dm.n_genes == 10
    assert dm.classes == ["kidney", "lung"] and dm.n_classes == 2
    # top_genes is capped at the genes selected (16 protein-coding in the fake data).
    assert VAEDataModule(top_genes=10_000).n_genes == VAEDataModule().n_genes == 16
    assert VAEDataModule().n_classes == 0


def test_setup_yields_the_top_genes_min_max_scaled(fake_dataset) -> None:
    dm = VAEDataModule(**TYBALT, batch_size=8)
    dm.setup()
    assert dm.data.values.shape == (len(dm.data.samples), 10)
    assert len(dm.data.genes) == 10
    train = dm.data.values[dm.data.rows("train")]
    np.testing.assert_allclose(train.min(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(train.max(axis=0), 1.0, atol=1e-6)
    assert dm.data.values.min() >= 0.0 and dm.data.values.max() <= 1.0
    batch = next(iter(dm.train_dataloader()))
    assert batch["values"].shape == (8, 10)
    assert "label" not in batch


def test_zscore_standardizes_every_gene_on_the_training_rows(fake_dataset) -> None:
    dm = VAEDataModule(quantification="tpm_unstranded", transform="log1p", scaling="zscore")
    dm.setup()
    assert dm.data.values.shape[1] == dm.n_genes == 16
    train = dm.data.values[dm.data.rows("train")].astype(np.float64)
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(train.std(axis=0), 1.0, atol=1e-4)


@pytest.mark.parametrize("settings", [TYBALT, MMDAE], ids=["minmax", "zscore"])
def test_scaler_is_fit_on_training_rows_only(settings, fake_dataset, monkeypatch) -> None:
    """Rewriting every val and test value leaves the gene ranking and scaling unchanged."""
    fold = 3

    def fitted() -> VAEDataModule:
        dm = VAEDataModule(**settings, fold=fold)
        dm.setup()
        return dm

    clean = fitted()
    # What it holds is the training rows' own statistics.
    kwargs = {k: settings[k] for k in ("quantification", "transform")}
    data = load_expression(**kwargs, fold=fold)
    direct = GeneScaler(settings["scaling"], settings["top_genes"])
    direct.fit(data.values[data.rows("train")])
    for name in STATISTICS:
        np.testing.assert_array_equal(getattr(clean.scaler, name), getattr(direct, name))
    everything = GeneScaler(settings["scaling"], settings["top_genes"]).fit(data.values)
    assert not np.array_equal(clean.scaler.scale_, everything.scale_)

    scramble_held_out(monkeypatch, fold)
    scrambled = fitted()
    for name in STATISTICS:
        np.testing.assert_array_equal(getattr(clean.scaler, name), getattr(scrambled.scaler, name))
    train = clean.data.rows("train")
    np.testing.assert_array_equal(scrambled.data.values[train], clean.data.values[train])
    assert not np.array_equal(scrambled.data.values, clean.data.values), "scramble never reached"


def test_supervised_items_carry_labels_for_tumours_and_normals(fake_dataset, project_organ) -> None:
    dm = VAEDataModule(supervision="organ", project_organ=project_organ, batch_size=1000)
    dm.setup()
    (batch,) = list(dm.predict_dataloader())
    samples = dm.data.samples
    expected = samples["project_id"].map({"TCGA-AAA": 1, "TCGA-BBB": 1, "TCGA-CCC": 0})
    assert batch["label"].tolist() == expected.tolist()
    assert samples["sample_type"].eq("Solid Tissue Normal").any()
    assert "label" in dm.datasets["train"][0]


def test_a_checkpoint_restores_the_scaler_without_refitting(fake_dataset, tmp_path, monkeypatch):
    dm = VAEDataModule(**TYBALT, batch_size=8, fold=1)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(LitVAE(n_genes=dm.n_genes, latent_dim=4), datamodule=dm)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)

    monkeypatch.setattr(GeneScaler, "fit", lambda self, values: pytest.fail("refit on load"))
    restored = VAEDataModule.load_from_checkpoint(ckpt)
    assert restored.hparams.fold == 1 and restored.hparams.top_genes == 10
    restored.setup()
    np.testing.assert_array_equal(restored.data.values, dm.data.values)
