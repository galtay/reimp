import json
from pathlib import Path

import lightning as L
import pytest

from reimp_shared import hub
from reimp_shared.eval import read_embeddings
from reimp_shared.testing import assert_embedding_ignores_held_out
from reimp_vae.cli import build_cli
from reimp_vae.data import VAEDataModule
from reimp_vae.embed import embed
from reimp_vae.lit import LitVAE

CONFIGS = Path(__file__).parents[1] / "configs"


def _args(config: str, tmp_path: Path, project_organ: dict[str, str]) -> list[str]:
    return [
        "--config",
        str(CONFIGS / config),
        f"--trainer.default_root_dir={tmp_path}",
        f"--data.project_organ={json.dumps(project_organ)}",
        "--trainer.accelerator=cpu",
        # The fake dataset's training split is smaller than a real batch.
        "--data.batch_size=8",
    ]


def test_cli_links_data_settings_into_the_model(fake_dataset, project_organ, tmp_path) -> None:
    cli = build_cli(_args("mmdae_organ.yaml", tmp_path, project_organ), run=False)
    assert cli.model.hparams.n_genes == cli.datamodule.n_genes == 16
    assert cli.model.hparams.n_classes == cli.datamodule.n_classes == 2
    assert cli.model.hparams.xavier_init and not cli.model.hparams.glorot_init
    cli = build_cli(_args("tybalt.yaml", tmp_path, project_organ), run=False)
    assert cli.model.hparams.n_classes == 0
    assert cli.model.hparams.glorot_init and not cli.model.hparams.xavier_init


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_runs_a_batch(
    config, fake_dataset, project_organ, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)  # configs name relative run directories
    build_cli(["fit", *_args(config, tmp_path, project_organ), "--trainer.fast_dev_run=true"])


@pytest.mark.parametrize(
    ("config", "fold", "latent_dim"), [("tybalt.yaml", 0, 256), ("mmdae_organ.yaml", 2, 256)]
)
def test_one_fold_fit_then_embed(
    config, fold, latent_dim, fake_dataset, project_organ, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    args = [
        "fit",
        *_args(config, tmp_path / "runs", project_organ),
        "--trainer.max_epochs=2",
        "--trainer.enable_progress_bar=false",
        f"--data.fold={fold}",
    ]
    cli = build_cli(args)
    run = tmp_path / "runs" / f"fold{fold}"
    best = run / "checkpoints" / "best.ckpt"
    assert Path(cli.trainer.checkpoint_callback.best_model_path) == best
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == ["best.ckpt", "last.ckpt"]
    assert (run / "config.yaml").is_file() and (run / "metrics.csv").is_file()

    # A rerun replaces the fold's run: no version_N directory, no best-v1.ckpt.
    build_cli(args)
    assert [p.name for p in (tmp_path / "runs").iterdir()] == [f"fold{fold}"]
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == ["best.ckpt", "last.ckpt"]

    sample_index, embeddings, folds = read_embeddings(
        embed(best, tmp_path / "out" / f"fold{fold}.parquet", accelerator="cpu")
    )
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), latent_dim)
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}


TINY = {
    "tybalt": (
        dict(quantification="unstranded", transform="lognorm", top_genes=10, scaling="minmax"),
        dict(latent_dim=4),
    ),
    "mmdae_organ": (
        dict(
            quantification="tpm_unstranded",
            transform="log1p",
            scaling="zscore",
            supervision="organ",
        ),
        dict(
            latent_dim=4,
            hidden_factor=0.5,
            heads="linear",
            reconstruction="mse",
            regularizer="mmd",
            logvar_max=0.0,
            glorot_init=False,
            xavier_init=True,
        ),
    ),
}


@pytest.mark.parametrize("variant", sorted(TINY))
def test_embedding_ignores_held_out_rows(
    variant, fake_dataset, project_organ, monkeypatch, tmp_path
) -> None:
    """A tiny model fit on fold 3, saved, then embedded with `vae-embed`'s own function."""
    data, model = TINY[variant]
    dm = VAEDataModule(**data, project_organ=project_organ, batch_size=8, fold=3)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(LitVAE(n_genes=dm.n_genes, n_classes=dm.n_classes, **model), datamodule=dm)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)
    assert_embedding_ignores_held_out(
        lambda out: embed(ckpt, out, accelerator="cpu"), 3, monkeypatch, tmp_path
    )
