import json
from pathlib import Path

import pytest

from reimp_shared import hub
from reimp_shared.eval import read_embeddings
from reimp_vae.cli import build_cli
from reimp_vae.embed import embed

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
    cli = build_cli(_args("tybalt.yaml", tmp_path, project_organ), run=False)
    assert cli.model.hparams.n_classes == 0


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_runs_a_batch(
    config, fake_dataset, project_organ, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)  # configs name relative run directories
    build_cli(["fit", *_args(config, tmp_path, project_organ), "--trainer.fast_dev_run=true"])


@pytest.mark.parametrize(
    ("config", "fold", "latent_dim"), [("tybalt.yaml", 0, 100), ("mmdae_organ.yaml", 2, 121)]
)
def test_one_fold_fit_then_embed(
    config, fold, latent_dim, fake_dataset, project_organ, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cli = build_cli(
        [
            "fit",
            *_args(config, tmp_path, project_organ),
            "--trainer.max_epochs=2",
            "--trainer.enable_progress_bar=false",
            f"--data.fold={fold}",
        ]
    )
    best = Path(cli.trainer.checkpoint_callback.best_model_path)
    assert best.name == "best.ckpt"

    sample_index, embeddings, folds = read_embeddings(
        embed(best, tmp_path / "out" / f"fold{fold}.parquet", accelerator="cpu")
    )
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), latent_dim)
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}
