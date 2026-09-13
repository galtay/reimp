from pathlib import Path

import lightning as L
import pytest

from reimp_shared import hub
from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import read_embeddings
from reimp_tabssl.cli import build_cli
from reimp_tabssl.embed import embed
from reimp_tabssl.lit import OBJECTIVES, LitTabSSL

CONFIGS = Path(__file__).parents[1] / "configs"


def test_cli_links_the_gene_count_into_the_model(fake_dataset, tmp_path) -> None:
    cli = build_cli(
        ["--config", str(CONFIGS / "debug.yaml"), f"--trainer.default_root_dir={tmp_path}"],
        run=False,
    )
    assert cli.model.hparams.n_genes == cli.datamodule.n_genes


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_runs_a_batch(config, fake_dataset, tmp_path) -> None:
    build_cli(
        [
            "fit",
            "--config",
            str(CONFIGS / config),
            "--trainer.fast_dev_run=true",
            "--trainer.accelerator=cpu",
            f"--trainer.default_root_dir={tmp_path}",
            # The fake dataset's training split is smaller than a real batch.
            "--data.batch_size=8",
        ]
    )


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_debug_config_fits_and_embeds_one_fold(
    objective, fake_dataset, tmp_path, monkeypatch
) -> None:
    """The debug config end to end for each objective: fit, then embed from the last checkpoint."""
    monkeypatch.chdir(tmp_path)
    build_cli(
        [
            "fit",
            "--config",
            str(CONFIGS / "debug.yaml"),
            f"--model.objective={objective}",
            "--trainer.accelerator=cpu",
            "--trainer.enable_progress_bar=false",
            f"--trainer.default_root_dir={tmp_path / 'runs'}",
            "--data.batch_size=8",
            "--data.fold=1",
        ]
    )
    ckpt = tmp_path / "runs" / "checkpoints" / "last.ckpt"
    assert ckpt.exists()
    out = embed(ckpt, accelerator="cpu")
    # Each objective writes its own directory, one file per fold.
    assert out == Path("out") / f"tabssl_{objective}" / "fold1.parquet"
    sample_index, embeddings, folds = read_embeddings(out)
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 256)
    assert set(folds.tolist()) == {1}


@pytest.mark.parametrize("fold", [0, 2])
def test_embed_from_a_checkpoint(fold, fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(transform="lognorm", batch_size=8, fold=fold)
    model = LitTabSSL(n_genes=dm.n_genes, objective="byol", hidden_dim=16, byol_hidden_dim=32)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=dm)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)

    sample_index, embeddings, folds = read_embeddings(
        embed(ckpt, tmp_path / "embeddings.parquet", accelerator="cpu")
    )
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 16)
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}
