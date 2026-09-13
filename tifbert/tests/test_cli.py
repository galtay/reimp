from pathlib import Path

import lightning as L
import numpy as np
import pytest

from reimp_shared import hub
from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import read_embeddings
from reimp_shared.testing import assert_embedding_ignores_held_out
from reimp_tifbert.cli import build_cli
from reimp_tifbert.embed import embed
from reimp_tifbert.lit import LitTifBERT

CONFIGS = Path(__file__).parents[1] / "configs"


def test_cli_sizes_the_vocabulary_from_the_data(fake_dataset, tmp_path) -> None:
    cli = build_cli(
        ["--config", str(CONFIGS / "debug.yaml"), f"--trainer.default_root_dir={tmp_path}"],
        run=False,
    )
    assert cli.model.hparams.n_genes == cli.datamodule.n_genes
    assert cli.model.model.token_embedding.num_embeddings == cli.datamodule.n_genes + 2


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
            # The fake dataset's training split is smaller than a real batch,
            # and its 16 genes fit in one short window.
            "--data.batch_size=8",
            "--model.window=16",
            "--model.stride=8",
        ]
    )


@pytest.mark.parametrize(
    ("config", "checkpoint"), [("debug.yaml", "last.ckpt"), ("tcga_base.yaml", "best.ckpt")]
)
def test_a_fold_runs_in_its_own_directory(config, checkpoint, fake_dataset, tmp_path) -> None:
    root = tmp_path / "runs"
    args = [
        "fit",
        "--config",
        str(CONFIGS / config),
        "--data.fold=2",
        f"--trainer.default_root_dir={root}",
        "--trainer.accelerator=cpu",
        "--trainer.max_epochs=1",
        "--trainer.enable_progress_bar=false",
        "--data.batch_size=8",
        "--model.window=16",
        "--model.stride=8",
        "--model.d_model=16",
        "--model.n_layers=1",
        "--model.n_heads=2",
    ]
    build_cli(args)
    build_cli(args)  # a rerun replaces the fold's run
    run = root / "fold2"
    assert [p.name for p in root.iterdir()] == ["fold2"]
    assert (run / "config.yaml").is_file()
    ckpt = run / "checkpoints" / checkpoint
    assert ckpt.is_file()
    assert not list(run.rglob("version_*")) and not list(run.rglob("*-v1.ckpt"))
    # The documented path is the one to embed from, and it knows its fold.
    _, _, folds = read_embeddings(embed(ckpt, tmp_path / "fold2.parquet", accelerator="cpu"))
    assert set(folds.tolist()) == {2}


def _checkpoint(tmp_path: Path, fold: int) -> Path:
    """A tiny TifBERT fit on fold `fold` of the fake dataset, saved."""
    dm = ExpressionDataModule(quantification="tpm_unstranded", batch_size=8, fold=fold)
    model = LitTifBERT(n_genes=dm.n_genes, window=8, stride=4, d_model=16, n_layers=1, n_heads=2)
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
    return ckpt


def test_embedding_ignores_held_out_rows(fake_dataset, monkeypatch, tmp_path) -> None:
    ckpt = _checkpoint(tmp_path, fold=1)
    assert_embedding_ignores_held_out(
        lambda out: embed(ckpt, out, accelerator="cpu"), 1, monkeypatch, tmp_path
    )


@pytest.mark.parametrize("fold", [0, 2])
def test_embed_from_a_checkpoint(fold, fake_dataset, tmp_path) -> None:
    ckpt = _checkpoint(tmp_path, fold)
    sample_index, embeddings, folds = read_embeddings(
        embed(ckpt, tmp_path / "embeddings.parquet", accelerator="cpu")
    )
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 16)
    assert np.isfinite(embeddings).all()
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}
