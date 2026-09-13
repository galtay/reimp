from pathlib import Path

import lightning as L
import pytest

from reimp_mojo.cli import build_cli
from reimp_mojo.embed import embed
from reimp_mojo.lit import LitMOJO
from reimp_shared import hub
from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import read_embeddings
from reimp_shared.testing import assert_embedding_ignores_held_out

CONFIGS = Path(__file__).parents[1] / "configs"


def test_cli_links_the_gene_count_into_the_model(fake_dataset, tmp_path) -> None:
    cli = build_cli(
        ["--config", str(CONFIGS / "debug.yaml"), f"--trainer.default_root_dir={tmp_path}"],
        run=False,
    )
    assert cli.model.hparams.n_genes == cli.datamodule.n_genes
    assert cli.datamodule.hparams.quantification == "tpm_unstranded"
    assert cli.datamodule.hparams.transform == "log1p"


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


@pytest.mark.parametrize(
    "config, checkpoints",
    [("debug.yaml", ["last.ckpt"]), ("tcga.yaml", ["best.ckpt", "last.ckpt"])],
)
def test_a_fold_runs_in_its_own_directory(config, checkpoints, fake_dataset, tmp_path) -> None:
    root = tmp_path / "runs"
    args = [
        "fit",
        "--config",
        str(CONFIGS / config),
        "--trainer.accelerator=cpu",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=1",
        "--trainer.limit_val_batches=1",
        "--trainer.enable_progress_bar=false",
        f"--trainer.default_root_dir={root}",
        "--data.batch_size=8",
        "--data.fold=2",
    ]
    build_cli(args)
    run = root / "fold2"
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == checkpoints
    assert (run / "config.yaml").exists()
    # A rerun replaces the fold's run: no version_N directories or -v1 files.
    build_cli(args)
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == checkpoints
    assert not list(root.rglob("version_*"))
    # The documented path is what mojo-embed reads, and the fold travels with it.
    _, _, folds = read_embeddings(
        embed(run / "checkpoints" / checkpoints[0], tmp_path / "e.parquet", accelerator="cpu")
    )
    assert set(folds.tolist()) == {2}


def _checkpoint(tmp_path: Path, fold: int) -> tuple[Path, LitMOJO]:
    """A tiny MOJO fit on fold `fold` of the fake dataset, saved; and the model."""
    dm = ExpressionDataModule("tpm_unstranded", transform="log1p", batch_size=8, fold=fold)
    model = LitMOJO(
        n_genes=dm.n_genes, embed_dim=16, conv_channels=16, d_model=16, n_down=2, n_heads=2
    )
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
    return ckpt, model


def test_embedding_ignores_held_out_rows(fake_dataset, monkeypatch, tmp_path) -> None:
    # Embedding bins every sample by the checkpoint's training-row maximum;
    # refitting it on all samples would move the training samples' tokens.
    ckpt, _ = _checkpoint(tmp_path, fold=1)
    assert_embedding_ignores_held_out(
        lambda out: embed(ckpt, out, accelerator="cpu"), 1, monkeypatch, tmp_path
    )


@pytest.mark.parametrize("fold", [0, 2])
def test_embed_from_a_checkpoint(fold, fake_dataset, tmp_path) -> None:
    ckpt, model = _checkpoint(tmp_path, fold)
    # The fold's fitted tokenizer maximum travels with the checkpoint.
    restored = LitMOJO.load_from_checkpoint(ckpt, map_location="cpu")
    assert restored.tokenizer.max_ == model.tokenizer.max_ is not None

    sample_index, embeddings, folds = read_embeddings(
        embed(ckpt, tmp_path / "embeddings.parquet", accelerator="cpu")
    )
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 16)
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}
