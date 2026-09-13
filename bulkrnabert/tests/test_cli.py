from pathlib import Path

import lightning as L
import pytest

from reimp_bulkrnabert.cli import build_cli
from reimp_bulkrnabert.embed import embed
from reimp_bulkrnabert.lit import LitBulkRNABert
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
    ("config", "checkpoints"),
    [("debug.yaml", ["last.ckpt"]), ("tcga.yaml", ["best.ckpt", "last.ckpt"])],
)
def test_a_fold_runs_in_its_own_directory(
    config, checkpoints, fake_dataset, tmp_path, monkeypatch
) -> None:
    """Fold 2 runs in `<root>/fold2/`, where the README's embed commands read its checkpoints."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    root = tmp_path / "root"
    args = [
        "fit",
        "--config",
        str(CONFIGS / config),
        "--trainer.accelerator=cpu",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=1",
        "--trainer.accumulate_grad_batches=1",
        f"--trainer.default_root_dir={root}",
        "--data.batch_size=8",
        "--data.fold=2",
        # The paper-size encoder, cut down to run in a test.
        "--model.d_model=16",
        "--model.n_layers=1",
        "--model.n_heads=2",
        "--model.dim_ff=32",
    ]
    build_cli(args)
    build_cli(args)  # a rerun of the fold replaces its run
    assert [p.name for p in root.iterdir()] == ["fold2"]
    assert sorted(p.name for p in (root / "fold2" / "checkpoints").iterdir()) == checkpoints
    assert (root / "fold2" / "config.yaml").exists()
    assert not any(cwd.iterdir())
    for name in checkpoints:
        loaded = LitBulkRNABert.load_from_checkpoint(root / "fold2" / "checkpoints" / name)
        assert loaded.hparams.token_max is not None


def _checkpoint(tmp_path: Path, fold: int) -> Path:
    """A tiny BulkRNABert fit on fold `fold` of the fake dataset, saved."""
    dm = ExpressionDataModule(
        quantification="tpm_unstranded", transform="log1p", batch_size=8, fold=fold
    )
    model = LitBulkRNABert(n_genes=dm.n_genes, d_model=16, n_layers=1, n_heads=2, dim_ff=32)
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
    """Embedding bins by the checkpoint's training maximum, never one refit on held-out rows."""
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
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}


def test_embed_refuses_an_untrained_checkpoint(fake_dataset, tmp_path) -> None:
    dm = ExpressionDataModule(quantification="tpm_unstranded", transform="log1p")
    model = LitBulkRNABert(n_genes=dm.n_genes, d_model=16, n_layers=1, n_heads=2, dim_ff=32)
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.strategy.connect(model)
    ckpt = tmp_path / "untrained.ckpt"
    trainer.save_checkpoint(ckpt)
    with pytest.raises(ValueError, match="tokenizer"):
        embed(ckpt, tmp_path / "embeddings.parquet", accelerator="cpu")
