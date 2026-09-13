from pathlib import Path

import lightning as L
import pytest

from reimp_bulkrnabert.cli import build_cli
from reimp_bulkrnabert.embed import embed
from reimp_bulkrnabert.lit import LitBulkRNABert
from reimp_shared import hub
from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import read_embeddings

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


def test_debug_config_checkpoints_under_its_root_dir(fake_dataset, tmp_path, monkeypatch) -> None:
    """`last.ckpt` follows `default_root_dir`, where the README's embed command reads it."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    root = tmp_path / "root"
    args = [
        "fit",
        "--config",
        str(CONFIGS / "debug.yaml"),
        "--trainer.accelerator=cpu",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=1",
        f"--trainer.default_root_dir={root}",
        "--data.batch_size=8",
    ]
    build_cli(args)
    build_cli(args)  # a rerun overwrites the checkpoint
    assert [p.name for p in (root / "checkpoints").iterdir()] == ["last.ckpt"]
    assert not any(cwd.iterdir())
    loaded = LitBulkRNABert.load_from_checkpoint(root / "checkpoints" / "last.ckpt")
    assert loaded.hparams.token_max is not None


@pytest.mark.parametrize("fold", [0, 2])
def test_embed_from_a_checkpoint(fold, fake_dataset, tmp_path) -> None:
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
