from pathlib import Path

import lightning as L
import pytest

from reimp_bulkformer.cli import build_cli
from reimp_bulkformer.embed import embed
from reimp_bulkformer.lit import LitBulkFormer
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


def test_a_fit_lands_in_its_folds_run_directory(fake_dataset, tmp_path) -> None:
    """Fold 2 of a run rooted at `root` writes `root/fold2/`; embed reads its checkpoint."""
    root = tmp_path / "runs" / "bulkformer_debug"
    build_cli(
        [
            "fit",
            "--config",
            str(CONFIGS / "debug.yaml"),
            "--trainer.accelerator=cpu",
            "--trainer.enable_progress_bar=false",
            f"--trainer.default_root_dir={root}",
            "--data.batch_size=8",
            "--data.fold=2",
        ]
    )
    assert [p.name for p in root.iterdir()] == ["fold2"]
    run_dir = root / "fold2"
    assert [p.name for p in (run_dir / "checkpoints").iterdir()] == ["last.ckpt"]
    assert "config.yaml" in {p.name for p in run_dir.iterdir()}
    assert not any(p.name.startswith("version_") for p in run_dir.rglob("*"))

    ckpt = run_dir / "checkpoints" / "last.ckpt"
    out = embed(ckpt, tmp_path / "embeddings.parquet", accelerator="cpu")
    assert set(read_embeddings(out)[2].tolist()) == {2}


def test_the_tcga_config_saves_best_and_last_with_its_logs(fake_dataset, tmp_path) -> None:
    root = tmp_path / "runs" / "bulkformer"
    build_cli(
        [
            "fit",
            "--config",
            str(CONFIGS / "tcga.yaml"),
            "--trainer.accelerator=cpu",
            "--trainer.enable_progress_bar=false",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=1",
            f"--trainer.default_root_dir={root}",
            "--model.d_model=16",
            "--model.n_heads=2",
            "--model.n_layers=1",
            "--model.graph_k=5",
            "--data.batch_size=8",
            "--data.fold=2",
        ]
    )
    run_dir = root / "fold2"
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert ckpts == ["best.ckpt", "last.ckpt"]
    assert {"config.yaml", "metrics.csv"} <= {p.name for p in run_dir.iterdir()}
    assert not any(p.name.startswith("version_") for p in run_dir.rglob("*"))


def _checkpoint(tmp_path: Path, fold: int) -> Path:
    """A tiny BulkFormer fit on fold `fold` of the fake dataset, saved."""
    dm = ExpressionDataModule("tpm_unstranded", transform="log1p", batch_size=8, fold=fold)
    model = LitBulkFormer(n_genes=dm.n_genes, d_model=16, n_layers=1, n_heads=2, graph_k=5)
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
    """Embedding reads held-out samples but fits nothing on them: the graph is the checkpoint's."""
    ckpt = _checkpoint(tmp_path, fold=1)
    assert_embedding_ignores_held_out(
        lambda out: embed(ckpt, out, accelerator="cpu"), 1, monkeypatch, tmp_path
    )


@pytest.mark.parametrize(("fold", "pooling"), [(0, None), (2, "mean")])
def test_embed_from_a_checkpoint(fold, pooling, fake_dataset, tmp_path) -> None:
    ckpt = _checkpoint(tmp_path, fold)
    out = embed(ckpt, tmp_path / "embeddings.parquet", pooling=pooling, accelerator="cpu")
    sample_index, embeddings, folds = read_embeddings(out)
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 16)
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}
