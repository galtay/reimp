from pathlib import Path

import lightning as L
import pytest

from reimp_compass.cli import build_cli
from reimp_compass.embed import embed, main
from reimp_compass.lit import LitCompass
from reimp_shared import hub
from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import read_embeddings
from reimp_shared.testing import assert_embedding_ignores_held_out

CONFIGS = Path(__file__).parents[1] / "configs"

pytestmark = pytest.mark.filterwarnings("ignore:.*concept genes")


def _fake_overrides(hierarchy_path: str, tmp_path) -> list[str]:
    """The fake dataset has neither the concept genes' IDs nor their symbols."""
    return [
        "--data.gene_ids_path=null",
        f"--model.hierarchy_path={hierarchy_path}",
        f"--trainer.default_root_dir={tmp_path}",
    ]


def test_cli_links_the_gene_count_into_the_model(fake_dataset, hierarchy_path, tmp_path) -> None:
    cli = build_cli(
        ["--config", str(CONFIGS / "debug.yaml"), *_fake_overrides(hierarchy_path, tmp_path)],
        run=False,
    )
    assert cli.model.hparams.n_genes == cli.datamodule.n_genes


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_runs_a_batch(config, fake_dataset, hierarchy_path, tmp_path) -> None:
    build_cli(
        [
            "fit",
            "--config",
            str(CONFIGS / config),
            *_fake_overrides(hierarchy_path, tmp_path),
            "--trainer.fast_dev_run=true",
            "--trainer.accelerator=cpu",
            # The fake dataset's training split is smaller than a real batch.
            "--data.batch_size=8",
        ]
    )


def test_fit_runs_in_the_folds_directory(fake_dataset, hierarchy_path, tmp_path) -> None:
    build_cli(
        [
            "fit",
            "--config",
            str(CONFIGS / "tcga.yaml"),
            *_fake_overrides(hierarchy_path, tmp_path),
            "--data.fold=2",
            "--data.batch_size=8",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=1",
            "--trainer.accelerator=cpu",
            "--trainer.enable_progress_bar=false",
        ]
    )
    run = tmp_path / "fold2"
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == ["best.ckpt", "last.ckpt"]
    assert (run / "config.yaml").is_file() and (run / "metrics.csv").is_file()
    assert not list(tmp_path.rglob("version_*"))
    best = run / "checkpoints" / "best.ckpt"
    concepts, _ = embed(best, tmp_path / "f.parquet", accelerator="cpu")
    assert set(read_embeddings(concepts)[2].tolist()) == {2}


def _checkpoint(fold: int, hierarchy_path: str, tmp_path) -> Path:
    """A tiny COMPASS fit on fold `fold` of the fake dataset, saved."""
    dm = ExpressionDataModule("tpm_unstranded", transform="log1p", batch_size=8, fold=fold)
    model = LitCompass(
        n_genes=dm.n_genes, hierarchy_path=hierarchy_path, d_model=8, head_dim=4, dim_ff=16
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
    return ckpt


@pytest.mark.parametrize("output", ["concepts", "sets"])
def test_embedding_ignores_held_out_rows(
    output, fake_dataset, hierarchy_path, monkeypatch, tmp_path
) -> None:
    # Embedding scales with the checkpoint's training-row min-max, never refit.
    ckpt = _checkpoint(1, hierarchy_path, tmp_path)

    def embed_one(out: Path) -> Path:
        concepts, sets = embed(ckpt, out, out.with_name(f"sets-{out.name}"), accelerator="cpu")
        return concepts if output == "concepts" else sets

    assert_embedding_ignores_held_out(embed_one, 1, monkeypatch, tmp_path)


@pytest.mark.parametrize("fold", [0, 2])
def test_embed_writes_concept_and_set_scores(fold, fake_dataset, hierarchy_path, tmp_path) -> None:
    ckpt = _checkpoint(fold, hierarchy_path, tmp_path)
    concepts, sets = embed(
        ckpt, tmp_path / "compass" / "f.parquet", tmp_path / "sets" / "f.parquet", accelerator="cpu"
    )
    everyone = hub.load_samples()["sample_index"].tolist()
    for path, width in [(concepts, 3), (sets, 6)]:
        sample_index, embeddings, folds = read_embeddings(path)
        assert sample_index.tolist() == everyone
        assert embeddings.shape == (len(sample_index), width)
        # A fold's model records its fold in every row, for the probes to pool.
        assert set(folds.tolist()) == {fold}


def test_embed_command_line(fake_dataset, hierarchy_path, tmp_path, capsys) -> None:
    ckpt = _checkpoint(0, hierarchy_path, tmp_path)
    out = tmp_path / "out" / "compass" / "fold0.parquet"
    main(["--ckpt", str(ckpt), "--out", str(out), "--accelerator", "cpu"])
    assert read_embeddings(out.parent)[1].shape[1] == 3
    assert f"wrote {out}" in capsys.readouterr().out
