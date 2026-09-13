from pathlib import Path

import lightning as L
import pytest

from reimp_bulkformer.cli import build_cli
from reimp_bulkformer.embed import embed
from reimp_bulkformer.lit import LitBulkFormer
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


@pytest.mark.parametrize(("fold", "pooling"), [(0, None), (2, "mean")])
def test_embed_from_a_checkpoint(fold, pooling, fake_dataset, tmp_path) -> None:
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

    out = embed(ckpt, tmp_path / "embeddings.parquet", pooling=pooling, accelerator="cpu")
    sample_index, embeddings, folds = read_embeddings(out)
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 16)
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {fold}
