from pathlib import Path

import lightning as L
import pytest
import torch
import yaml

from reimp_shared.data import ExpressionDataModule
from reimp_shared.foldcli import FoldCLI


class Tiny(L.LightningModule):
    def __init__(self, n_genes: int = 16) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(n_genes, 1)

    def _loss(self, batch):
        return self.layer(batch["values"]).pow(2).mean()

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self._loss(batch))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _config(tmp_path: Path, **trainer) -> Path:
    config = {
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": [
                {
                    "class_path": "lightning.pytorch.loggers.CSVLogger",
                    "init_args": {"save_dir": str(tmp_path / "ignored")},
                }
            ],
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {"monitor": "val/loss", "filename": "best", "save_last": True},
                }
            ],
            **trainer,
        },
        "data": {"transform": "lognorm", "batch_size": 8, "fold": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _cli(*args: str, run: bool = True) -> FoldCLI:
    return FoldCLI(Tiny, ExpressionDataModule, args=list(args), run=run)


def test_trainer_loggers_and_checkpoints_point_at_the_folds_directory(fake_dataset, tmp_path):
    config = _config(tmp_path, default_root_dir=str(tmp_path / "runs" / "tiny"))
    cli = _cli("--config", str(config), run=False)
    run_dir = (tmp_path / "runs" / "tiny" / "fold2").resolve()
    assert Path(cli.trainer.default_root_dir).resolve() == run_dir
    assert Path(cli.trainer.loggers[0].log_dir).resolve() == run_dir
    assert Path(cli.trainer.checkpoint_callback.dirpath).resolve() == run_dir / "checkpoints"


def test_a_rerun_replaces_the_folds_run_and_other_folds_get_their_own(fake_dataset, tmp_path):
    root = tmp_path / "runs" / "tiny"
    config = _config(tmp_path, default_root_dir=str(root))
    for _ in range(2):
        _cli("fit", "--config", str(config))
    _cli("fit", "--config", str(config), "--data.fold=0")
    assert sorted(p.name for p in root.iterdir()) == ["fold0", "fold2"]
    run_dir = root / "fold2"
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["best.ckpt", "last.ckpt"]
    assert {"config.yaml", "metrics.csv"} <= {p.name for p in run_dir.iterdir()}
    assert not any(p.name.startswith("version_") for p in run_dir.rglob("*"))
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["data"]["fold"] == 2


def test_a_run_root_is_required(fake_dataset, tmp_path):
    with pytest.raises(ValueError, match="default_root_dir"):
        _cli("--config", str(_config(tmp_path)), run=False)
