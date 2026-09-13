"""`FoldCLI`: a LightningCLI whose runs land in one directory per fold.

Every reimp model trains once per cross-validation fold, so a run's
directory is named by its fold: `<trainer.default_root_dir>/fold<k>/`, with
`k` from `data.fold`. Before anything is instantiated, `FoldCLI` points the
trainer, every logger and every `ModelCheckpoint` there:

    runs/<model>/fold<k>/                config.yaml, metrics.csv, TensorBoard events
    runs/<model>/fold<k>/checkpoints/    e.g. best.ckpt, last.ckpt

Loggers get an empty `name` and `version`, and checkpoint version counters
are off, so rerunning a fold replaces that fold's run rather than adding a
`version_N` directory or a `-v1` checkpoint beside it. A config names the
model's run root in `trainer.default_root_dir`. Loggers still need a
`save_dir` to parse, conventionally that same root; it and any checkpoint
`dirpath` are overridden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightning.pytorch.cli import LightningCLI


class FoldCLI(LightningCLI):
    """LightningCLI with per-fold run directories; the saved config is overwritten on rerun."""

    def __init__(self, *args: Any, save_config_kwargs: dict | None = None, **kwargs: Any) -> None:
        save_config_kwargs = {"overwrite": True, **(save_config_kwargs or {})}
        super().__init__(*args, save_config_kwargs=save_config_kwargs, **kwargs)

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        fold_run_dir(self.config.get(str(self.subcommand), self.config))


def fold_run_dir(config) -> Path:
    """Point a parsed config's trainer, loggers and checkpoints at `<root>/fold<k>`; return it."""
    trainer = config.trainer
    if trainer.default_root_dir is None:
        raise ValueError(
            "set trainer.default_root_dir, the model's run root: fold k runs in <root>/fold<k>"
        )
    run_dir = Path(trainer.default_root_dir) / f"fold{config.data.fold}"
    trainer.default_root_dir = str(run_dir)
    loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
    for logger in loggers:
        # A class_path spec; `logger: false` (or true) has nothing to point.
        if hasattr(logger, "init_args"):
            logger.init_args.save_dir = str(run_dir)
            logger.init_args.name = ""
            logger.init_args.version = ""
    for callback in trainer.callbacks or []:
        if getattr(callback, "class_path", "").endswith("ModelCheckpoint"):
            callback.init_args.dirpath = str(run_dir / "checkpoints")
            callback.init_args.enable_version_counter = False
    return run_dir
