"""`compass-embed` — concept and set scores for every sample from a trained checkpoint.

Writes the shared embeddings parquet (`sample_index`, `embedding`, and the
`fold` the checkpoint was trained for) that `reimp-shared probe` scores:
the 43 concept scores to `--out`, and with `--sets-out` the 132 set scores
to a second file.

  compass-embed --ckpt runs/compass/checkpoints/best.ckpt \\
      --out out/compass/fold0.parquet --sets-out out/compass_sets/fold0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from reimp_compass.lit import LitCompass
from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import write_embeddings


def embed(
    ckpt_path: Path | str,
    out_path: Path | str,
    sets_out_path: Path | str | None = None,
    batch_size: int | None = None,
    accelerator: str = "auto",
) -> tuple[Path, Path | None]:
    """Embed every sample the checkpoint's DataModule selects, with its own settings.

    Returns the concept file and the set file (None without `sets_out_path`).
    """
    model = LitCompass.load_from_checkpoint(ckpt_path, map_location="cpu")
    overrides = {} if batch_size is None else {"batch_size": batch_size}
    datamodule = ExpressionDataModule.load_from_checkpoint(ckpt_path, **overrides)
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=False)
    outputs = trainer.predict(model, datamodule=datamodule)
    sample_index = torch.cat([o["sample_index"] for o in outputs]).numpy()
    fold = datamodule.hparams.fold

    def write(path: Path | str, key: str) -> Path:
        scores = torch.cat([o[key] for o in outputs]).float().numpy()
        return write_embeddings(path, sample_index, scores, fold)

    concepts = write(out_path, "concepts")
    sets = None if sets_out_path is None else write(sets_out_path, "sets")
    return concepts, sets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="compass-embed", description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="concept scores (43-d)")
    parser.add_argument("--sets-out", type=Path, default=None, help="set scores (132-d)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    for path in embed(args.ckpt, args.out, args.sets_out, args.batch_size, args.accelerator):
        if path is not None:
            print(f"wrote {path}")
