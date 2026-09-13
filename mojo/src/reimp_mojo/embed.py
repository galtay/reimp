"""`mojo-embed` — pooled-position embeddings for every sample from a trained checkpoint.

Writes the shared embeddings parquet (`sample_index`, `embedding`, and the
`fold` the checkpoint was trained for) that `reimp-shared probe` scores:

  mojo-embed --ckpt runs/mojo/checkpoints/last.ckpt --out out/mojo/fold0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from reimp_mojo.lit import LitMOJO
from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import write_embeddings


def embed(
    ckpt_path: Path | str,
    out_path: Path | str,
    batch_size: int | None = None,
    accelerator: str = "auto",
) -> Path:
    """Embed every sample the checkpoint's DataModule selects, with its own settings.

    The tokenizer's maximum comes with the checkpoint: the one fit on that
    fold's training rows.
    """
    model = LitMOJO.load_from_checkpoint(ckpt_path, map_location="cpu")
    overrides = {} if batch_size is None else {"batch_size": batch_size}
    datamodule = ExpressionDataModule.load_from_checkpoint(ckpt_path, **overrides)
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=False)
    outputs = trainer.predict(model, datamodule=datamodule)
    sample_index = torch.cat([o["sample_index"] for o in outputs]).numpy()
    embeddings = torch.cat([o["embedding"] for o in outputs]).float().numpy()
    return write_embeddings(out_path, sample_index, embeddings, datamodule.hparams.fold)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mojo-embed", description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    out = embed(args.ckpt, args.out, args.batch_size, args.accelerator)
    print(f"wrote {out}")
