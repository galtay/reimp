"""`vae-embed` — posterior means for every sample from a trained checkpoint.

Writes the shared embeddings parquet (`sample_index`, `embedding`, and the
`fold` the checkpoint was trained for) that `reimp-shared probe` scores. The
checkpoint carries the scaler fit on its fold's training rows, so every
sample is scaled as in training:

  vae-embed --ckpt runs/tybalt/version_0/checkpoints/best.ckpt --out out/tybalt/fold0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from reimp_shared.eval import write_embeddings
from reimp_vae.data import VAEDataModule
from reimp_vae.lit import LitVAE


def embed(
    ckpt_path: Path | str,
    out_path: Path | str,
    batch_size: int | None = None,
    accelerator: str = "auto",
) -> Path:
    """Embed every sample the checkpoint's DataModule selects, with its own settings."""
    model = LitVAE.load_from_checkpoint(ckpt_path, map_location="cpu")
    overrides = {} if batch_size is None else {"batch_size": batch_size}
    datamodule = VAEDataModule.load_from_checkpoint(ckpt_path, **overrides)
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=False)
    outputs = trainer.predict(model, datamodule=datamodule)
    sample_index = torch.cat([o["sample_index"] for o in outputs]).numpy()
    embeddings = torch.cat([o["embedding"] for o in outputs]).float().numpy()
    return write_embeddings(out_path, sample_index, embeddings, datamodule.hparams.fold)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vae-embed", description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    out = embed(args.ckpt, args.out, args.batch_size, args.accelerator)
    print(f"wrote {out}")
