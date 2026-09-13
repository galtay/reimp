"""`tifbert-embed` — sample embeddings for every sample from a trained checkpoint.

Each sample is ranked by the checkpoint's own gene ranker (fit on its
fold's training samples), cut into windows, and embedded as the mean over
its windows of their mean-pooled hidden states. Writes the shared
embeddings parquet (`sample_index`, `embedding`, and the `fold` the
checkpoint was trained for) that `reimp-shared probe` scores:

  tifbert-embed --ckpt runs/tifbert_base/fold0/checkpoints/best.ckpt \
      --out out/tifbert/fold0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import write_embeddings
from reimp_tifbert.lit import LitTifBERT


def embed(
    ckpt_path: Path | str,
    out_path: Path | str,
    batch_size: int | None = None,
    embed_chunk: int | None = None,
    accelerator: str = "auto",
) -> Path:
    """Embed every sample the checkpoint's DataModule selects, with its own settings."""
    overrides = {} if embed_chunk is None else {"embed_chunk": embed_chunk}
    model = LitTifBERT.load_from_checkpoint(ckpt_path, map_location="cpu", **overrides)
    if model.ranker is None:
        raise ValueError(f"{ckpt_path}: no gene ranker in the checkpoint")
    overrides = {} if batch_size is None else {"batch_size": batch_size}
    datamodule = ExpressionDataModule.load_from_checkpoint(ckpt_path, **overrides)
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=False)
    outputs = trainer.predict(model, datamodule=datamodule)
    sample_index = torch.cat([o["sample_index"] for o in outputs]).numpy()
    embeddings = torch.cat([o["embedding"] for o in outputs]).float().cpu().numpy()
    return write_embeddings(out_path, sample_index, embeddings, datamodule.hparams.fold)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tifbert-embed", description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None, help="samples per batch")
    parser.add_argument(
        "--embed-chunk", type=int, default=None, help="windows encoded at once (memory)"
    )
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    out = embed(args.ckpt, args.out, args.batch_size, args.embed_chunk, args.accelerator)
    print(f"wrote {out}")
