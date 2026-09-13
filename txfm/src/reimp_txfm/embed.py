"""`txfm-embed` — CLS embeddings for every sample from a trained checkpoint.

Writes the shared embeddings parquet (`sample_index`, `embedding`, and the
`fold` the checkpoint was trained for) that `reimp-shared probe` scores:

  txfm-embed --ckpt runs/txfm_s/version_0/checkpoints/last.ckpt --out out/txfm_s/fold0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import write_embeddings
from reimp_txfm.lit import LitTxFM


def embed(
    ckpt_path: Path | str,
    out_path: Path | str,
    draws: int = 1,
    seed: int = 0,
    batch_size: int | None = None,
    accelerator: str = "auto",
) -> Path:
    """Embed every sample the checkpoint's DataModule selects, with its own settings."""
    model = LitTxFM.load_from_checkpoint(
        ckpt_path, map_location="cpu", embed_draws=draws, seed=seed
    )
    overrides = {} if batch_size is None else {"batch_size": batch_size}
    datamodule = ExpressionDataModule.load_from_checkpoint(ckpt_path, **overrides)
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=False)
    outputs = trainer.predict(model, datamodule=datamodule)
    sample_index = torch.cat([o["sample_index"] for o in outputs]).numpy()
    embeddings = torch.cat([o["embedding"] for o in outputs]).float().numpy()
    return write_embeddings(out_path, sample_index, embeddings, datamodule.hparams.fold)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="txfm-embed", description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1, help="masks to average per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    out = embed(args.ckpt, args.out, args.draws, args.seed, args.batch_size, args.accelerator)
    print(f"wrote {out}")
