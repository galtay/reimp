"""`tabssl-embed` — encoder embeddings for every sample from a trained checkpoint.

Writes the shared embeddings parquet (`sample_index`, `embedding`, and the
`fold` the checkpoint was trained for) that `reimp-shared probe` scores.
Each objective has its own directory; by default the file lands at
`out/tabssl_<objective>/fold<k>.parquet`:

  tabssl-embed --ckpt runs/tabssl_scarf/fold0/checkpoints/last.ckpt
  tabssl-embed --ckpt <ckpt> --out out/tabssl_debug/scarf/fold0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from reimp_shared.data import ExpressionDataModule
from reimp_shared.eval import write_embeddings
from reimp_tabssl.lit import LitTabSSL


def default_out(objective: str, fold: int, root: Path | str = "out") -> Path:
    """`<root>/tabssl_<objective>/fold<k>.parquet`: one directory per objective."""
    return Path(root) / f"tabssl_{objective}" / f"fold{fold}.parquet"


def embed(
    ckpt_path: Path | str,
    out_path: Path | str | None = None,
    batch_size: int | None = None,
    accelerator: str = "auto",
) -> Path:
    """Embed every sample the checkpoint's DataModule selects, with its own settings.

    The inputs are scaled by the checkpoint's training-set scaler and not
    corrupted; the encoder runs in eval mode.
    """
    model = LitTabSSL.load_from_checkpoint(ckpt_path, map_location="cpu")
    overrides = {} if batch_size is None else {"batch_size": batch_size}
    datamodule = ExpressionDataModule.load_from_checkpoint(ckpt_path, **overrides)
    fold = datamodule.hparams.fold
    if out_path is None:
        out_path = default_out(model.hparams.objective, fold)
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=False)
    outputs = trainer.predict(model, datamodule=datamodule)
    sample_index = torch.cat([o["sample_index"] for o in outputs]).numpy()
    embeddings = torch.cat([o["embedding"] for o in outputs]).float().numpy()
    return write_embeddings(out_path, sample_index, embeddings, fold)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tabssl-embed", description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: out/tabssl_<objective>/fold<k>.parquet"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    out = embed(args.ckpt, args.out, args.batch_size, args.accelerator)
    print(f"wrote {out}")
