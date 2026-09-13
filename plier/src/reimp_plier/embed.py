"""`plier-embed` — every sample's embedding from a fitted fold.

Each sample is clipped to each gene's training range, z-scored with the
fold's training means and SDs and projected, B = (ZᵀZ + λ2 I)⁻¹Zᵀy — the
same map for training, validation and test samples. Writes the shared
embeddings parquet (`sample_index`, `embedding`, and the `fold` the model
was trained for) that `reimp-shared probe` scores:

  plier-embed --model runs/plier/fold0 --out out/plier/fold0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from reimp_plier.pipeline import CONFIG_FILE, FoldModel
from reimp_shared.data import load_expression
from reimp_shared.eval import write_embeddings


def embed(model_dir: Path | str, out_path: Path | str) -> Path:
    """Embed every sample the fold's data config selects."""
    model_dir = Path(model_dir)
    config = yaml.safe_load((model_dir / CONFIG_FILE).read_text())
    fold = FoldModel.load(model_dir)
    data = load_expression(**config["data"])
    embeddings = fold.embed(data)
    sample_index = data.samples["sample_index"].to_numpy()
    return write_embeddings(out_path, sample_index, embeddings, config["data"]["fold"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plier-embed", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", type=Path, required=True, help="a fitted fold's directory, <out_dir>/fold<k>"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"wrote {embed(args.model, args.out)}")
