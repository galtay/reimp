"""`tifbert` — LightningCLI over `LitTifBERT` and the shared `ExpressionDataModule`.

tifbert fit --config tifbert/configs/debug.yaml
tifbert fit --config tifbert/configs/tcga_base.yaml --data.fold 2

Fold k runs in `<trainer.default_root_dir>/fold<k>/` (`FoldCLI`), e.g.
`runs/tifbert_base/fold2/checkpoints/best.ckpt` for the second command.
"""

from __future__ import annotations

from reimp_shared.data import ExpressionDataModule
from reimp_shared.foldcli import FoldCLI
from reimp_tifbert.lit import LitTifBERT


class TifBERTCLI(FoldCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The vocabulary is one token per gene of the DataModule's selection,
        # so its size is read off the instantiated DataModule.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> TifBERTCLI:
    return TifBERTCLI(LitTifBERT, ExpressionDataModule, args=args, run=run)


def main() -> None:
    build_cli()
