"""`bulkrnabert` — LightningCLI over `LitBulkRNABert` and the shared `ExpressionDataModule`.

bulkrnabert fit --config bulkrnabert/configs/debug.yaml
bulkrnabert fit --config bulkrnabert/configs/tcga.yaml --data.fold 2

Fold k runs in `<trainer.default_root_dir>/fold<k>/` (`FoldCLI`), e.g.
`runs/bulkrnabert/fold2/checkpoints/best.ckpt`.
"""

from __future__ import annotations

from reimp_bulkrnabert.lit import LitBulkRNABert
from reimp_shared.data import ExpressionDataModule
from reimp_shared.foldcli import FoldCLI


class BulkRNABertCLI(FoldCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> BulkRNABertCLI:
    return BulkRNABertCLI(LitBulkRNABert, ExpressionDataModule, args=args, run=run)


def main() -> None:
    build_cli()
