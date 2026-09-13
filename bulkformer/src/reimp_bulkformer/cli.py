"""`bulkformer` — LightningCLI over `LitBulkFormer` and the shared `ExpressionDataModule`.

bulkformer fit --config bulkformer/configs/debug.yaml
bulkformer fit --config bulkformer/configs/tcga.yaml --data.fold 2
bulkformer fit --config bulkformer/configs/tcga.yaml --model.graph none   # Performer only

Fold k runs in `<trainer.default_root_dir>/fold<k>/` (`FoldCLI`), e.g.
`runs/bulkformer/fold2/checkpoints/best.ckpt`; rerunning a fold replaces it.
"""

from __future__ import annotations

from reimp_bulkformer.lit import LitBulkFormer
from reimp_shared.data import ExpressionDataModule
from reimp_shared.foldcli import FoldCLI


class BulkFormerCLI(FoldCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> BulkFormerCLI:
    return BulkFormerCLI(LitBulkFormer, ExpressionDataModule, args=args, run=run)


def main() -> None:
    build_cli()
