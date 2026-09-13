"""`bulkformer` — LightningCLI over `LitBulkFormer` and the shared `ExpressionDataModule`.

bulkformer fit --config bulkformer/configs/debug.yaml
bulkformer fit --config bulkformer/configs/tcga.yaml --data.fold 2
bulkformer fit --config bulkformer/configs/tcga.yaml --model.graph none   # Performer only
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from reimp_bulkformer.lit import LitBulkFormer
from reimp_shared.data import ExpressionDataModule


class BulkFormerCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> BulkFormerCLI:
    return BulkFormerCLI(
        LitBulkFormer,
        ExpressionDataModule,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=run,
    )


def main() -> None:
    build_cli()
