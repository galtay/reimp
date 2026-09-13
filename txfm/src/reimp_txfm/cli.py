"""`txfm` — LightningCLI over `LitTxFM` and the shared `ExpressionDataModule`.

txfm fit --config txfm/configs/debug.yaml
txfm fit --config txfm/configs/tcga_s.yaml --trainer.max_epochs 50
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from reimp_shared.data import ExpressionDataModule
from reimp_txfm.lit import LitTxFM


class TxFMCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The data normalizes each library to `library_size` and the output
        # activation is bounded by log(library_size + 1): one value, set once.
        parser.link_arguments("data.library_size", "model.library_size")
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> TxFMCLI:
    return TxFMCLI(
        LitTxFM,
        ExpressionDataModule,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=run,
    )


def main() -> None:
    build_cli()
