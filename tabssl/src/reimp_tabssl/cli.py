"""`tabssl` — LightningCLI over `LitTabSSL` and the shared `ExpressionDataModule`.

tabssl fit --config tabssl/configs/debug.yaml --model.objective vime
tabssl fit --config tabssl/configs/tcga_scarf.yaml --data.fold 2
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from reimp_shared.data import ExpressionDataModule
from reimp_tabssl.lit import LitTabSSL


class TabSSLCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> TabSSLCLI:
    return TabSSLCLI(
        LitTabSSL,
        ExpressionDataModule,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=run,
    )


def main() -> None:
    build_cli()
