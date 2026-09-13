"""`mojo` — LightningCLI over `LitMOJO` and the shared `ExpressionDataModule`.

mojo fit --config mojo/configs/debug.yaml
mojo fit --config mojo/configs/tcga.yaml --data.fold 2
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from reimp_mojo.lit import LitMOJO
from reimp_shared.data import ExpressionDataModule


class MOJOCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> MOJOCLI:
    return MOJOCLI(
        LitMOJO,
        ExpressionDataModule,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=run,
    )


def main() -> None:
    build_cli()
