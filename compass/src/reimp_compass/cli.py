"""`compass` — LightningCLI over `LitCompass` and the shared `ExpressionDataModule`.

compass fit --config compass/configs/debug.yaml
compass fit --config compass/configs/tcga.yaml --data.fold 2
compass fit --config compass/configs/tcga.yaml --model.negatives same_project
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from reimp_compass.lit import LitCompass
from reimp_shared.data import ExpressionDataModule


class CompassCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> CompassCLI:
    return CompassCLI(
        LitCompass,
        ExpressionDataModule,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=run,
    )


def main() -> None:
    build_cli()
