"""`mojo` — LightningCLI over `LitMOJO` and the shared `ExpressionDataModule`.

mojo fit --config mojo/configs/debug.yaml
mojo fit --config mojo/configs/tcga.yaml --data.fold 2

Fold k runs in `<trainer.default_root_dir>/fold<k>/` (`FoldCLI`), e.g.
`runs/mojo/fold2/checkpoints/best.ckpt`.
"""

from __future__ import annotations

from reimp_mojo.lit import LitMOJO
from reimp_shared.data import ExpressionDataModule
from reimp_shared.foldcli import FoldCLI


class MOJOCLI(FoldCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection, so it
        # is read off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> MOJOCLI:
    return MOJOCLI(LitMOJO, ExpressionDataModule, args=args, run=run)


def main() -> None:
    build_cli()
