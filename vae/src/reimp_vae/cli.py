"""`vae` — LightningCLI over `LitVAE` and `VAEDataModule`.

vae fit --config vae/configs/debug.yaml
vae fit --config vae/configs/tybalt.yaml --data.fold 0
vae fit --config vae/configs/mmdae_organ.yaml --data.fold 0

Fold k runs in `<trainer.default_root_dir>/fold<k>/` (`FoldCLI`), e.g.
`runs/tybalt/fold0/checkpoints/best.ckpt`.
"""

from __future__ import annotations

from reimp_shared.foldcli import FoldCLI
from reimp_vae.data import VAEDataModule
from reimp_vae.lit import LitVAE


class VAECLI(FoldCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # The gene count follows from the DataModule's gene selection (and
        # `top_genes`), the class count from its `supervision`: both are read
        # off the instantiated DataModule rather than set by hand.
        parser.link_arguments("data.n_genes", "model.n_genes", apply_on="instantiate")
        parser.link_arguments("data.n_classes", "model.n_classes", apply_on="instantiate")


def build_cli(args: list[str] | None = None, run: bool = True) -> VAECLI:
    return VAECLI(LitVAE, VAEDataModule, args=args, run=run)


def main() -> None:
    build_cli()
