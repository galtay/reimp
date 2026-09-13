from pathlib import Path

import numpy as np
import pytest
import yaml

from reimp_plier.cli import main
from reimp_plier.embed import embed
from reimp_plier.embed import main as embed_main
from reimp_plier.pipeline import CONFIG_FILE, FoldModel
from reimp_shared import hub
from reimp_shared.eval import read_embeddings

CONFIGS = Path(__file__).parents[1] / "configs"
# The miniature dataset has 16 protein-coding genes and tiny gene sets.
SMALL = ["--model.k=3", "--model.min_genes=3", "--model.max_iter=25"]


def _fit(tmp_path: Path, *args: str, config: str = "debug.yaml") -> None:
    main(["fit", "--config", str(CONFIGS / config), f"--out_dir={tmp_path / 'runs'}", *args])


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_fits_and_embeds(config, fake_dataset, fake_prior, tmp_path) -> None:
    _fit(tmp_path, f"--prior={fake_prior}", *SMALL, config=config)
    model_dir = tmp_path / "runs" / "fold0"
    recorded = yaml.safe_load((model_dir / CONFIG_FILE).read_text())
    assert recorded["model"]["k"] == 3
    assert recorded["prior"] == str(fake_prior)

    sample_index, embeddings, folds = read_embeddings(embed(model_dir, tmp_path / "fold0.parquet"))
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 3)
    assert np.isfinite(embeddings).all()
    assert set(folds.tolist()) == {0}


def test_fold_flag_picks_the_fold(fake_dataset, fake_prior, tmp_path) -> None:
    _fit(tmp_path, f"--prior={fake_prior}", "--data.fold=2", *SMALL)
    out = tmp_path / "fold2.parquet"
    embed_main(["--model", str(tmp_path / "runs" / "fold2"), "--out", str(out)])
    _, _, folds = read_embeddings(out)
    # A fold's model records its fold in every row, for the probes to pool.
    assert set(folds.tolist()) == {2}


def test_no_prior_ablation(fake_dataset, tmp_path) -> None:
    _fit(tmp_path, "--prior=null", *SMALL)
    fitted = FoldModel.load(tmp_path / "runs" / "fold0")
    assert fitted.model.u_.shape == (0, 3)
    assert fitted.model.l3_ is None
    assert fitted.model.annotations_.empty
    _, embeddings, _ = read_embeddings(embed(tmp_path / "runs" / "fold0", tmp_path / "e.parquet"))
    assert embeddings.shape[1] == 3
