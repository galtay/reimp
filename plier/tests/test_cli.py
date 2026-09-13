from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from reimp_plier.cli import main
from reimp_plier.embed import embed
from reimp_plier.embed import main as embed_main
from reimp_plier.pipeline import CONFIG_FILE, FoldModel
from reimp_shared import hub
from reimp_shared.data import load_expression
from reimp_shared.eval import read_embeddings
from reimp_shared.genesets import MSIGDB
from reimp_shared.testing import assert_embedding_ignores_held_out

CONFIGS = Path(__file__).parents[1] / "configs"
# The miniature dataset has 16 protein-coding genes and tiny gene sets. With
# tol = 0 the fit runs past iteration 20, where the prior enters; it would
# otherwise converge first.
SMALL = ["--model.k=3", "--model.min_genes=3", "--model.max_iter=25", "--model.tol=0"]


def _fit(tmp_path: Path, *args: str, config: str = "debug.yaml") -> None:
    main(["fit", "--config", str(CONFIGS / config), f"--out_dir={tmp_path / 'runs'}", *args])


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_names_msigdb_collections_reimp_knows(config) -> None:
    prior = yaml.safe_load((CONFIGS / config).read_text())["prior"]
    assert prior and set(prior) <= set(MSIGDB)


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_fits_and_embeds(config, fake_dataset, fake_prior, tmp_path) -> None:
    _fit(tmp_path, f"--prior=[{fake_prior}]", *SMALL, config=config)
    model_dir = tmp_path / "runs" / "fold0"
    recorded = yaml.safe_load((model_dir / CONFIG_FILE).read_text())
    assert recorded["model"]["k"] == 3
    assert recorded["prior"] == [str(fake_prior)]
    fitted = FoldModel.load(model_dir)
    assert fitted.model.u_.any() and fitted.model.l3_ is not None
    assert not fitted.model.annotations_.empty

    sample_index, embeddings, folds = read_embeddings(embed(model_dir, tmp_path / "fold0.parquet"))
    assert sample_index.tolist() == hub.load_samples()["sample_index"].tolist()
    assert embeddings.shape == (len(sample_index), 3)
    assert np.isfinite(embeddings).all()
    assert set(folds.tolist()) == {0}


def test_fold_flag_picks_the_fold(fake_dataset, fake_prior, tmp_path) -> None:
    _fit(tmp_path, f"--prior=[{fake_prior}]", "--data.fold=2", *SMALL)
    model_dir = tmp_path / "runs" / "fold2"
    out = tmp_path / "fold2.parquet"
    embed_main(["--model", str(model_dir), "--out", str(out)])
    _, _, folds = read_embeddings(out)
    # A fold's model records its fold in every row, for the probes to pool ...
    assert set(folds.tolist()) == {2}

    # ... and is fit on that fold's training rows, not another fold's.
    fitted = FoldModel.load(model_dir)
    data_config = yaml.safe_load((model_dir / CONFIG_FILE).read_text())["data"]

    def train_mean(fold: int) -> np.ndarray:
        data = load_expression(**{**data_config, "fold": fold})
        columns = pd.Index(data.genes["gene_id"]).get_indexer(fitted.gene_ids)
        return data.values[data.rows("train")][:, columns].mean(axis=0, dtype=np.float64)

    np.testing.assert_allclose(fitted.scaler.mean, train_mean(2))
    assert not np.allclose(fitted.scaler.mean, train_mean(0))


def test_embedding_ignores_held_out_rows(fake_dataset, fake_prior, monkeypatch, tmp_path) -> None:
    _fit(tmp_path, f"--prior=[{fake_prior}]", "--data.fold=1", *SMALL)
    model_dir = tmp_path / "runs" / "fold1"
    assert_embedding_ignores_held_out(lambda out: embed(model_dir, out), 1, monkeypatch, tmp_path)


def test_no_prior_ablation(fake_dataset, tmp_path) -> None:
    _fit(tmp_path, "--prior=null", *SMALL)
    fitted = FoldModel.load(tmp_path / "runs" / "fold0")
    assert fitted.model.u_.shape == (0, 3)
    assert fitted.model.l3_ is None
    assert fitted.model.annotations_.empty
    _, embeddings, _ = read_embeddings(embed(tmp_path / "runs" / "fold0", tmp_path / "e.parquet"))
    assert embeddings.shape[1] == 3
