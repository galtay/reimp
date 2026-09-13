import numpy as np
import pytest

from reimp_shared.data import load_expression
from reimp_shared.eval import pca_embeddings
from reimp_shared.splits import N_FOLDS
from reimp_shared.testing import use_fake_dataset, write_fake_dataset


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """The miniature dataset, with `hub` reading from it."""
    root = write_fake_dataset(tmp_path / "dataset")
    use_fake_dataset(monkeypatch, root)
    return root


@pytest.fixture
def cv_pca(fake_dataset):
    """Long-form `(sample_index, embeddings, fold)`: a PCA per fold, fit on its train split."""
    sample_index, embeddings, fold = [], [], []
    for k in range(N_FOLDS):
        data = load_expression(transform="lognorm", fold=k)
        sample_index.append(data.samples["sample_index"].to_numpy())
        embeddings.append(pca_embeddings(data.values, data.rows("train"), n_components=8))
        fold.append(np.full(len(data.samples), k))
    return np.concatenate(sample_index), np.concatenate(embeddings), np.concatenate(fold)
