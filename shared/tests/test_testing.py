import numpy as np
import pytest

from reimp_shared.data import load_expression
from reimp_shared.eval import write_embeddings
from reimp_shared.testing import assert_embedding_ignores_held_out, scramble_held_out

FOLD = 1


def _embedder(refit: bool):
    """A toy embedding model: z-score with training statistics, or refit on every sample."""
    data = load_expression(transform="lognorm", fold=FOLD)
    train = data.values[data.rows("train")]
    mean, sd = train.mean(axis=0), train.std(axis=0) + 1e-3

    def embed(path):
        data = load_expression(transform="lognorm", fold=FOLD)
        m, s = (data.values.mean(axis=0), data.values.std(axis=0) + 1e-3) if refit else (mean, sd)
        z = (data.values - m) / s
        return write_embeddings(path, data.samples["sample_index"].to_numpy(), z[:, :4], FOLD)

    return embed


def test_scrambling_changes_held_out_rows_only(fake_dataset, monkeypatch) -> None:
    clean = load_expression(fold=FOLD)
    scramble_held_out(monkeypatch, FOLD)
    scrambled = load_expression(fold=FOLD)
    train = clean.rows("train")
    held_out = np.setdiff1d(np.arange(len(clean.values)), train)
    np.testing.assert_array_equal(scrambled.values[train], clean.values[train])
    assert (scrambled.values[held_out] != clean.values[held_out]).any(axis=1).all()


def test_an_embedding_from_training_statistics_passes(fake_dataset, monkeypatch, tmp_path) -> None:
    assert_embedding_ignores_held_out(_embedder(refit=False), FOLD, monkeypatch, tmp_path)


def test_an_embedding_refit_on_every_sample_fails(fake_dataset, monkeypatch, tmp_path) -> None:
    with pytest.raises(AssertionError):
        assert_embedding_ignores_held_out(_embedder(refit=True), FOLD, monkeypatch, tmp_path)
