import numpy as np

from reimp_shared.eval.baselines import hvg_embeddings


def test_hvg_keeps_the_most_variable_training_genes_standardized_on_training_rows() -> None:
    rng = np.random.default_rng(0)
    scale = np.array([1.0, 5.0, 0.1, 3.0, 0.0, 2.0])  # gene 4 is constant
    values = (rng.normal(size=(200, 6)) * scale).astype(np.float32)
    train = np.arange(150)
    values[150:, 2] *= 100  # varies a lot, but only outside training
    out = hvg_embeddings(values, train, n_genes=3)
    # Genes 1, 3 and 5, in dataset order.
    np.testing.assert_allclose(out[train].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out[train].std(axis=0), 1.0, atol=1e-4)
    expected = (values[:, [1, 3, 5]] - values[train][:, [1, 3, 5]].mean(axis=0)) / values[train][
        :, [1, 3, 5]
    ].std(axis=0)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_hvg_never_picks_a_constant_gene() -> None:
    values = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    out = hvg_embeddings(values, np.arange(3), n_genes=5)
    assert out.shape == (3, 1)
    assert np.isfinite(out).all()
