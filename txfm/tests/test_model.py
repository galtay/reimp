import math

import pytest
import torch

from reimp_txfm.model import (
    Block,
    DropPath,
    MLPDecoder,
    TxFM,
    TxFMEncoder,
    poisson_loss,
    rectified_sigmoid,
    rectified_tanh,
    sample_unmasked,
)

LIBRARY_SIZE = 1e5


def _encoder(n_genes: int = 50) -> TxFMEncoder:
    encoder = TxFMEncoder(
        n_genes,
        d_model=32,
        n_layers=2,
        n_heads=4,
        dim_ff=64,
        dropout=0.0,
        drop_path_rate=0.0,
        layer_scale_init=None,
    )
    return encoder.eval()


# ---------- activation and loss ----------


def test_rectified_tanh_is_bounded_and_clamps_negatives() -> None:
    z = torch.linspace(-1e3, 1e3, 1001)
    out = rectified_tanh(z, LIBRARY_SIZE)
    assert (out >= 0).all()
    assert (out <= math.log1p(LIBRARY_SIZE)).all()
    assert (out[z <= 0] == 0).all()
    assert out[-1].item() == pytest.approx(math.log1p(LIBRARY_SIZE))


def test_tanh_and_sigmoid_forms_agree_in_value_and_gradient() -> None:
    z = torch.linspace(-200, 200, 4001, dtype=torch.float64, requires_grad=True)
    a, b = rectified_tanh(z, LIBRARY_SIZE), rectified_sigmoid(z, LIBRARY_SIZE)
    torch.testing.assert_close(a, b)
    (grad_a,) = torch.autograd.grad(a.sum(), z)
    (grad_b,) = torch.autograd.grad(b.sum(), z)
    torch.testing.assert_close(grad_a, grad_b)


def test_poisson_loss_is_minimized_at_the_target() -> None:
    x_hat = torch.linspace(0, 6, 601)
    best = x_hat[poisson_loss(x_hat, torch.tensor(3.0)).argmin()]
    assert best.item() == pytest.approx(3.0, abs=0.01)


# ---------- masking ----------


def test_sample_unmasked_draws_distinct_positions() -> None:
    idx = sample_unmasked(8, 100, 30)
    assert idx.shape == (8, 30)
    assert all(len(set(row.tolist())) == 30 for row in idx)
    assert 0 <= idx.min() and idx.max() < 100


def test_sample_unmasked_is_reproducible_with_a_generator() -> None:
    a = sample_unmasked(4, 100, 10, generator=torch.Generator().manual_seed(0))
    b = sample_unmasked(4, 100, 10, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


# ---------- architecture ----------


def test_encoder_ignores_token_order() -> None:
    """No positional encoding: a sample is a set of genes, not a sequence."""
    torch.manual_seed(0)
    encoder = _encoder()
    idx, values = torch.randint(0, 50, (3, 12)), torch.rand(3, 12) * 5
    perm = torch.randperm(12)
    with torch.no_grad():
        torch.testing.assert_close(
            encoder(idx, values), encoder(idx[:, perm], values[:, perm]), atol=1e-5, rtol=1e-5
        )


def test_encoder_depends_on_gene_identity() -> None:
    torch.manual_seed(0)
    encoder = _encoder()
    idx, values = torch.arange(12).repeat(3, 1), torch.rand(3, 12) * 5
    with torch.no_grad():
        assert not torch.allclose(encoder(idx, values), encoder(idx + 20, values), atol=1e-3)


def test_drop_path_drops_whole_samples_only_in_training() -> None:
    x = torch.ones(2000, 4)
    drop = DropPath(0.5)
    assert torch.equal(drop.eval()(x), x)
    out = drop.train()(x)
    kept = out[:, 0] > 0
    assert (out[kept] == 2.0).all()  # survivors rescaled by 1 / keep
    assert (out[~kept] == 0.0).all()
    assert 0.45 < kept.float().mean() < 0.55


def test_block_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Block(30, 4, 64, dropout=0.0, drop_path=0.0, layer_scale_init=None)


def test_decoder_depth_counts_linear_layers() -> None:
    decoder = MLPDecoder(16, 40, decoder_layers=4, dropout=0.0)
    assert len(decoder.hidden) == 3
    assert decoder(torch.randn(2, 16)).shape == (2, 40)


def test_txfm_reconstructs_every_gene_within_the_activation_range() -> None:
    model = TxFM(50, d_model=32, n_layers=1, n_heads=4, dim_ff=64).eval()
    out = model(torch.randint(0, 50, (2, 10)), torch.rand(2, 10))
    assert out.shape == (2, 50)
    assert (out >= 0).all()
    assert (out <= math.log1p(LIBRARY_SIZE)).all()


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        TxFM(10, activation="relu")
