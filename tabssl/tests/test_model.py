import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reimp_tabssl.model import (
    Decoder,
    GeneScaler,
    MLPEncoder,
    ProjectionHead,
    byol_loss,
    ema_update,
    nt_xent,
    vime_losses,
)

# ---------- scaler ----------


def test_gene_scaler_z_scores_the_samples_it_was_fit_on() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(3.0, 2.0, (50, 6)).astype(np.float32)
    scaler = GeneScaler(6).fit(values)
    z = scaler(torch.from_numpy(values))
    torch.testing.assert_close(z.mean(dim=0), torch.zeros(6), atol=1e-5, rtol=0)
    torch.testing.assert_close(z.std(dim=0, unbiased=False), torch.ones(6), atol=1e-5, rtol=0)


def test_gene_scaler_centres_a_constant_gene_without_dividing_by_zero() -> None:
    values = np.ones((4, 3), dtype=np.float32)
    values[:, 0] = [0.0, 1.0, 2.0, 3.0]
    z = GeneScaler(3).fit(values)(torch.from_numpy(values))
    assert torch.isfinite(z).all()
    assert (z[:, 1:] == 0).all()


def test_gene_scaler_refuses_to_scale_before_it_is_fit() -> None:
    with pytest.raises(RuntimeError, match="not fit"):
        GeneScaler(3)(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="expected"):
        GeneScaler(3).fit(np.zeros((4, 2)))


def test_gene_scaler_statistics_are_saved_with_the_model() -> None:
    scaler = GeneScaler(3).fit(np.arange(12, dtype=np.float32).reshape(4, 3))
    loaded = GeneScaler(3)
    loaded.load_state_dict(scaler.state_dict())
    x = torch.randn(2, 3)
    assert torch.equal(loaded(x), scaler(x))


# ---------- architecture ----------


def test_encoder_is_four_linear_bn_relu_dropout_blocks() -> None:
    encoder = MLPEncoder(100)
    assert len(encoder.blocks) == 4
    for i, block in enumerate(encoder.blocks):
        linear, bn, relu, dropout = block
        assert isinstance(linear, nn.Linear)
        assert linear.in_features == (100 if i == 0 else 256) and linear.out_features == 256
        assert isinstance(bn, nn.BatchNorm1d) and isinstance(relu, nn.ReLU)
        assert isinstance(dropout, nn.Dropout) and dropout.p == 0.2
    # The embedding is block 4's output, after ReLU: 256-d and non-negative.
    out = encoder.eval()(torch.randn(5, 100))
    assert out.shape == (5, 256) and (out >= 0).all()


def test_heads_and_decoders_have_the_papers_shapes() -> None:
    head = ProjectionHead(256, 4096, 256)
    widths = [m.out_features for m in head.net if isinstance(m, nn.Linear)]
    assert widths == [4096, 256]
    decoder = Decoder(256, 40, hidden_dim=256, n_layers=4)
    widths = [m.out_features for m in decoder.net if isinstance(m, nn.Linear)]
    assert widths == [256, 256, 256, 256, 40]
    assert decoder(torch.randn(3, 256)).shape == (3, 40)


# ---------- NT-Xent (SCARF) ----------


def _nt_xent_reference(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """Every view against the other 2N − 1, one row at a time."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    losses = []
    for i in range(2 * n):
        positive = (i + n) % (2 * n)
        sims = torch.stack([z[i] @ z[k] / tau for k in range(2 * n) if k != i])
        pos = z[i] @ z[positive] / tau
        losses.append(-(pos - torch.logsumexp(sims, dim=0)))
    return torch.stack(losses).mean()


@pytest.mark.parametrize("tau", [1.0, 0.1])
def test_nt_xent_matches_a_per_view_reference(tau) -> None:
    gen = torch.Generator().manual_seed(0)
    z1, z2 = torch.randn(6, 8, generator=gen), torch.randn(6, 8, generator=gen)
    torch.testing.assert_close(nt_xent(z1, z2, tau), _nt_xent_reference(z1, z2, tau))


def test_nt_xent_is_symmetric_and_scale_invariant() -> None:
    z1, z2 = torch.randn(5, 4), torch.randn(5, 4)
    torch.testing.assert_close(nt_xent(z1, z2), nt_xent(z2, z1))
    torch.testing.assert_close(nt_xent(z1, z2), nt_xent(3 * z1, 0.5 * z2))


def test_nt_xent_prefers_matched_pairs() -> None:
    z = torch.eye(4)  # four orthogonal samples
    matched = nt_xent(z, z, temperature=1.0)
    shuffled = nt_xent(z, z[[1, 2, 3, 0]], temperature=1.0)
    assert matched < shuffled
    # Matched: positive cos 1, six negatives at cos 0 → log(e + 6) − 1.
    assert matched.item() == pytest.approx(math.log(math.e + 6) - 1)


# ---------- VIME ----------


def test_vime_losses_are_bce_on_the_mask_and_mse_on_every_gene() -> None:
    gen = torch.Generator().manual_seed(0)
    logits, features = torch.randn(4, 6, generator=gen), torch.randn(4, 6, generator=gen)
    mask, target = torch.rand(4, 6, generator=gen) < 0.3, torch.randn(4, 6, generator=gen)
    bce, mse = vime_losses(logits, features, mask, target)
    p = torch.sigmoid(logits)
    m = mask.float()
    torch.testing.assert_close(bce, -(m * p.log() + (1 - m) * (1 - p).log()).mean())
    torch.testing.assert_close(mse, (features - target).pow(2).mean())


def test_vime_losses_vanish_for_perfect_estimates() -> None:
    mask = torch.tensor([[True, False], [False, True]])
    target = torch.randn(2, 2)
    bce, mse = vime_losses(torch.where(mask, 50.0, -50.0), target.clone(), mask, target)
    assert bce.item() < 1e-6 and mse.item() == 0.0


# ---------- BYOL ----------


def test_byol_loss_is_the_squared_distance_of_unit_vectors() -> None:
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    b = torch.tensor([[3.0, 0.0], [-1.0, 0.0], [0.0, 5.0], [1.0, 1.0]])
    torch.testing.assert_close(byol_loss(a, b), torch.tensor([0.0, 4.0, 2.0, 0.0]))


def test_byol_loss_sends_no_gradient_to_the_target() -> None:
    prediction = torch.randn(3, 4, requires_grad=True)
    target = torch.randn(3, 4, requires_grad=True)
    byol_loss(prediction, target).sum().backward()
    assert prediction.grad is not None and target.grad is None


def test_ema_update_moves_the_target_toward_the_online_network() -> None:
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    before = [p.clone() for p in target.parameters()]
    ema_update(target, online, decay=0.9)
    for t, b, o in zip(target.parameters(), before, online.parameters(), strict=True):
        torch.testing.assert_close(t, 0.9 * b + 0.1 * o)
    ema_update(target, online, decay=0.0)
    for t, o in zip(target.parameters(), online.parameters(), strict=True):
        assert torch.equal(t, o)


def test_ema_update_leaves_buffers_to_each_network() -> None:
    online, target = nn.BatchNorm1d(3), nn.BatchNorm1d(3)
    online.running_mean.fill_(5.0)
    ema_update(target, online, decay=0.5)
    assert (target.running_mean == 0).all()
