import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as torch_kl

from reimp_vae.model import (
    Autoencoder,
    Encoder,
    bce_loss,
    gaussian_kernel,
    kl_divergence,
    kl_weight,
    latent_head,
    mmd,
    mse_loss,
    reparameterize,
)

# ---------- losses ----------


def test_bce_loss_is_tybalts_gene_count_times_mean_bce() -> None:
    torch.manual_seed(0)
    logits, target = torch.randn(4, 50), torch.rand(4, 50)
    p = torch.sigmoid(logits)
    keras = 50 * -(target * p.log() + (1 - target) * (1 - p).log()).mean(-1)
    torch.testing.assert_close(bce_loss(logits, target), keras)


def test_bce_loss_is_minimized_at_the_target() -> None:
    target = torch.tensor([[0.2, 0.9, 0.5]])
    logits = torch.logit(target).requires_grad_()
    bce_loss(logits, target).sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_mse_loss_averages_over_genes() -> None:
    x_hat, target = torch.zeros(2, 4), torch.tensor([[1.0, 1, 1, 1], [2, 0, 0, 0]])
    torch.testing.assert_close(mse_loss(x_hat, target), torch.tensor([1.0, 1.0]))


def test_kl_divergence_matches_torch_distributions() -> None:
    torch.manual_seed(0)
    mu, logvar = torch.randn(5, 3), torch.randn(5, 3)
    expected = torch_kl(Normal(mu, torch.exp(0.5 * logvar)), Normal(0.0, 1.0)).sum(-1)
    torch.testing.assert_close(kl_divergence(mu, logvar), expected)
    assert kl_divergence(torch.zeros(1, 3), torch.zeros(1, 3)).item() == 0.0


def test_kl_warm_up() -> None:
    assert [kl_weight(e, 1.0) for e in range(3)] == [0.0, 1.0, 1.0]
    assert [kl_weight(e, 0.25) for e in range(6)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_gaussian_kernel_is_exp_of_squared_distance_over_d_squared() -> None:
    torch.manual_seed(0)
    x, y = torch.randn(3, 5), torch.randn(4, 5)
    torch.testing.assert_close(gaussian_kernel(x, y), torch.exp(-torch.cdist(x, y).pow(2) / 25))
    torch.testing.assert_close(gaussian_kernel(x, x).diagonal(), torch.ones(3))


def test_mmd_is_zero_for_one_sample_and_grows_with_a_shift() -> None:
    torch.manual_seed(0)
    x = torch.randn(300, 8)
    assert mmd(x, x).item() == pytest.approx(0.0, abs=1e-6)
    near, far = mmd(x, torch.randn(300, 8)), mmd(x, torch.randn(300, 8) + 3)
    assert 0.0 <= near.item() < far.item()


def test_mmd_pulls_a_shifted_sample_towards_the_prior() -> None:
    torch.manual_seed(0)
    z = (torch.randn(64, 4) + 2).requires_grad_()
    mmd(z, torch.randn(256, 4)).backward()
    assert (z.grad.mean(0) > 0).all()  # a descent step moves z back towards 0


# ---------- sampling ----------


def test_reparameterize_uses_exp_half_logvar_as_the_sd() -> None:
    mu = torch.ones(20_000, 2)
    logvar = torch.tensor([0.0, math.log(4.0)]).expand(20_000, 2)
    z = reparameterize(mu, logvar, torch.Generator().manual_seed(0))
    torch.testing.assert_close(z.mean(0), torch.ones(2), atol=0.05, rtol=0)
    torch.testing.assert_close(z.std(0), torch.tensor([1.0, 2.0]), atol=0.05, rtol=0)
    again = reparameterize(mu, logvar, torch.Generator().manual_seed(0))
    assert torch.equal(z, again)


# ---------- architecture ----------


def test_tybalt_heads_are_non_negative_and_linear_heads_are_not() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 30)
    mu, logvar = Encoder(30, 5, heads="bn_relu")(x)
    assert (mu >= 0).all() and (logvar >= 0).all()
    mu, logvar = Encoder(30, 5, heads="linear")(x)
    assert (mu < 0).any() and (logvar < 0).any()


def test_hidden_layer_and_mirrored_decoder() -> None:
    model = Autoencoder(40, 6, hidden_dim=8, n_classes=3)
    linear, activation, norm = model.encoder.hidden
    assert (linear.in_features, linear.out_features) == (40, 8)
    assert isinstance(activation, nn.LeakyReLU) and activation.negative_slope == 0.2
    assert isinstance(norm, nn.BatchNorm1d)
    assert model.decoder.net[0][0].in_features == 6
    assert (model.decoder.net[1].in_features, model.decoder.net[1].out_features) == (8, 40)
    out = model(torch.randn(5, 40))
    assert out["mu"].shape == out["logvar"].shape == out["z"].shape == (5, 6)
    assert out["output"].shape == (5, 40)
    assert out["logits"].shape == (5, 3)


def test_tybalt_shape_has_no_hidden_layer_and_no_classifier() -> None:
    model = Autoencoder(40, 6, heads="bn_relu")
    assert isinstance(model.encoder.hidden, nn.Identity)
    assert isinstance(model.decoder.net, nn.Linear)
    assert model.classifier is None
    assert "logits" not in model(torch.randn(3, 40))


def test_glorot_init() -> None:
    model = Autoencoder(400, 10, glorot_init=True)
    bound = math.sqrt(6 / (10 + 400))
    assert model.decoder.net.weight.abs().max() <= bound
    assert model.decoder.net.weight.abs().max() > 0.9 * bound
    assert (model.decoder.net.bias == 0).all()


def test_logvar_max_caps_the_sampling_width_but_not_the_mean() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 20) * 50
    free = Autoencoder(20, 4).eval()
    capped = Autoencoder(20, 4, logvar_max=0.0).eval()
    capped.load_state_dict(free.state_dict())
    with torch.no_grad():
        free.encoder.logvar.bias.fill_(30.0)
        capped.encoder.logvar.bias.fill_(30.0)
        a, b = free(x), capped(x)
    assert (a["logvar"] > 0).any()
    assert (b["logvar"] <= 0).all()
    torch.testing.assert_close(a["mu"], b["mu"])


def test_unknown_heads_raise() -> None:
    with pytest.raises(ValueError, match="unknown heads"):
        latent_head(4, 2, "tanh")
