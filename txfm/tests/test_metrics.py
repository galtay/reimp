import torch

from reimp_txfm.metrics import holdout_metrics
from reimp_txfm.model import poisson_loss


def _batch(b: int = 4, g: int = 30, k: int = 6, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    target = torch.rand(b, g, generator=gen) * 8
    x_hat = torch.rand(b, g, generator=gen) * 8
    idx = torch.stack([torch.randperm(g, generator=gen)[:k] for _ in range(b)])
    visible = torch.zeros(b, g, dtype=torch.bool).scatter_(1, idx, True)
    return x_hat, target, idx, visible


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_losses_split_by_what_the_encoder_saw() -> None:
    x_hat, target, idx, visible = _batch()
    per_gene = poisson_loss(x_hat, target)
    out = holdout_metrics(x_hat, target, idx)
    _close(out["loss_visible"], per_gene[visible].mean())
    _close(out["loss_holdout"], per_gene[~visible].mean())


def test_perfect_reconstruction_scores_one() -> None:
    _, target, idx, _ = _batch()
    out = holdout_metrics(target.clone(), target, idx)
    _close(out["pearson_holdout"], torch.tensor(1.0))
    _close(out["r2_holdout"], torch.tensor(1.0))


def test_matches_a_per_sample_reference() -> None:
    x_hat, target, idx, visible = _batch(seed=1)
    pearsons, r2s = [], []
    for b in range(len(target)):
        a, t = x_hat[b][~visible[b]], target[b][~visible[b]]
        pearsons.append(torch.corrcoef(torch.stack([a, t]))[0, 1])
        r2s.append(1 - (a - t).pow(2).sum() / (t - t.mean()).pow(2).sum())
    out = holdout_metrics(x_hat, target, idx)
    _close(out["pearson_holdout"], torch.stack(pearsons).mean())
    _close(out["r2_holdout"], torch.stack(r2s).mean())


def test_samples_with_constant_holdout_are_left_out_of_the_average() -> None:
    x_hat, target, idx, _ = _batch(b=3)
    target[0], x_hat[0] = 1.0, 1.0  # correlation undefined for this sample
    out = holdout_metrics(x_hat, target, idx)
    rest = holdout_metrics(x_hat[1:], target[1:], idx[1:])
    _close(out["pearson_holdout"], rest["pearson_holdout"])
    _close(out["r2_holdout"], rest["r2_holdout"])
