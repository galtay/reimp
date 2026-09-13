import pytest
import torch

from reimp_tabssl.corrupt import (
    bernoulli_mask,
    corrupt,
    fixed_count_mask,
    marginal_draws,
    uniform_draws,
)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---------- SCARF: a fixed count per sample ----------


def test_fixed_count_mask_takes_exactly_rate_of_every_row() -> None:
    mask = fixed_count_mask(16, 101, 0.3)
    assert mask.shape == (16, 101) and mask.dtype == torch.bool
    assert (mask.sum(dim=1) == 30).all()  # ⌊0.3 · 101⌋


def test_fixed_count_mask_picks_genes_uniformly() -> None:
    mask = fixed_count_mask(20_000, 10, 0.3, generator=_gen())
    torch.testing.assert_close(mask.float().mean(dim=0), torch.full((10,), 0.3), atol=0.02, rtol=0)
    # Rows are drawn independently, not one pattern repeated.
    assert len({tuple(row.tolist()) for row in mask[:50]}) > 40


def test_fixed_count_mask_edge_rates() -> None:
    assert not fixed_count_mask(3, 10, 0.05).any()  # ⌊0.5⌋ = 0
    assert fixed_count_mask(3, 10, 1.0).all()


# ---------- VIME: independent entries ----------


def test_bernoulli_mask_has_the_rate_on_average_but_not_per_row() -> None:
    mask = bernoulli_mask(1000, 200, 0.3, generator=_gen())
    assert mask.float().mean().item() == pytest.approx(0.3, abs=0.01)
    assert mask.sum(dim=1).unique().numel() > 1


# ---------- replacements ----------


def test_uniform_draws_fill_each_genes_range() -> None:
    low = torch.tensor([-1.0, 0.0, 5.0])
    high = torch.tensor([1.0, 0.0, 9.0])
    draws = uniform_draws(low, high, 20_000, generator=_gen())
    assert draws.shape == (20_000, 3)
    assert (draws >= low).all() and (draws <= high).all()
    assert (draws[:, 1] == 0).all()  # a constant gene stays at its value
    # Uniform, not concentrated: the median sits mid-range, a quarter lies in each quarter.
    torch.testing.assert_close(draws.median(dim=0).values, (low + high) / 2, atol=0.05, rtol=0)
    quarter = ((draws[:, 2] - 5.0) < 1.0).float().mean().item()
    assert quarter == pytest.approx(0.25, abs=0.02)


def test_marginal_draws_take_each_gene_from_the_pool_independently() -> None:
    # Pool entry (r, j) = 10 · r + j: a draw's row and gene can be read off it.
    pool = (10 * torch.arange(5)[:, None] + torch.arange(4)).float()
    draws = marginal_draws(pool, 2000, generator=_gen())
    assert draws.shape == (2000, 4)
    assert torch.equal(draws % 10, torch.arange(4).float().expand(2000, 4))  # gene j from column j
    rows = (draws // 10).long()
    assert set(rows.unique().tolist()) == set(range(5))
    # Each entry picks its own row: a sample is not one pool row copied whole.
    assert (rows != rows[:, :1]).any(dim=1).float().mean() > 0.9
    counts = torch.bincount(rows.flatten(), minlength=5).float()
    torch.testing.assert_close(counts / counts.sum(), torch.full((5,), 0.2), atol=0.02, rtol=0)


def test_marginal_draws_need_a_pool() -> None:
    with pytest.raises(RuntimeError, match="pool"):
        marginal_draws(torch.empty(0, 4), 2)


def test_draws_are_reproducible_with_a_generator() -> None:
    pool = torch.randn(7, 12)
    low, high = pool.min(dim=0).values, pool.max(dim=0).values
    for draw in (
        lambda g: fixed_count_mask(4, 12, 0.3, generator=g),
        lambda g: bernoulli_mask(4, 12, 0.3, generator=g),
        lambda g: uniform_draws(low, high, 4, generator=g),
        lambda g: marginal_draws(pool, 4, generator=g),
    ):
        assert torch.equal(draw(_gen(3)), draw(_gen(3)))
        assert not torch.equal(draw(_gen(3)), draw(_gen(4)))


def test_corrupt_replaces_the_masked_entries_only() -> None:
    values = torch.zeros(3, 5)
    replacement = torch.ones(3, 5)
    mask = torch.tensor([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(corrupt(values, mask, replacement), mask.float())
