import numpy as np
import pytest
import torch

from reimp_compass.triplets import NegativeSampler, augment


def _views(no_augment_prob: float, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return augment(torch.ones(4000, 50), 0.3, 0.5, no_augment_prob, generator)


def test_each_view_is_masked_jittered_or_left_alone() -> None:
    views = _views(no_augment_prob=0.1)
    unchanged = (views == 1).all(1)
    masked = ((views == 0) | (views == 1)).all(1) & ~unchanged
    jittered = ~unchanged & ~masked
    assert 0.08 < unchanged.float().mean() < 0.12
    assert 0.42 < masked.float().mean() < 0.48
    assert 0.42 < jittered.float().mean() < 0.48
    # Masked rows zero each gene with probability mask_prob ...
    assert (views[masked] == 0).float().mean().item() == pytest.approx(0.3, abs=0.01)
    # ... and jittered rows carry noise of standard deviation jitter_std.
    assert (views[jittered] - 1).std().item() == pytest.approx(0.5, abs=0.01)


def test_no_augment_prob_one_is_the_identity() -> None:
    x = torch.rand(10, 5)
    torch.testing.assert_close(augment(x, 0.5, 1.0, no_augment_prob=1.0), x)


def test_augmentations_are_reproducible_with_a_generator() -> None:
    torch.testing.assert_close(_views(0.1, seed=3), _views(0.1, seed=3))
    assert not torch.equal(_views(0.1, seed=3), _views(0.1, seed=4))


# Two samples per case; cases 0-3 in project P, 4-5 in Q, 6 in R.
CASES = np.repeat([f"c{i}" for i in range(7)], 2)
PROJECTS = np.repeat(["P"] * 4 + ["Q"] * 2 + ["R"], 2)


def test_negatives_come_from_another_patient_in_the_pool() -> None:
    pool = np.arange(0, 12)  # c6 (R) is not in the pool
    sampler = NegativeSampler(CASES, PROJECTS, pool)
    anchors = torch.from_numpy(np.repeat(pool, 200))
    negatives = sampler.draw(anchors, torch.Generator().manual_seed(0))
    assert np.isin(negatives.numpy(), pool).all()
    assert (CASES[negatives.numpy()] != CASES[anchors.numpy()]).all()
    # Any project: an anchor in P meets negatives from Q too.
    assert (PROJECTS[negatives.numpy()] != PROJECTS[anchors.numpy()]).any()


def test_same_project_negatives_stay_in_the_project() -> None:
    pool = np.arange(14)
    sampler = NegativeSampler(CASES, PROJECTS, pool, negatives="same_project")
    anchors = torch.from_numpy(np.repeat(pool[:12], 100))
    negatives = sampler.draw(anchors).numpy()
    assert (PROJECTS[negatives] == PROJECTS[anchors.numpy()]).all()
    assert (CASES[negatives] != CASES[anchors.numpy()]).all()
    # R has a single patient, so its negatives come from the whole pool.
    lone = sampler.draw(torch.full((200,), 12)).numpy()
    assert (CASES[lone] != "c6").all()
    assert set(PROJECTS[lone]) == {"P", "Q"}


def test_negatives_are_reproducible_with_a_generator() -> None:
    sampler = NegativeSampler(CASES, PROJECTS, np.arange(14))
    anchors = torch.arange(14).repeat(10)
    first = sampler.draw(anchors, torch.Generator().manual_seed(1))
    assert torch.equal(first, sampler.draw(anchors, torch.Generator().manual_seed(1)))


def test_a_pool_of_one_patient_has_no_negatives() -> None:
    with pytest.raises(ValueError, match="two patients"):
        NegativeSampler(CASES, PROJECTS, np.array([0, 1]))
    with pytest.raises(ValueError, match="unknown negatives"):
        NegativeSampler(CASES, PROJECTS, np.arange(14), negatives="nearest")
