import numpy as np
import pytest
import torch

from reimp_shared.tokens import IGNORE_INDEX, BinTokenizer, mask_tokens


def test_tokenizing_needs_a_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        BinTokenizer().tokens(np.ones((1, 3)))


def test_fit_rejects_all_zero_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        BinTokenizer().fit(np.zeros((2, 3)))


def test_zero_is_bin_zero_and_the_maximum_the_top_bin() -> None:
    values = np.array([[0.0, 1e-6, 2.0, 4.0]])
    tokens = BinTokenizer(n_bins=5).fit(values).tokens(values)
    assert tokens.tolist() == [[0, 1, 2, 4]]
    assert tokens.dtype == np.int64


def test_bins_are_equal_width_over_the_training_maximum() -> None:
    tokenizer = BinTokenizer(n_bins=65).fit(np.array([[6.4]]))
    values = np.array([[0.05, 0.1, 0.15, 3.2, 6.35]])
    assert tokenizer.tokens(values).tolist() == [[1, 1, 2, 32, 64]]


def test_values_above_the_training_maximum_land_in_the_top_bin() -> None:
    tokenizer = BinTokenizer(n_bins=64).fit(np.array([[1.0, 2.0]]))
    assert tokenizer.tokens(np.array([[2.0, 50.0]])).tolist() == [[63, 63]]


def test_numpy_and_torch_give_the_same_tokens() -> None:
    values = np.random.default_rng(0).gamma(0.5, 2.0, size=(8, 30)).astype(np.float32)
    values[values < 0.2] = 0.0
    tokenizer = BinTokenizer().fit(values)
    from_torch = tokenizer.tokens(torch.from_numpy(values))
    assert from_torch.dtype == torch.long
    np.testing.assert_array_equal(from_torch.numpy(), tokenizer.tokens(values))


def test_mask_tokens_selects_about_mask_prob_and_splits_80_10_10() -> None:
    tokens = torch.randint(64, (200, 500), generator=torch.Generator().manual_seed(0))
    mask_id = 64
    inputs, labels = mask_tokens(tokens, mask_id, 64, generator=torch.Generator().manual_seed(1))
    selected = labels != IGNORE_INDEX
    assert selected.float().mean().item() == pytest.approx(0.15, abs=0.01)
    # Labels hold the originals where selected; unselected inputs are untouched.
    assert torch.equal(labels[selected], tokens[selected])
    assert torch.equal(inputs[~selected], tokens[~selected])
    masked = (inputs == mask_id) & selected
    assert (masked.sum() / selected.sum()).item() == pytest.approx(0.8, abs=0.02)
    # Random replacements are valid bins; kept plus lucky random draws are ~10%.
    assert inputs[selected & ~masked].max().item() < 64
    kept = (inputs == tokens) & selected
    assert (kept.sum() / selected.sum()).item() == pytest.approx(0.1 + 0.1 / 64, abs=0.02)


def test_mask_tokens_is_reproducible_with_a_generator() -> None:
    tokens = torch.randint(64, (4, 100))
    a = mask_tokens(tokens, 64, 64, generator=torch.Generator().manual_seed(3))
    b = mask_tokens(tokens, 64, 64, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
