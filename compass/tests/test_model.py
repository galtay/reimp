import numpy as np
import pytest
import torch

from reimp_compass.hierarchy import load_hierarchy
from reimp_compass.model import (
    Compass,
    ConceptProjector,
    EncoderLayer,
    GeneTokenizer,
    MinMaxScaler,
    triplet_loss,
)

FAKE_PROTEIN_CODING = [f"GENE{i}" for i in range(0, 46, 3)]


def _projector(hierarchy_path, d_model: int = 8) -> ConceptProjector:
    hierarchy = load_hierarchy(hierarchy_path)
    projector = ConceptProjector(hierarchy, d_model)
    projector.bind(hierarchy.member_positions(FAKE_PROTEIN_CODING))
    return projector


# ---------- scaling and tokens ----------


def test_min_max_scaler_maps_the_fit_samples_to_the_unit_interval() -> None:
    rng = np.random.default_rng(0)
    values = rng.gamma(2.0, 2.0, (50, 6)).astype(np.float32)
    values[:, 3] = 1.5  # constant: scale 1, as scikit-learn does
    scaler = MinMaxScaler(6).fit(values)
    scaled = scaler(torch.from_numpy(values))
    fit = np.delete(np.arange(6), 3)
    torch.testing.assert_close(scaled[:, fit].min(0).values, torch.zeros(5))
    torch.testing.assert_close(scaled[:, fit].max(0).values, torch.ones(5))
    assert (scaled[:, 3] == 0).all()
    # Unseen values are not clipped.
    assert scaler(torch.full((1, 6), 1e3)).max() > 1


def test_min_max_scaler_must_be_fit_first() -> None:
    with pytest.raises(RuntimeError, match="fit the scaler"):
        MinMaxScaler(3)(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="expected"):
        MinMaxScaler(3).fit(np.zeros((2, 4)))


def test_gene_tokens_are_relu_of_a_per_gene_affine_map() -> None:
    torch.manual_seed(0)
    tokenizer = GeneTokenizer(5, 4)
    x = torch.rand(3, 5)
    expected = torch.relu(x[..., None] * tokenizer.weight + tokenizer.bias)
    torch.testing.assert_close(tokenizer(x), expected)
    # A gene's token depends on that gene's value alone.
    y = x.clone()
    y[:, 2] += 1.0
    changed = (tokenizer(y) != tokenizer(x)).any(dim=(0, 2))
    assert changed.tolist() == [False, False, True, False, False]


# ---------- encoder ----------


def test_chunked_attention_matches_full_attention_with_gradients() -> None:
    torch.manual_seed(0)
    full = EncoderLayer(8, n_heads=2, head_dim=4, dim_ff=16, dropout=0.0)
    chunked = EncoderLayer(8, n_heads=2, head_dim=4, dim_ff=16, dropout=0.0, chunk_size=3)
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(2, 10, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    out_full, out_chunked = full(x), chunked(y)
    torch.testing.assert_close(out_chunked, out_full)
    out_full.square().sum().backward()
    out_chunked.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad)
    torch.testing.assert_close(chunked.qkv.weight.grad, full.qkv.weight.grad)
    with torch.no_grad():
        torch.testing.assert_close(chunked(x), full(x))


def test_heads_can_be_wider_than_d_model_over_heads() -> None:
    """COMPASS's layer: 2 heads of 32 on d = 32, an inner width of 64."""
    layer = EncoderLayer(32, n_heads=2, head_dim=32, dim_ff=64, dropout=0.2)
    assert layer.qkv.out_features == 3 * 64
    assert layer(torch.randn(2, 7, 32)).shape == (2, 7, 32)


def test_the_released_layer_size() -> None:
    """paper.md: the transformer layer is ~13k parameters."""
    layer = EncoderLayer(32, n_heads=2, head_dim=32, dim_ff=64, dropout=0.2)
    assert 12_000 < sum(p.numel() for p in layer.parameters()) < 14_000


# ---------- concept projector ----------


def test_set_weights_are_a_softmax_over_each_sets_present_genes(hierarchy_path) -> None:
    projector = _projector(hierarchy_path)
    weights = projector.set_weights().detach()
    member_set = projector.set_member.float().argmax(0)
    present = projector.member_present
    for s in range(5):
        own = (member_set == s) & present
        assert weights[s, own].sum().item() == pytest.approx(1.0)
        assert (weights[s, ~own] == 0).all()
        # One learned logit per membership, independent of the input.
        expected = torch.softmax(projector.gene_logits.detach()[own], 0)
        torch.testing.assert_close(weights[s, own], expected)
    assert (weights[5] == 0).all()  # B_ghost: no gene in the data
    assert not present[10] and not present[14]  # GENE1, NOTAGENE


def test_set_scores_are_one_shared_linear_of_weighted_gene_embeddings(hierarchy_path) -> None:
    torch.manual_seed(0)
    projector = _projector(hierarchy_path)
    genes = torch.randn(2, 16, 8)
    set_scores, _ = projector(genes)
    weights = projector.set_weights()
    # B_two: GENE30 twice (columns 10, 10) and GENE33 (column 11).
    w = weights[4, [11, 12, 13]]
    pooled = w[0] * genes[:, 10] + w[1] * genes[:, 11] + w[2] * genes[:, 10]
    torch.testing.assert_close(set_scores[:, 4], projector.set_scorer(pooled).squeeze(-1))
    # A set with no gene in the data scores the bias alone.
    torch.testing.assert_close(set_scores[:, 5], projector.set_scorer.bias.expand(2))


def test_concepts_are_convex_combinations_of_their_own_sets(hierarchy_path) -> None:
    projector = _projector(hierarchy_path)
    weights = projector.concept_weights().detach()
    torch.testing.assert_close(weights.sum(1), torch.ones(3))
    own = torch.tensor([[0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0]]).bool()
    assert (weights[~own] == 0).all()
    assert (weights[own] > 0).all()
    # Equal set scores give that score to every concept.
    with torch.no_grad():
        projector.set_scorer.weight.zero_()
        projector.set_scorer.bias.fill_(0.7)
    _, concepts = projector(torch.randn(3, 16, 8))
    torch.testing.assert_close(concepts, torch.full((3, 3), 0.7))


def test_an_unbound_projector_refuses_to_project(hierarchy_path) -> None:
    projector = ConceptProjector(load_hierarchy(hierarchy_path), 8)
    with pytest.raises(RuntimeError, match="bind"):
        projector(torch.randn(1, 16, 8))
    with pytest.raises(ValueError, match="positions"):
        projector.bind(np.zeros(3, dtype=int))


def test_compass_scores_every_set_and_concept(hierarchy_path) -> None:
    hierarchy = load_hierarchy(hierarchy_path)
    model = Compass(16, hierarchy, d_model=8, head_dim=4, dim_ff=16).eval()
    model.projector.bind(hierarchy.member_positions(FAKE_PROTEIN_CODING))
    sets, concepts = model(torch.rand(4, 16))
    assert sets.shape == (4, 6)
    assert concepts.shape == (4, 3)


# ---------- loss ----------


def test_triplet_loss_uses_cosine_distance_and_a_margin() -> None:
    e1, e2 = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    assert triplet_loss(e1, e1, -e1).item() == 0.0  # d_pos 0, d_neg 2
    assert triplet_loss(e1, e1, e1).item() == pytest.approx(1.0)  # the margin
    assert triplet_loss(e1, e2, e1).item() == pytest.approx(2.0)  # d_pos 1, d_neg 0
    assert triplet_loss(e1, 3 * e1, e2, margin=0.5).item() == 0.0  # scale-free
    assert triplet_loss(torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)).shape == (5,)
