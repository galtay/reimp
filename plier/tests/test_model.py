import math

import numpy as np
import pytest

from reimp_plier.model import (
    LAMBDA3_PATH,
    PLIER,
    GeneScaler,
    _ridge_z,
    b_step,
    benjamini_hochberg,
    candidate_sets,
    cross_validate,
    holdout_prior,
    initialize,
    nonnegative_enet,
    num_pc,
    pinv_ridge,
    ridge_inverse_prior,
    u_step,
    wilcoxon_auc,
    z_step,
)


def _planted(seed: int = 0, n_genes: int = 150, n_samples: int = 90, n_sets: int = 10):
    """Z-scored Y (genes x samples) driven by 4 of the prior's gene sets, and the prior."""
    rng = np.random.default_rng(seed)
    prior = np.zeros((n_genes, n_sets))
    for j in range(n_sets):
        prior[rng.choice(n_genes, 18, replace=False), j] = 1.0
    loadings = prior[:, :4] * rng.uniform(1.0, 2.0, (n_genes, 4))
    x = loadings @ rng.normal(size=(4, n_samples)) + 0.3 * rng.normal(size=(n_genes, n_samples))
    return GeneScaler.fit(x.T).transform(x.T), prior


def _fit(max_iter: int = 40, **kwargs) -> tuple[PLIER, np.ndarray, np.ndarray]:
    y, prior = _planted()
    model = PLIER(k=kwargs.pop("k", 6), max_iter=max_iter, **kwargs).fit(y, prior)
    return model, y, prior


def test_gene_scaler_uses_n_minus_1_and_returns_genes_by_samples() -> None:
    x = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 9.0]])
    scaler = GeneScaler.fit(x)
    np.testing.assert_allclose(scaler.sd, [2.0, 0.0, np.std([2, 4, 9], ddof=1)])
    assert scaler.varying().tolist() == [0, 2]
    y = scaler.subset(scaler.varying()).transform(x[:, [0, 2]])
    assert y.shape == (2, 3)
    np.testing.assert_allclose(y.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(y.std(axis=1, ddof=1), 1)


def test_z_step_is_the_ridge_minimizer_clipped_at_zero() -> None:
    rng = np.random.default_rng(0)
    y, b, cu, l1 = rng.normal(size=(30, 20)), rng.normal(size=(4, 20)), rng.random((30, 4)), 2.5
    z = _ridge_z(y, b, l1, cu)
    # d/dZ of ‖Y − ZB‖² + λ1‖Z − CU‖² vanishes at the unclipped solution.
    np.testing.assert_allclose(-(y - z @ b) @ b.T + l1 * (z - cu), 0, atol=1e-10)
    assert (z < 0).any()
    np.testing.assert_array_equal(z_step(y, b, l1, cu), np.maximum(z, 0))
    # Without a prior the penalty is λ1‖Z‖².
    z0 = _ridge_z(y, b, l1, None)
    np.testing.assert_allclose(-(y - z0 @ b) @ b.T + l1 * z0, 0, atol=1e-10)


def test_b_step_minimizes_the_objective_in_b() -> None:
    rng = np.random.default_rng(1)
    y, z, l2 = rng.normal(size=(30, 20)), rng.random((30, 4)), 3.0
    b = b_step(y, z, l2)
    np.testing.assert_allclose(-z.T @ (y - z @ b) + l2 * b, 0, atol=1e-10)
    np.testing.assert_allclose(b, np.linalg.inv(z.T @ z + l2 * np.eye(4)) @ z.T @ y)


def test_initialize_from_the_svd() -> None:
    y, _ = _planted()
    _, d, vt = np.linalg.svd(y, full_matrices=False)
    z, b = initialize(y, d, vt, 5, d[4] / 2)
    assert (z >= 0).all()
    assert (z > 0).any(axis=0).all()  # no column clipped away entirely
    np.testing.assert_allclose(np.abs(b), np.abs(d[:5, None] * vt[:5]))


def test_pinv_ridge_is_a_ridge_pseudo_inverse() -> None:
    m = np.random.default_rng(2).normal(size=(6, 4))
    expected = np.linalg.inv(m.T @ m + 25 * np.eye(4)) @ m.T
    np.testing.assert_allclose(pinv_ridge(m, 5.0), expected, atol=1e-12)


def test_candidate_sets_complete_pools_every_lvs_top_sets() -> None:
    rng = np.random.default_rng(3)
    prior = (rng.random((80, 30)) < 0.2).astype(float)
    z = rng.random((80, 5))
    chat = ridge_inverse_prior(prior)
    fast = candidate_sets(chat, z, max_path=4, selection="fast")
    assert all(len(sets) == 4 for sets in fast)
    complete = candidate_sets(chat, z, max_path=4)
    np.testing.assert_array_equal(complete[0], np.unique(np.concatenate(fast)))
    assert all(np.array_equal(sets, complete[0]) for sets in complete)


# glmnet 5.0 in R: glmnet(x, y, alpha = 0.9, lower.limits = 0, intercept = TRUE,
# standardize = FALSE, lambda = c(0.5, 0.2, 0.05, 0.01), thresh = 1e-14)$beta, rows 1 and 2
# (the other three rows are 0). Unscaled, scikit-learn's ElasticNet is 0.07 away.
GLMNET_BETA = [
    [0.0, 0.095233748235, 0.493505901511, 0.601903708660],
    [0.0, 1.157992440908, 1.622747981213, 1.750303335832],
]


def test_nonnegative_enet_is_on_glmnets_lambda_scale() -> None:
    i = np.arange(1, 41)
    x = np.stack([(((i * (j + 2)) % 5 == 0) | ((i + j) % 7 == 0)) for j in range(5)], 1)
    x = x.astype(float)
    y = 3 * np.sin(i) + 2 * x[:, 1] + 1.5 * x[:, 3]
    centred = np.asfortranarray(x - x.mean(axis=0))
    lambdas = np.array([0.5, 0.2, 0.05, 0.01])
    beta = nonnegative_enet(centred, centred.T @ centred, y, lambdas, tol=1e-14)
    np.testing.assert_allclose(beta[[0, 1]], GLMNET_BETA, atol=1e-7)
    np.testing.assert_array_equal(beta[[2, 3, 4]], 0)
    # Any λ order gives the same fits, column for column.
    shuffled = nonnegative_enet(centred, centred.T @ centred, y, lambdas[::-1], tol=1e-14)
    np.testing.assert_allclose(shuffled[:, ::-1], beta, atol=1e-10)


def test_nonnegative_enet_of_a_constant_is_zero() -> None:
    x = np.asfortranarray(np.eye(4) - 0.25)
    assert not nonnegative_enet(x, x.T @ x, np.ones(4), LAMBDA3_PATH).any()


def test_u_step_is_nonnegative_on_candidate_sets_only() -> None:
    y, prior = _planted()
    rng = np.random.default_rng(4)
    z = np.maximum(prior[:, :6] @ rng.random((6, 6)) + 0.1 * rng.normal(size=(150, 6)), 0)
    chat = ridge_inverse_prior(prior)
    u, l3 = u_step(z, prior, chat, max_path=3)
    assert (u >= 0).all() and (u > 0).any()
    allowed = np.zeros(prior.shape[1], dtype=bool)
    allowed[candidate_sets(chat, z, max_path=3)[0]] = True
    assert not u[~allowed].any()

    # λ3 comes off glmnet's path, where the share of LVs with a set is nearest 0.7.
    assert l3 in LAMBDA3_PATH
    shares = [
        (u_step(z, prior, chat, lam, max_path=3)[0] > 0).any(axis=0).mean() for lam in LAMBDA3_PATH
    ]
    best = np.min(np.abs(0.7 - np.array(shares)))
    assert abs(0.7 - (u > 0).any(axis=0).mean()) == pytest.approx(best)
    # A given λ3 is used as is.
    fixed, same = u_step(z, prior, chat, l3, max_path=3)
    assert same == l3
    np.testing.assert_allclose(fixed, u, atol=1e-6)


def test_holdout_prior_removes_a_fifth_of_each_set() -> None:
    prior = np.zeros((50, 3))
    prior[:10, 0] = prior[:23, 1] = prior[:4, 2] = 1
    c = holdout_prior(prior, np.random.default_rng(0))
    assert ((c > 0) <= (prior > 0)).all()
    assert c.sum(axis=0).tolist() == [8, 19, 4]


def test_wilcoxon_auc_matches_r() -> None:
    # R: wilcox.test(pos, neg, alternative = "greater"): exact, W = 11, p = 0.0571...
    labels = np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool)
    values = np.array([3.1, 4.2, 5.5, 1.0, 2.2, 3.0, 4.0])
    auc, p = wilcoxon_auc(labels, values)
    assert auc == pytest.approx(11 / 12)
    assert p == pytest.approx(0.057142857142857141, rel=1e-12)
    # With ties, the normal approximation: W = 300.5, p = 0.7395...
    i = np.arange(60)
    auc, p = wilcoxon_auc(i % 4 == 0, ((i * 7) % 13).astype(float))
    assert auc == pytest.approx(300.5 / (15 * 45))
    assert p == pytest.approx(0.7395829961065804, rel=1e-10)
    assert wilcoxon_auc(np.zeros(3, dtype=bool), np.arange(3.0))[0] == 0.5


def test_benjamini_hochberg_matches_r_and_keeps_nan() -> None:
    adjusted = benjamini_hochberg(np.array([0.01, np.nan, 0.04, 0.03, 0.5]))
    np.testing.assert_allclose(
        adjusted, [0.04, np.nan, 0.05333333333333333, 0.05333333333333333, 0.5]
    )


def test_cross_validate_scores_held_out_members_against_outside_genes() -> None:
    prior = np.zeros((40, 2))
    prior[:10, 0] = prior[10:20, 1] = 1
    prior_cv = prior.copy()
    prior_cv[:2, 0] = 0  # genes 0 and 1 held out of set 0
    z = np.zeros((40, 2))
    z[:10, 0] = 5.0  # LV 0 loads set 0's genes, held-out ones included
    z[20:, 0] = np.linspace(0, 1, 20)
    u = np.array([[0.7, 0.0], [0.0, 0.0]])
    table = cross_validate(z, u, prior, prior_cv, ["A", "B"])
    assert table[["gene_set", "lv"]].values.tolist() == [["A", 0]]
    assert table["auc"].iloc[0] == 1.0  # both held-out genes above every outside gene
    assert table["p_value"].iloc[0] < 0.05


def test_lambdas_and_k_follow_the_singular_values() -> None:
    model, y, _ = _fit(k=None)
    d = model.singular_values_
    assert model.k_ == min(round(2 * num_pc(d)), math.floor(0.9 * y.shape[1]))
    assert model.l2_ == d[model.k_ - 1]
    assert model.l1_ == model.l2_ / 2


def test_prior_enters_at_iteration_20() -> None:
    before, _, _ = _fit(max_iter=19, tol=0)
    assert before.n_iter_ == 19
    assert not before.u_.any() and before.l3_ is None
    after, _, _ = _fit(max_iter=20, tol=0)
    assert after.u_.any() and after.l3_ in LAMBDA3_PATH


def test_fit_gives_nonnegative_z_and_u_and_annotations() -> None:
    # tol = 0: this easy problem would otherwise converge before the prior enters.
    model, y, prior = _fit(max_iter=60, tol=0)
    assert (model.z_ >= 0).all() and (model.u_ >= 0).all()
    assert model.z_.shape == (150, 6) and model.b_.shape == (6, 90) and model.u_.shape == (10, 6)
    # Held-out genes of the planted sets are recovered by the LVs that use them.
    table = model.annotations_
    assert set(table["gene_set"]) <= {f"set{j}" for j in range(10)}
    assert (table["auc"] > 0.7).any()
    assert len(model.annotated()) > 0
    # The prior's C is the prior with a fifth of each set held out.
    assert (model.prior_cv_ <= model.prior_).all()
    # A fitted model embeds its training samples as its own B.
    np.testing.assert_allclose(model.project(y), model.b_)


def test_no_prior_is_the_same_solver_with_u_zero() -> None:
    y, prior = _planted()
    with_prior = PLIER(max_iter=30).fit(y, prior)
    without = PLIER(max_iter=30).fit(y)
    assert without.u_.shape == (0, without.k_)
    assert without.annotations_.empty and without.l3_ is None
    # Same k, λ1 and λ2: the ablation isolates the prior.
    assert (without.k_, without.l1_, without.l2_) == (with_prior.k_, with_prior.l1_, with_prior.l2_)
    # Sets smaller than min_genes are zeroed, so a prior of only those is no prior.
    tiny = PLIER(max_iter=30, min_genes=100).fit(y, prior)
    assert not tiny.u_.any()
    np.testing.assert_allclose(tiny.z_, without.z_)
