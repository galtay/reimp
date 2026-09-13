"""The solver against the paper's definitions, on synthetic data."""

import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import scipy.stats
from sklearn.linear_model import Lasso

from reimp_plier.solver import (
    PLIER,
    annotate,
    b_step,
    elbow,
    hold_out,
    lambda_max,
    marchenko_pastur_median,
    noise_threshold,
    objective,
    optimal_threshold,
    spectrum,
    tune_l3,
    u_step,
    z_step,
)


def _simulated(seed: int = 0, n: int = 1000, p: int = 200, r: int = 10, noise_sets: int = 40):
    """The paper's simulation (bioRxiv v2, §2.2), smaller.

    Y = ZB + E with Z ~ Gamma(5, 1), B's columns Beta draws summing to one
    and E ~ N(0, 1), then z-scored per gene. C holds, for each true LV, the
    genes with its top 1-10% loadings (plus 10), and `noise_sets` random
    sets. The first r names are the true sets.
    """
    rng = np.random.default_rng(seed)
    z = rng.gamma(5, 1, (n, r))
    b = rng.beta(1, 3, (r, p))
    b /= b.sum(axis=0)
    y = z @ b * 10 + rng.normal(size=(n, p))
    y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
    sets = [np.argsort(-z[:, j])[: int(n * rng.integers(1, 11) / 100) + 10] for j in range(r)]
    sets += [rng.choice(n, rng.integers(10, 100), replace=False) for _ in range(noise_sets)]
    c = np.zeros((n, len(sets)))
    for s, genes in enumerate(sets):
        c[genes, s] = 1.0
    return y, c, [f"TRUE{s}" if s < r else f"NOISE{s}" for s in range(len(sets))]


def _lasso_problem(seed: int = 1, n: int = 200, m: int = 15, k: int = 10):
    rng = np.random.default_rng(seed)
    c = (rng.random((n, m)) < 0.15).astype(np.float64)
    z = np.maximum(rng.normal(size=(n, k)) + c[:, :k] * 2, 0)
    cs = scipy.sparse.csc_array(c)
    return z, c, np.asfortranarray(cs.T @ z), (cs.T @ cs).toarray()


@pytest.fixture(scope="module")
def simulated():
    return _simulated()


@pytest.fixture(scope="module")
def fitted(simulated):
    y, c, names = simulated
    return PLIER(seed=0).fit(y, c, names)


# ---- the three block updates


def test_b_step_is_the_ridge_minimizer() -> None:
    rng = np.random.default_rng(0)
    y, z, l2 = rng.normal(size=(30, 20)), rng.random((30, 4)), 2.5
    b = b_step(y, z, l2)
    # The gradient of ||Y - ZB||² + λ2 ||B||² vanishes there.
    np.testing.assert_allclose(z.T @ (z @ b - y) + l2 * b, 0, atol=1e-10)
    np.testing.assert_allclose(b, np.linalg.inv(z.T @ z + l2 * np.eye(4)) @ z.T @ y)


def test_z_step_is_the_unconstrained_minimizer_with_negatives_set_to_zero() -> None:
    rng = np.random.default_rng(0)
    y, b, cu, l1 = rng.normal(size=(40, 25)), rng.normal(size=(5, 25)), rng.random((40, 5)), 1.5
    # argmin_Z ||Y - ZB||² + λ1 ||Z - CU||²: (ZB - Y)Bᵀ + λ1 (Z - CU) = 0.
    free = (y @ b.T + l1 * cu) @ np.linalg.inv(b @ b.T + l1 * np.eye(5))
    np.testing.assert_allclose((free @ b - y) @ b.T + l1 * (free - cu), 0, atol=1e-10)
    assert (free < 0).any()
    np.testing.assert_allclose(z_step(y, b, l1, cu), np.maximum(free, 0))
    # U = 0 is the ridge fit λ1 ||Z||².
    ridge = y @ b.T @ np.linalg.inv(b @ b.T + l1 * np.eye(5))
    np.testing.assert_allclose(z_step(y, b, l1), np.maximum(ridge, 0))


def test_u_step_solves_the_nonnegative_lasso() -> None:
    z, c, ctz, gram = _lasso_problem()
    l3 = float(np.median(lambda_max(ctz)))
    u = u_step(z, ctz, gram, c, l3)
    assert (u >= 0).all() and u.any() and not u.all()
    # KKT for ||z - Cu||² + λ3 Σu, u ≥ 0: gradient 2(CᵀCu - Cᵀz) + λ3 is ≥ 0, and 0 where u > 0.
    grad = 2 * (gram @ u - ctz) + l3
    assert grad.min() > -1e-6
    np.testing.assert_allclose(grad[u > 0], 0, atol=1e-6)
    # scikit-learn's Lasso on C itself, with α = λ3 / 2n, agrees.
    for j in range(z.shape[1]):
        lasso = Lasso(alpha=l3 / (2 * len(c)), fit_intercept=False, positive=True, tol=1e-12)
        lasso.set_params(max_iter=100_000).fit(c, z[:, j])
        np.testing.assert_allclose(u[:, j], lasso.coef_, atol=1e-5)


def test_an_lv_uses_a_gene_set_exactly_below_lambda_max() -> None:
    z, c, ctz, gram = _lasso_problem()
    top = lambda_max(ctz)
    np.testing.assert_allclose(top, 2 * ctz.max(axis=0))
    for j, value in enumerate(top):
        assert not u_step(z, ctz, gram, c, value * 1.001)[:, j].any()
        assert u_step(z, ctz, gram, c, value * 0.999)[:, j].any()


@pytest.mark.parametrize("frac", [0.0, 0.3, 0.7, 1.0])
def test_l3_is_set_for_frac_of_the_lvs(frac) -> None:
    z, c, ctz, gram = _lasso_problem()
    u = u_step(z, ctz, gram, c, tune_l3(ctz, frac))
    assert int((u.sum(axis=0) > 0).sum()) == math.floor(frac * z.shape[1] + 0.5)


def test_l3_controls_the_sparsity_of_u() -> None:
    z, c, ctz, gram = _lasso_problem()
    grid = np.geomspace(1e-2, 1.0, 12) * lambda_max(ctz).max()
    nonzero = [int((u_step(z, ctz, gram, c, l3) > 0).sum()) for l3 in grid]
    assert nonzero[0] > nonzero[-2] > 0
    assert all(a >= b for a, b in zip(nonzero, nonzero[1:], strict=False))
    assert nonzero[-1] == 0  # λ3 at the largest λmax: U = 0


# ---- k, λ1 and λ2


def test_gavish_donoho_constants() -> None:
    # λ*(1) = 4/√3, the paper's title, and ω(1) ≈ 2.858 (Gavish and Donoho 2014).
    assert optimal_threshold(1.0) == pytest.approx(4 / math.sqrt(3))
    assert optimal_threshold(1.0) / math.sqrt(marchenko_pastur_median(1.0)) == pytest.approx(
        2.858, abs=1e-3
    )
    # Their cubic approximation of ω(β), to within 0.01.
    for beta in (0.1, 0.25, 0.5, 0.75, 0.9):
        omega = optimal_threshold(beta) / math.sqrt(marchenko_pastur_median(beta))
        cubic = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        assert omega == pytest.approx(cubic, abs=0.01)


def test_elbow_is_the_point_farthest_below_the_chord() -> None:
    # Scaled to [0, 1], the chord runs from (0, 1) to (1, 0); the value 2, at x = 3/9 and
    # 0.11 high, is the point farthest below it, so the three values before it are kept.
    values = np.array([10.0, 9.0, 8.0, 2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.0])
    assert elbow(values) == 3
    # A concave curve lies above its chord: no elbow.
    assert elbow(np.sqrt(np.linspace(25, 1, 20))) == 0
    assert elbow(np.array([2.0, 1.0])) == 2


def _low_rank(rank: int = 5, n: int = 400, p: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, p)) * 3 + rng.normal(size=(n, p))
    return y - y.mean(axis=1, keepdims=True)


@pytest.mark.parametrize("rule", ["elbow", "gavish_donoho"])
def test_k_is_twice_the_components_the_rule_keeps(rule) -> None:
    y = _low_rank(rank=5)
    # From the Gram matrix, so exact but for the smallest values (centring makes the last ~0).
    np.testing.assert_allclose(spectrum(y), np.linalg.svd(y, compute_uv=False), atol=1e-4)
    model = PLIER(k_rule=rule).fit(y)
    assert model.n_pc_ == 5 and model.k_ == 10
    if rule == "elbow":
        assert model.n_pc_ == elbow(model.singular_values_) and model.svd_threshold_ is None
    else:
        threshold = noise_threshold(model.singular_values_, y.shape)
        assert model.svd_threshold_ == pytest.approx(threshold)
        assert int((model.singular_values_ > threshold).sum()) == 5
    assert PLIER(k_rule=rule, k_factor=1.0).fit(y).k_ == 5
    # A fixed k skips the rule.
    fixed = PLIER(k=4).fit(y)
    assert fixed.n_pc_ is None and len(fixed.singular_values_) == 4


def test_k_is_capped_at_the_rank_of_y() -> None:
    y = np.random.default_rng(0).normal(size=(30, 12))
    y -= y.mean(axis=1, keepdims=True)
    assert PLIER(k=50).fit(y).k_ == 11


def test_lambdas_come_from_the_kth_singular_value(simulated) -> None:
    y, _, _ = simulated
    d = np.linalg.svd(y, compute_uv=False)
    model = PLIER(k=7).fit(y)
    assert model.l2_ == pytest.approx(d[6]) and model.l1_ == pytest.approx(d[6] / 2)
    np.testing.assert_allclose(model.singular_values_, d[:7])
    fixed = PLIER(k=7, l1=3.0, l2=4.0).fit(y)
    assert (fixed.l1_, fixed.l2_) == (3.0, 4.0)


# ---- the fit


def test_fit_keeps_z_and_u_nonnegative_and_lowers_the_objective(simulated, fitted) -> None:
    y, c, _ = simulated
    m = fitted
    assert (m.z_ >= 0).all() and (m.u_ >= 0).all() and m.u_.any()
    assert m.objective_[-1] < m.objective_[0]
    # The recorded objective is the paper's, computed directly.
    cu = m.prior_cv_ @ m.u_
    direct = (
        np.sum((y - m.z_ @ m.b_) ** 2)
        + m.l1_ * np.sum((m.z_ - cu) ** 2)
        + m.l2_ * np.sum(m.b_**2)
        + m.l3_ * m.u_.sum()
    )
    args = (np.sum(y**2), m.z_.T @ y, m.z_, m.b_)
    assert objective(*args, cu, m.u_, m.l1_, m.l2_, m.l3_) == pytest.approx(direct)
    # The U and B steps are exact minimizers, so neither raises it.
    cs = scipy.sparse.csc_array(m.prior_cv_)
    ctz = np.asfortranarray(cs.T @ m.z_)
    u = u_step(m.z_, ctz, (cs.T @ cs).toarray(), m.prior_cv_, m.l3_ * 0.5, m.u_)
    before = objective(*args, cs @ m.u_, m.u_, m.l1_, m.l2_, m.l3_ * 0.5)
    assert objective(*args, cs @ u, u, m.l1_, m.l2_, m.l3_ * 0.5) <= before + 1e-9 * abs(before)
    b = b_step(y, m.z_, m.l2_) + 0.01
    moved = objective(np.sum(y**2), m.z_.T @ y, m.z_, b, cu, m.u_, m.l1_, m.l2_, m.l3_)
    assert moved > objective(*args, cu, m.u_, m.l1_, m.l2_, m.l3_)


def test_with_a_fixed_l3_the_objective_falls_once_the_prior_enters(simulated, fitted) -> None:
    y, c, _ = simulated
    m = PLIER(seed=0, l3=fitted.l3_).fit(y, c)
    assert m.l3_ == fitted.l3_
    assert m.objective_[-1] < m.objective_[m.prior_iter_]


def test_the_prior_enters_once_the_no_prior_fit_converges(simulated, fitted) -> None:
    y, _, _ = simulated
    alone = PLIER(seed=0).fit(y)
    m = fitted
    # The no-prior ablation is the prior fit's first phase: same k, λ1, λ2 and iterates.
    assert (alone.k_, alone.l1_, alone.l2_) == (m.k_, m.l1_, m.l2_)
    assert alone.n_iter_ == m.prior_iter_ < m.n_iter_
    np.testing.assert_allclose(alone.objective_, m.objective_[: m.prior_iter_])
    # ... after which the prior moves Z, and 70% of the LVs use a gene set.
    assert not np.allclose(alone.z_, m.z_)
    assert int((m.u_.sum(axis=0) > 0).sum()) == pytest.approx(0.7 * m.k_, abs=1)
    assert alone.u_.shape == (0, m.k_) and alone.l3_ is None and alone.prior_iter_ is None
    assert alone.annotations_.empty


def test_recovers_the_simulated_gene_sets(fitted) -> None:
    annotated = fitted.annotated()
    true = {name for name in fitted.names_ if name.startswith("TRUE")}
    assert len(true & set(annotated["gene_set"])) >= len(true) - 1
    assert annotated.loc[annotated["gene_set"].isin(true), "auc"].min() > 0.9


def test_projection_is_multipliers_formula(simulated, fitted) -> None:
    y, _, _ = simulated
    z, l2 = fitted.z_, fitted.l2_
    new = np.random.default_rng(5).normal(size=(y.shape[0], 7))
    expected = np.linalg.inv(z.T @ z + l2 * np.eye(z.shape[1])) @ z.T @ new
    np.testing.assert_allclose(fitted.project(new), expected, atol=1e-10)
    # The training samples' projection is the fit's own B.
    np.testing.assert_allclose(fitted.project(y), fitted.b_, atol=1e-10)


def test_state_round_trip(simulated, fitted) -> None:
    y, _, _ = simulated
    again = PLIER.from_state(fitted.params(), fitted.state(), fitted.annotations_)
    np.testing.assert_array_equal(again.project(y), fitted.project(y))
    assert again.params() == fitted.params()
    assert (again.k_, again.l3_, again.prior_iter_) == (fitted.k_, fitted.l3_, fitted.prior_iter_)


# ---- gene-holdout cross-validation


def test_hold_out_removes_a_fifth_of_each_set_and_empties_small_ones() -> None:
    prior = np.zeros((40, 4))
    for s, size in enumerate([3, 5, 12, 20]):
        prior[:size, s] = 1
    c = hold_out(prior, 0.2, 5, np.random.default_rng(0))
    assert c.sum(axis=0).tolist() == [0, 4, 10, 16]
    assert (c <= prior).all()
    np.testing.assert_array_equal(c, hold_out(prior, 0.2, 5, np.random.default_rng(0)))


def test_annotations_score_held_out_genes_against_genes_outside_the_set() -> None:
    rng = np.random.default_rng(0)
    n = 60
    prior = np.zeros((n, 2))
    prior[:20, 0] = prior[10:30, 1] = 1
    prior_cv = prior.copy()
    prior_cv[:4, 0] = prior_cv[10:14, 1] = 0
    z = rng.random((n, 3))
    z[:4, 1] += 1  # set 0's held-out genes load on LV 1
    z[:5, 2] = 0.0  # ties with the zeros of a clipped Z
    z[30:40, 2] = 0.0
    u = np.zeros((2, 3))
    u[0, 1] = u[1, 0] = u[0, 2] = 0.5
    table = annotate(z, u, prior, prior_cv, ["A", "B"])
    assert table[["gene_set", "lv"]].values.tolist() == [["B", 0], ["A", 1], ["A", 2]]
    for row in table.itertuples():
        s = ["A", "B"].index(row.gene_set)
        pos = z[(prior[:, s] > 0) & (prior_cv[:, s] == 0), row.lv]
        neg = z[prior[:, s] == 0, row.lv]
        pairs = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        assert row.auc == pytest.approx(pairs.mean())
    assert table.loc[table["lv"] == 1, "auc"].item() == 1.0
    np.testing.assert_allclose(table["fdr"], scipy.stats.false_discovery_control(table["p_value"]))
    model = PLIER()
    model.annotations_ = table
    pd.testing.assert_frame_equal(
        model.annotated(), table[(table["auc"] > 0.7) & (table["fdr"] < 0.05)]
    )
