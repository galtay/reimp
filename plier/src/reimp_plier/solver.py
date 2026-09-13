"""PLIER's solver, written from the papers.

Mao, Zaslavsky, Hartmann, Sealfon and Chikina, "Pathway-level information
extractor (PLIER) for gene expression data", Nature Methods 16:607-610
(2019), Methods; new samples are projected as in MultiPLIER (Taroni et al.,
Cell Systems 8:380-394, 2019). Where the papers leave a choice open, the
choice made here is reimp's own; each is marked "ours", and
plier/README.md lists them.

Notation is the paper's. Y is genes x samples, z-scored per gene. C is the
binary genes x gene-sets prior, Z (genes x k) the loadings, B (k x samples)
the latent variables (LVs) and U (gene sets x k) the prior's coefficients.
PLIER minimizes

    ||Y - ZB||² + λ1 ||Z - CU||² + λ2 ||B||² + λ3 ||U||₁    subject to Z ≥ 0, U ≥ 0

by block coordinate descent from the SVD of Y (Y = UDVᵀ, so Z ≈ UD^½ and
B ≈ D^½Vᵀ):

- Z step: the minimizer of the first two terms, (YBᵀ + λ1 CU)(BBᵀ + λ1 I)⁻¹,
  with its negative part set to 0, as the paper does;
- U step: per LV, a non-negative lasso of Z's column on C;
- B step: (ZᵀZ + λ2 I)⁻¹ZᵀY, which is also how MultiPLIER projects new
  samples.

Paper defaults: λ2 = d_k and λ1 = d_k / 2 (d_k the k-th singular value of
Y); λ3 set so that a fraction `frac` = 0.7 of the LVs use a gene set; k
twice the number of principal components worth keeping, found by an elbow
or a significance rule; stopping when the relative change in B falls below
5e-6 or levels off. Gene-holdout cross-validation: a random fifth of every
gene set's genes is left out of C, and each positive entry of U is scored
by the AUC with which that Z column ranks the set's held-out genes above
the genes outside the set.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Literal

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import scipy.stats
from scipy.integrate import quad
from scipy.optimize import brentq
from sklearn.linear_model import lasso_path
from sklearn.utils.extmath import randomized_svd

log = logging.getLogger(__name__)

# Relative duality-gap tolerance of scikit-learn's coordinate descent in the U step.
LASSO_TOL = 1e-8
# At or below this many samples (or genes), the SVD is exact rather than randomized.
EXACT_SVD = 1000


def z_step(y: np.ndarray, b: np.ndarray, l1: float, cu: np.ndarray | None = None) -> np.ndarray:
    """(YBᵀ + λ1 CU)(BBᵀ + λ1 I)⁻¹ with negatives set to 0; `cu` None means U = 0."""
    rhs = y @ b.T
    if cu is not None:
        rhs += l1 * cu
    gram = b @ b.T + l1 * np.eye(len(b))
    z = scipy.linalg.solve(gram, rhs.T, assume_a="pos").T
    return np.maximum(z, 0.0, out=z)


def b_step(y: np.ndarray, z: np.ndarray, l2: float) -> np.ndarray:
    """(ZᵀZ + λ2 I)⁻¹ZᵀY, the minimizer of ||Y - ZB||² + λ2 ||B||² over B."""
    gram = z.T @ z + l2 * np.eye(z.shape[1])
    return scipy.linalg.solve(gram, z.T @ y, assume_a="pos")


def lambda_max(ctz: np.ndarray) -> np.ndarray:
    """Per LV, the λ3 at and above which its U column is 0: 2 max_s (Cᵀz)_s.

    With U ≥ 0, U = 0 solves the LV's lasso exactly when no gene set's
    gradient, -2 (Cᵀz)_s, outweighs the penalty λ3 (the KKT conditions).
    `ctz` is CᵀZ, gene sets x LVs.
    """
    if ctz.shape[0] == 0:
        return np.zeros(ctz.shape[1])
    return 2.0 * np.maximum(ctz.max(axis=0), 0.0)


def tune_l3(ctz: np.ndarray, frac: float) -> float:
    """λ3 at which round(frac k) LVs use a gene set (ours: the paper's binary search, solved).

    An LV uses a gene set exactly when λ3 < `lambda_max`, so the paper's
    target is met by a λ3 between the target-th and next largest λmax: we
    take their midpoint. LVs tied at that boundary all fall on one side.
    """
    k = ctz.shape[1]
    target = min(k, max(0, math.floor(frac * k + 0.5)))
    top = np.sort(lambda_max(ctz))[::-1]
    if target == 0:
        l3 = top[0]
    elif target == k:
        l3 = top[-1] / 2
    else:
        l3 = (top[target - 1] + top[target]) / 2
    return float(l3) if l3 > 0 else 1.0


def u_step(
    z: np.ndarray,
    ctz: np.ndarray,
    gram: np.ndarray,
    c: np.ndarray,
    l3: float,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """argmin over U ≥ 0 of ||Z - CU||² + λ3 ||U||₁, one LV at a time.

    Ours: scikit-learn's coordinate descent (`lasso_path`, positive, no
    intercept) on the precomputed Gram matrix `gram` = CᵀC and `ctz` =
    CᵀZ, warm-started from `u`. LVs whose `lambda_max` is at most λ3 are 0
    without a fit. `c` is C itself, dense; scikit-learn needs it only for
    its shape on this path. Its objective is ||z - Cu||² / (2n) + α||u||₁,
    so α = λ3 / 2n.
    """
    m, k = ctz.shape
    out = np.zeros((m, k))
    if m == 0:
        return out
    alpha = l3 / (2 * c.shape[0])
    for j in np.flatnonzero(lambda_max(ctz) > l3):
        start = None if u is None else np.ascontiguousarray(u[:, j])
        _, coefs, _ = lasso_path(
            c,
            np.ascontiguousarray(z[:, j]),
            alphas=[alpha],
            precompute=gram,
            Xy=np.ascontiguousarray(ctz[:, j]),
            coef_init=start,
            positive=True,
            tol=LASSO_TOL,
            max_iter=10_000,
            check_input=False,
        )
        out[:, j] = coefs[:, 0]
    return out


def objective(y_norm2, zty, z, b, cu, u, l1, l2, l3) -> float:
    """The paper's objective, with ||Y - ZB||² from ||Y||², ZᵀY and ZᵀZ rather than Y."""
    fit = y_norm2 - 2 * np.vdot(zty, b) + np.vdot(z.T @ z, b @ b.T)
    prior = np.sum(z**2) if cu is None else np.sum((z - cu) ** 2)
    return float(fit + l1 * prior + l2 * np.sum(b**2) + l3 * np.abs(u).sum())


# ---- how many principal components (ours; the paper names an elbow or significance)


def elbow(values: np.ndarray) -> int:
    """The number of components before the elbow of the scree curve, by the chord rule.

    With the component index and the singular values both scaled to
    [0, 1], the elbow is the point farthest below the straight line from
    the first value to the last; the components before it are kept.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3 or values[0] <= values[-1]:
        return len(values)
    x = np.linspace(0.0, 1.0, len(values))
    scaled = (values - values[-1]) / (values[0] - values[-1])
    return int(np.argmax((1.0 - x) - scaled))


# Gavish and Donoho, "The optimal hard threshold for singular values is 4/√3",
# IEEE Trans. Inf. Theory 60:5040, 2014. For an m x n matrix (m ≤ n, β = m/n) of
# low rank plus white noise of unknown level, singular values below ω(β) times
# the median singular value are noise; ω(β) = λ*(β) / √μ_β, μ_β the median of
# the Marchenko-Pastur law.


def optimal_threshold(beta: float) -> float:
    """λ*(β): the hard threshold for known noise σ is λ*(β) √n σ."""
    return math.sqrt(2 * (beta + 1) + 8 * beta / (beta + 1 + math.sqrt(beta**2 + 14 * beta + 1)))


@cache
def marchenko_pastur_median(beta: float) -> float:
    """The median of the Marchenko-Pastur law with aspect ratio β ∈ (0, 1] and unit variance."""
    lo, hi = (1 - math.sqrt(beta)) ** 2, (1 + math.sqrt(beta)) ** 2

    def density(x: float) -> float:
        return math.sqrt(max((hi - x) * (x - lo), 0.0)) / (2 * math.pi * beta * x)

    def cdf(x: float) -> float:
        return quad(density, lo, x, limit=200)[0]

    return brentq(lambda x: cdf(x) - 0.5, lo, hi, xtol=1e-12)


def noise_threshold(singular_values: np.ndarray, shape: tuple[int, int]) -> float:
    """ω(β) times the median of the full spectrum `singular_values` of a `shape` matrix."""
    m, n = sorted(shape)
    beta = m / n
    omega = optimal_threshold(beta) / math.sqrt(marchenko_pastur_median(beta))
    return float(omega * np.median(singular_values))


def spectrum(y: np.ndarray) -> np.ndarray:
    """Every singular value of `y`, largest first, from the eigenvalues of its smaller Gram."""
    gram = y.T @ y if y.shape[0] >= y.shape[1] else y @ y.T
    values = scipy.linalg.eigvalsh(gram, overwrite_a=True)
    return np.sqrt(np.clip(values[::-1], 0.0, None))


def top_svd(y: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The leading k singular triplets: exact for small matrices, else randomized.

    Signs (ours): each pair (u_j, v_j) is flipped so that u_j's positive part
    is the larger, which is the part the non-negative Z keeps.
    """
    if min(y.shape) <= EXACT_SVD:
        u, d, vt = np.linalg.svd(y, full_matrices=False)
        u, d, vt = u[:, :k], d[:k], vt[:k]
    else:
        u, d, vt = randomized_svd(y, k, n_oversamples=10, random_state=seed)
    flip = np.sum(np.maximum(u, 0) ** 2, axis=0) < np.sum(np.minimum(u, 0) ** 2, axis=0)
    u[:, flip] *= -1
    vt[flip] *= -1
    return u, d, vt


# ---- gene-holdout cross-validation


def hold_out(
    prior: np.ndarray, fraction: float, min_genes: int, rng: np.random.Generator
) -> np.ndarray:
    """C for training: a random `fraction` of each set's genes removed, small sets emptied.

    Sets with fewer than `min_genes` genes (ours) are emptied, so they never
    enter U. From each other set, floor(fraction x size) genes, drawn at
    random, are set to 0 (the paper holds out a fifth).
    """
    c = prior.astype(np.float64, copy=True)
    for s in range(c.shape[1]):
        members = np.flatnonzero(c[:, s])
        if len(members) < min_genes:
            c[:, s] = 0.0
            continue
        n_held = int(fraction * len(members))
        if n_held:
            c[rng.choice(members, n_held, replace=False), s] = 0.0
    return c


def annotate(
    z: np.ndarray, u: np.ndarray, prior: np.ndarray, prior_cv: np.ndarray, names: Sequence[str]
) -> pd.DataFrame:
    """One row per positive U entry: the held-out AUC of its set on its Z column.

    For gene set s and LV i with U[s, i] > 0 (the paper): positives are
    the genes of s held out of training, negatives the genes not in s,
    scored by Z[:, i]. AUC is the Mann-Whitney U over the number of pairs
    (ties count a half); p is scipy's one-sided Mann-Whitney test and FDR
    is Benjamini-Hochberg over every row with a p-value (ours: the paper
    names AUC, p and FDR, not the test). Sets with no held-out genes get
    NaN.
    """
    columns = ["gene_set", "lv", "u", "auc", "p_value", "fdr"]
    lvs, sets = np.nonzero(u.T > 0)
    rows = []
    for lv, s in zip(lvs, sets, strict=True):
        members = prior[:, s] > 0
        held = members & (prior_cv[:, s] == 0)
        auc = p = np.nan
        if held.any() and not members.all():
            test = scipy.stats.mannwhitneyu(z[held, lv], z[~members, lv], alternative="greater")
            auc = test.statistic / (held.sum() * (~members).sum())
            p = test.pvalue
        rows.append((names[s], int(lv), float(u[s, lv]), auc, p))
    table = pd.DataFrame(rows, columns=columns[:-1]).astype({"lv": np.int64})
    table["fdr"] = np.nan
    tested = table["p_value"].notna().to_numpy()
    if tested.any():
        table.loc[tested, "fdr"] = scipy.stats.false_discovery_control(
            table.loc[tested, "p_value"].to_numpy()
        )
    return table[columns]


class PLIER:
    """PLIER, fit by the paper's block coordinate descent (see the module docstring).

    Args:
        k: number of LVs; null for `k_factor` times the number of principal
            components that `k_rule` keeps.
        k_rule: "elbow", the components before the elbow of Y's singular
            values (chord rule), or "gavish_donoho", those above Gavish and
            Donoho's noise threshold.
        k_factor: multiplies that number of components (paper: 2).
        l1: λ1; null for d_k / 2 (paper).
        l2: λ2; null for d_k (paper).
        l3: λ3; null to set it so that `frac` of the LVs use a gene set.
        frac: the share of LVs with a gene set that λ3 is set for (paper: 0.7).
        l3_every: λ3 is set again every this many iterations with the prior.
        holdout: share of each gene set's genes held out of C, for the
            annotations (paper: 1/5); 0 annotates nothing.
        min_genes: gene sets with fewer genes than this are left out of C.
        max_iter: iteration cap for each phase, without the prior and with it.
        tol: stop a phase when ||ΔB|| / ||B|| falls below this (paper: 5e-6).
        patience: or when that ratio has not reached a new low for this many
            iterations (the paper's "leveling off").
        seed: seeds the randomized SVD and the held-out genes.
    """

    def __init__(
        self,
        k: int | None = None,
        k_rule: Literal["elbow", "gavish_donoho"] = "elbow",
        k_factor: float = 2.0,
        l1: float | None = None,
        l2: float | None = None,
        l3: float | None = None,
        frac: float = 0.7,
        l3_every: int = 10,
        holdout: float = 0.2,
        min_genes: int = 5,
        max_iter: int = 300,
        tol: float = 5e-6,
        patience: int = 20,
        seed: int = 0,
    ) -> None:
        if k is not None and k < 1:
            raise ValueError("k must be at least 1")
        if k_rule not in ("elbow", "gavish_donoho"):
            raise ValueError(f"unknown k_rule {k_rule!r}")
        if not 0 <= frac <= 1 or not 0 <= holdout < 1:
            raise ValueError("frac must be in [0, 1] and holdout in [0, 1)")
        if max_iter < 1 or l3_every < 1 or patience < 1:
            raise ValueError("max_iter, l3_every and patience must be at least 1")
        self.k = k
        self.k_rule = k_rule
        self.k_factor = k_factor
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.frac = frac
        self.l3_every = l3_every
        self.holdout = holdout
        self.min_genes = min_genes
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.seed = seed

    def params(self) -> dict:
        """The constructor's arguments, as `PLIER(**params)` takes them."""
        names = (
            "k k_rule k_factor l1 l2 l3 frac l3_every holdout min_genes max_iter tol patience seed"
        )
        return {name: getattr(self, name) for name in names.split()}

    # ---- fitting

    def choose_k(self, y: np.ndarray) -> int:
        """k, from `k` or from the rule, which also sets `n_pc_` and `singular_values_`."""
        rank = min(y.shape[0], y.shape[1] - 1)  # Y's rows are centred
        if self.k is not None:
            if self.k > rank:
                log.warning("k = %d exceeds Y's rank %d; using %d", self.k, rank, rank)
            return min(self.k, rank)
        values = spectrum(y)
        self.singular_values_ = values
        if self.k_rule == "elbow":
            self.n_pc_ = elbow(values)
        else:
            self.svd_threshold_ = noise_threshold(values, y.shape)
            self.n_pc_ = int((values > self.svd_threshold_).sum())
        k = min(rank, max(1, round(self.k_factor * self.n_pc_)))
        log.info("k: %d components (%s) x %g = %d", self.n_pc_, self.k_rule, self.k_factor, k)
        return k

    def fit(
        self, y: np.ndarray, prior: np.ndarray | None = None, names: Sequence[str] | None = None
    ) -> PLIER:
        """Fit on `y`, genes x samples, z-scored per gene; `prior` is C, genes x sets.

        Without `prior` this is the no-prior ablation: U = 0, so λ1 ||Z||²
        replaces λ1 ||Z - CU||². With it, the same U = 0 factorization runs
        first, to convergence, and the prior then enters (ours: the paper
        does not say when U starts). The no-prior fit is therefore exactly
        the first phase of the prior's fit, at the same k, λ1 and λ2.
        """
        y = np.asarray(y, dtype=np.float64)
        n, p = y.shape
        rng = np.random.default_rng(self.seed)
        self.singular_values_, self.svd_threshold_, self.n_pc_ = None, None, None
        k = self.choose_k(y)
        _, d, vt = top_svd(y, k, self.seed)
        if self.singular_values_ is None:
            self.singular_values_ = d
        self.k_ = k
        self.l2_ = float(d[k - 1]) if self.l2 is None else float(self.l2)
        self.l1_ = self.l2_ / 2 if self.l1 is None else float(self.l1)

        if prior is None:
            prior = np.zeros((n, 0))
            names = []
        prior = np.asarray(prior)
        if prior.shape[0] != n:
            raise ValueError(f"the prior has {prior.shape[0]} genes, Y has {n}")
        names = list(names) if names is not None else [f"set{s}" for s in range(prior.shape[1])]
        m = prior.shape[1]
        c = hold_out(prior, self.holdout, self.min_genes, rng)
        self.prior_, self.prior_cv_, self.names_ = prior, c, names
        if m:
            used = int((c.sum(axis=0) > 0).sum())
            log.info("prior: %d of %d gene sets have at least %d genes", used, m, self.min_genes)
            c_sparse = scipy.sparse.csc_array(c)
            gram = (c_sparse.T @ c_sparse).toarray()

        y_norm2 = float(np.sum(y**2))
        b = np.sqrt(d)[:, None] * vt
        z = None  # the first phase always runs, and starts with a Z step
        u = np.zeros((m, k))
        cu = None
        l3 = 0.0
        changes, objectives = [], []
        self.prior_iter_ = None
        for with_prior in [False] + ([True] if m else []):
            if with_prior:
                self.prior_iter_ = len(changes)
            best, since_best = np.inf, 0
            for i in range(self.max_iter):
                if with_prior:
                    ctz = np.asfortranarray(c_sparse.T @ z)
                    if self.l3 is not None:
                        l3 = float(self.l3)
                    elif i % self.l3_every == 0:
                        l3 = tune_l3(ctz, self.frac)
                    u = u_step(z, ctz, gram, c, l3, u)
                    cu = c_sparse @ u
                z = z_step(y, b, self.l1_, cu)
                zty = z.T @ y
                b_new = scipy.linalg.solve(z.T @ z + self.l2_ * np.eye(k), zty, assume_a="pos")
                change = float(np.linalg.norm(b_new - b) / max(np.linalg.norm(b), 1e-300))
                b = b_new
                changes.append(change)
                objectives.append(objective(y_norm2, zty, z, b, cu, u, self.l1_, self.l2_, l3))
                if len(changes) % 10 == 0:
                    log.info(
                        "iteration %d: ||dB||/||B|| %.3g, objective %.6g, lambda3 %.4g, "
                        "LVs with a gene set %d",
                        len(changes),
                        change,
                        objectives[-1],
                        l3,
                        int((u.sum(axis=0) > 0).sum()),
                    )
                if change < self.tol:
                    break
                if change < best:
                    best, since_best = change, 0
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break

        self.z_, self.b_, self.u_ = z, b, u
        self.l3_ = l3 if m else None
        self.n_iter_ = len(changes)
        self.change_ = np.array(changes)
        self.objective_ = np.array(objectives)
        if m and self.holdout > 0:
            self.annotations_ = annotate(z, u, prior, c, names)
        else:
            self.annotations_ = annotate(z, np.zeros((0, k)), prior[:, :0], c[:, :0], [])
        return self

    # ---- using a fit

    def project(self, y: np.ndarray) -> np.ndarray:
        """B for `y`, genes x samples: (ZᵀZ + λ2 I)⁻¹Zᵀy (MultiPLIER), k x samples."""
        return b_step(np.asarray(y, dtype=np.float64), self.z_, self.l2_)

    def annotated(self, auc: float = 0.7, fdr: float = 0.05) -> pd.DataFrame:
        """The annotations the paper calls high confidence: AUC > 0.7 and FDR < 0.05."""
        a = self.annotations_
        return a[(a["auc"] > auc) & (a["fdr"] < fdr)]

    # ---- saving

    def state(self) -> dict[str, np.ndarray]:
        """Every fitted array and number, for `np.savez`; `from_state` rebuilds the model."""
        return {
            "z": self.z_,
            "b": self.b_,
            "u": self.u_,
            "prior": self.prior_.astype(bool),
            "prior_cv": self.prior_cv_.astype(bool),
            "gene_set": np.array(self.names_, dtype=str),
            "singular_values": self.singular_values_,
            "n_pc": -1 if self.n_pc_ is None else self.n_pc_,
            "svd_threshold": np.nan if self.svd_threshold_ is None else self.svd_threshold_,
            "change": self.change_,
            "objective": self.objective_,
            "k": self.k_,
            "l1": self.l1_,
            "l2": self.l2_,
            "l3": np.nan if self.l3_ is None else self.l3_,
            "n_iter": self.n_iter_,
            "prior_iter": -1 if self.prior_iter_ is None else self.prior_iter_,
        }

    @classmethod
    def from_state(
        cls, params: Mapping, state: Mapping[str, np.ndarray], annotations: pd.DataFrame
    ) -> PLIER:
        def optional(value, missing):
            return None if value == missing or np.isnan(value) else value

        model = cls(**params)
        model.z_, model.b_, model.u_ = state["z"], state["b"], state["u"]
        model.prior_ = state["prior"].astype(np.float64)
        model.prior_cv_ = state["prior_cv"].astype(np.float64)
        model.names_ = state["gene_set"].tolist()
        model.singular_values_ = state["singular_values"]
        n_pc = optional(int(state["n_pc"]), -1)
        model.n_pc_ = n_pc
        model.svd_threshold_ = optional(float(state["svd_threshold"]), None)
        model.change_, model.objective_ = state["change"], state["objective"]
        model.k_, model.n_iter_ = int(state["k"]), int(state["n_iter"])
        model.l1_, model.l2_ = float(state["l1"]), float(state["l2"])
        model.l3_ = optional(float(state["l3"]), None)
        model.prior_iter_ = optional(int(state["prior_iter"]), -1)
        model.annotations_ = annotations
        return model
