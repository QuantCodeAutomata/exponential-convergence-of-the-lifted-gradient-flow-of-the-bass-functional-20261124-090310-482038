"""Tests for exp/exp2.py: Lemma 2.3 (generic Gaussian-shift bound), Lemma 2.4
(Brenier-potential identity / M-Lipschitz v_Z), Lemma 2.5 (M-Lipschitz g),
and convexity of F = (M/2)||.||_2^2 - V (Baillon-Haddad).

All tests exercise the real numpy / POT (ot.emd) computations on small,
fast configurations -- no mocks.
"""

import numpy as np

from data.generate_synthetic import sample_bass_pair
from exp.exp2 import (
    _make_gamma,
    _random_Z,
    check_attainment,
    check_lemma_2_3,
    f_convexity_residual,
    f_max,
    f_norm,
    g_lipschitz_ratio,
    identity_discrepancy,
    lemma_2_3_mc_estimate,
    subgrad_max,
    subgrad_norm,
)
from src.bass_functional import make_bass_problem

H_GRID_SMALL = (0.2, 1.0)
N_MC_SMALL = 4000


def _small_problem(d: int, n_atoms: int = 8, seed: int = 123):
    synth = sample_bass_pair(d=d, n_atoms=n_atoms, radius=1.0, seed=seed)
    return make_bass_problem(synth)


# ---------------------------------------------------------------------------
# Lemma 2.3
# ---------------------------------------------------------------------------

def test_lemma_2_3_holds_d1():
    rng = np.random.default_rng(1)
    rows = check_lemma_2_3(d=1, rng=rng, n_mc=N_MC_SMALL, h_grid=H_GRID_SMALL, n_x_trials=3)
    # 2 test functions (euclidean_norm, max_coordinate) x 3 x-trials x 2 h-values
    assert len(rows) == 2 * 3 * len(H_GRID_SMALL)
    for row in rows:
        assert row["lower_holds"], row
        assert row["upper_holds"], row
        assert row["holds"], row


def test_lemma_2_3_holds_d2():
    rng = np.random.default_rng(2)
    rows = check_lemma_2_3(d=2, rng=rng, n_mc=N_MC_SMALL, h_grid=H_GRID_SMALL, n_x_trials=3)
    assert len(rows) == 2 * 3 * len(H_GRID_SMALL)
    for row in rows:
        assert row["holds"], row


def test_lemma_2_3_bound_scales_with_h_squared():
    """Sanity check that the reported bound is exactly (K/2)|h|^2."""
    rng = np.random.default_rng(3)
    rows = check_lemma_2_3(d=2, rng=rng, n_mc=1000, h_grid=(0.5, 1.5), n_x_trials=1)
    for row in rows:
        assert abs(row["bound"] - 0.5 * row["h_mag"] ** 2) < 1e-12


def test_lemma_2_3_zero_h_gives_zero_estimate_norm():
    """Delta_0 f = f(x) - f(x) - subgrad(x).0 = 0 identically, for any f."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=2) + 0.7
    h = np.zeros(2)
    gamma = rng.normal(size=(500, 2))
    result = lemma_2_3_mc_estimate(f_norm, subgrad_norm, 1.0, x, h, gamma)
    assert result["bound"] == 0.0
    assert abs(result["estimate"]) < 1e-9
    assert result["holds"]


def test_lemma_2_3_zero_h_gives_zero_estimate_max():
    rng = np.random.default_rng(5)
    x = rng.normal(size=3) + 0.7
    h = np.zeros(3)
    gamma = rng.normal(size=(500, 3))
    result = lemma_2_3_mc_estimate(f_max, subgrad_max, 1.0, x, h, gamma)
    assert result["bound"] == 0.0
    assert abs(result["estimate"]) < 1e-9
    assert result["holds"]


# ---------------------------------------------------------------------------
# Lemma 2.4: identity V(Z) = E[v_Z(Z+Gamma)] + E_nu1[v_Z*] - E[Z.X]
# ---------------------------------------------------------------------------

def test_identity_matches_bass_functional_value_d1():
    problem = _small_problem(d=1)
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(10)
    Z = _random_Z(problem, 1.0, rng)
    gamma = _make_gamma(n_x, d, k=4, seed=11)
    result = identity_discrepancy(problem, Z, gamma)
    assert np.isfinite(result["V"])
    assert np.isfinite(result["identity_value"])
    assert result["abs_diff"] < 0.2 * (abs(result["V"]) + 1e-6)


def test_identity_matches_bass_functional_value_d2():
    problem = _small_problem(d=2)
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(12)
    Z = _random_Z(problem, 1.0, rng)
    gamma = _make_gamma(n_x, d, k=4, seed=13)
    result = identity_discrepancy(problem, Z, gamma)
    assert result["abs_diff"] < 0.2 * (abs(result["V"]) + 1e-6)


# ---------------------------------------------------------------------------
# Lemma 2.5: ||g(W)-g(Z)||_2 / ||W-Z||_2 <= M
# ---------------------------------------------------------------------------

def test_g_lipschitz_ratio_bounded_by_M_small_problem():
    problem = _small_problem(d=2)
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(20)
    ratios = []
    for p in range(6):
        Z = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        W = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        # Common random numbers: shared Gamma for Z and W, see exp2.py's
        # g_lipschitz_ratio docstring for why this matters at small K_GAMMA.
        gamma = _make_gamma(n_x, d, k=4, seed=100 + p)
        result = g_lipschitz_ratio(problem, Z, W, gamma, gamma)
        ratios.append(result["ratio"])
        assert result["holds"], result
    assert max(ratios) <= problem.M * 1.05


def test_g_lipschitz_ratio_bounded_by_M_d1():
    problem = _small_problem(d=1)
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(21)
    for p in range(6):
        Z = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        W = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        gamma = _make_gamma(n_x, d, k=4, seed=200 + p)
        result = g_lipschitz_ratio(problem, Z, W, gamma, gamma)
        assert result["ratio"] <= problem.M * 1.05, result


# ---------------------------------------------------------------------------
# Baillon-Haddad / convexity of F(Z) = (M/2)||Z||_2^2 - V(Z)
# ---------------------------------------------------------------------------

def test_f_convexity_residual_nonnegative_small_problem():
    problem = _small_problem(d=2)
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(30)
    for p in range(6):
        Z = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        W = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        gamma = _make_gamma(n_x, d, k=4, seed=300 + p)
        result = f_convexity_residual(problem, Z, W, gamma, gamma)
        assert result["holds"], result
        scale = max(abs(result["F_Z"]), abs(result["F_W"]), abs(result["linear_term"]), 1.0)
        assert result["residual"] >= -0.02 * scale


def test_f_convexity_residual_nonnegative_d1():
    problem = _small_problem(d=1)
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(31)
    for p in range(6):
        Z = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        W = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        gamma = _make_gamma(n_x, d, k=4, seed=400 + p)
        result = f_convexity_residual(problem, Z, W, gamma, gamma)
        assert result["holds"], result


# ---------------------------------------------------------------------------
# Property 1.2 (attainment), reused from exp_1
# ---------------------------------------------------------------------------

def test_attainment_small_wasserstein_distance():
    problem = _small_problem(d=2)
    n_x, d = problem.x_atoms.shape
    gamma = _make_gamma(n_x, d, k=10, seed=500)
    result = check_attainment(problem, gamma)
    assert np.isfinite(result["wasserstein2_Y_Z_to_nu1"])
    assert result["relative_w2"] < 0.5
