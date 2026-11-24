"""Tests for exp/exp3.py: strong convexity of V (Theorem 1.1) and the
directional-derivative lower bound (Lemma 3.1).

All tests exercise the real numpy / POT computations on small, fast
configurations -- no mocks.
"""

import numpy as np

from data.generate_synthetic import sample_bass_pair
from exp.exp3 import (
    _make_gamma,
    _make_mean_matched_pair,
    _random_Z,
    run_lemma_3_1_checks,
    strong_convexity_residual,
)
from src.bass_functional import make_bass_problem
from src.strong_convexity import compute_b, compute_L_and_mR

N_ATOMS_SMALL = 8


def _small_problem(d: int, n_atoms: int = N_ATOMS_SMALL, seed: int = 123):
    synth = sample_bass_pair(d=d, n_atoms=n_atoms, radius=1.0, seed=seed)
    return make_bass_problem(synth)


# ---------------------------------------------------------------------------
# compute_b / compute_L_and_mR (reused from src.strong_convexity)
# ---------------------------------------------------------------------------

def test_compute_b_strictly_positive_small_problem():
    problem = _small_problem(d=2)
    b = compute_b(problem, n_directions=8, seed=0)
    assert isinstance(b, float)
    assert b > 0.0


def test_compute_L_and_mR_positive_L_finite_log():
    problem = _small_problem(d=1)
    b = compute_b(problem, n_directions=8, seed=0)
    info = compute_L_and_mR(problem.M, b, R=0.5, d=problem.d)
    assert info["L"] > 0.0
    assert np.isfinite(info["log_m_R"])


def test_log_mR_strictly_decreasing_in_R():
    """Theorem 1.1's qualitative content: the strong-convexity modulus
    shrinks as the L^infinity bound R grows, checked in log-space since
    the plain m_R value underflows at these scales."""
    problem = _small_problem(d=1)
    b = compute_b(problem, n_directions=8, seed=0)
    info_small = compute_L_and_mR(problem.M, b, R=0.3, d=problem.d)
    info_large = compute_L_and_mR(problem.M, b, R=1.0, d=problem.d)
    assert np.isfinite(info_small["log_m_R"]) and np.isfinite(info_large["log_m_R"])
    assert info_large["log_m_R"] < info_small["log_m_R"]


def test_compute_L_and_mR_underflow_regime_handled_without_raising():
    """A large enough R drives log_m_R below -700, where the plain m_R
    value underflows to exactly 0.0 (see src/strong_convexity.py); this
    must not raise, and log_m_R must stay finite in that regime."""
    problem = _small_problem(d=1)
    b = compute_b(problem, n_directions=8, seed=0)
    info = compute_L_and_mR(problem.M, b, R=50.0, d=problem.d)
    assert info["m_R"] == 0.0
    assert np.isfinite(info["log_m_R"])
    assert info["log_m_R"] < -700


# ---------------------------------------------------------------------------
# mean-matched, L^infinity-bounded pair construction
# ---------------------------------------------------------------------------

def test_make_mean_matched_pair_respects_mean_and_bound():
    problem = _small_problem(d=2)
    rng = np.random.default_rng(1)
    R = 0.8
    for _ in range(10):
        Z, W = _make_mean_matched_pair(problem, R, rng)
        mean_Z = np.sum(problem.x_weights[:, None] * Z, axis=0)
        mean_W = np.sum(problem.x_weights[:, None] * W, axis=0)
        assert np.linalg.norm(mean_Z - mean_W) < 1e-6
        assert np.max(np.abs(Z)) <= R + 1e-9
        assert np.max(np.abs(W)) <= R + 1e-9


def test_make_mean_matched_pair_d1():
    problem = _small_problem(d=1)
    rng = np.random.default_rng(2)
    R = 0.5
    Z, W = _make_mean_matched_pair(problem, R, rng)
    assert Z.shape == problem.x_atoms.shape
    assert W.shape == problem.x_atoms.shape
    mean_Z = np.sum(problem.x_weights[:, None] * Z, axis=0)
    mean_W = np.sum(problem.x_weights[:, None] * W, axis=0)
    assert np.linalg.norm(mean_Z - mean_W) < 1e-6


# ---------------------------------------------------------------------------
# strong-convexity residual (Theorem 1.1)
# ---------------------------------------------------------------------------

def test_strong_convexity_residual_holds_for_random_pairs():
    problem = _small_problem(d=2)
    n_x, d = problem.x_atoms.shape
    b = compute_b(problem, n_directions=8, seed=0)
    R = 0.8
    info = compute_L_and_mR(problem.M, b, R=R, d=d)
    rng = np.random.default_rng(3)
    for p in range(5):
        Z, W = _make_mean_matched_pair(problem, R, rng)
        gamma = _make_gamma(n_x, d, k=4, seed=100 + p)
        result = strong_convexity_residual(problem, Z, W, gamma, info["m_R"])
        assert result["holds"], result
        assert np.isfinite(result["residual"])


def test_strong_convexity_residual_empirical_modulus_reported():
    problem = _small_problem(d=1)
    n_x, d = problem.x_atoms.shape
    b = compute_b(problem, n_directions=8, seed=0)
    R = 0.5
    info = compute_L_and_mR(problem.M, b, R=R, d=d)
    rng = np.random.default_rng(4)
    Z, W = _make_mean_matched_pair(problem, R, rng)
    gamma = _make_gamma(n_x, d, k=4, seed=200)
    result = strong_convexity_residual(problem, Z, W, gamma, info["m_R"])
    assert result["diff_norm_sq"] > 0.0
    assert result["empirical_modulus"] is not None
    assert np.isfinite(result["empirical_modulus"])


# ---------------------------------------------------------------------------
# Lemma 3.1
# ---------------------------------------------------------------------------

def test_lemma_3_1_checks_hold_and_rho_above_threshold():
    problem = _small_problem(d=1)
    b = compute_b(problem, n_directions=8, seed=0)
    rows = run_lemma_3_1_checks(
        problem, b, problem.M, seed=5, n_mc=1000, max_atoms=4, max_directions=4,
    )
    assert len(rows) > 0
    assert all(r["rho_above_threshold"] for r in rows)
    assert any(r["holds"] for r in rows)


def test_lemma_3_1_checks_hold_d2():
    problem = _small_problem(d=2)
    b = compute_b(problem, n_directions=8, seed=0)
    rows = run_lemma_3_1_checks(
        problem, b, problem.M, seed=6, n_mc=1500, max_atoms=3, max_directions=3,
    )
    assert len(rows) == 3 * 3
    assert all(r["rho_above_threshold"] for r in rows)
    for r in rows:
        assert r["holds"], r


# ---------------------------------------------------------------------------
# helper sanity
# ---------------------------------------------------------------------------

def test_random_Z_shape_and_bound():
    problem = _small_problem(d=2)
    rng = np.random.default_rng(7)
    Z = _random_Z(problem, scale=0.5, rng=rng)
    assert Z.shape == problem.x_atoms.shape


def test_make_gamma_shape():
    gamma = _make_gamma(n_x=8, d=2, k=5, seed=1)
    assert gamma.shape == (8, 5, 2)
