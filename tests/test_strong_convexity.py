import numpy as np

from src.strong_convexity import compute_b, compute_L_and_mR, lemma_3_1_bound, unit_directions


def test_unit_directions_are_normalized():
    d = unit_directions(3, 16, seed=1)
    norms = np.linalg.norm(d, axis=1)
    assert np.allclose(norms, 1.0)


def test_compute_b_is_strictly_positive(small_problem_1d):
    b = compute_b(small_problem_1d, n_directions=8, seed=0)
    assert b > 0


def test_compute_b_positive_in_2d(small_problem_2d):
    b = compute_b(small_problem_2d, n_directions=16, seed=0)
    assert b > 0


def test_m_R_is_positive_and_decreases_with_R(small_problem_1d):
    b = compute_b(small_problem_1d, n_directions=8, seed=0)
    M = small_problem_1d.M
    d = small_problem_1d.d
    # m_R = b*exp(-(L+R)^2)/(2L*2^(d/2)) underflows float64 at these O(1)
    # scales (exp(-(L+R)^2) with (L+R)^2 of order 10^2-10^3); log_m_R stays
    # finite and lets us check m_R > 0 and its decrease in R in log-space.
    log_small = compute_L_and_mR(M, b, R=0.3, d=d)["log_m_R"]
    log_large = compute_L_and_mR(M, b, R=2.0, d=d)["log_m_R"]
    assert np.isfinite(log_small) and np.isfinite(log_large)
    # the exp(-(L+R)^2) factor decays faster than any growth from 1/L, so the
    # strong-convexity constant shrinks as the L^infinity bound R grows
    assert log_large < log_small


def test_lemma_3_1_bound_holds_above_threshold(small_problem_1d):
    problem = small_problem_1d
    b = compute_b(problem, n_directions=8, seed=0)
    M = problem.M
    d = problem.d
    Q = np.zeros_like(problem.x_atoms)
    gamma_for_Q = np.random.default_rng(1).normal(size=(problem.x_atoms.shape[0], 4, d))
    e = np.array([1.0]) if d == 1 else np.eye(d)[0]
    x = problem.x_atoms[0]
    threshold = M * (2.0 * 0.0 + np.sqrt(d)) / b
    rho = 2.0 * threshold + 1.0
    result = lemma_3_1_bound(
        problem, Q, gamma_for_Q, x, e, rho=rho, sigma=1.0, b=b, M=M, n_mc=4000, seed=2
    )
    assert result["rho_above_threshold"]
    # the Monte Carlo directional-derivative estimate should exceed the
    # analytic lower bound of Lemma 3.1 (up to Monte Carlo noise)
    assert result["mc_estimate"] >= result["analytic_lower_bound"] - 0.2
