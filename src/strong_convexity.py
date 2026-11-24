"""Strong-convexity constant m_R and the directional-derivative bound.

Implements the quantities of paper Theorem 1.1 and its proof (Section 3):

  b = min_{x in supp(nu0), |e|=1} integral (e . (y - x))_+ nu1(dy)      (Sec. 3.1)
  L = 4 M (2R + sqrt(d)) / b
  m_R = b * exp(-(L+R)^2) / (2 L 2^{d/2})                               (Thm 1.1)

and Lemma 3.1's lower bound on the expected directional derivative of the
Brenier potential,

  E[e . grad v_Q(rho e + sigma Gamma)] - e.x >= b - M(2E|Q|+sqrt(d))/rho,

valid for rho > M(2E|Q|+sqrt(d))/b.
"""

from __future__ import annotations

import numpy as np

from src.bass_functional import BassProblem, brenier_map_for_Z
from src.optimal_transport import BrenierPotential


def unit_directions(d: int, n_directions: int, seed: int) -> np.ndarray:
    """n_directions points on the unit sphere S^{d-1} (d=1: {-1,+1})."""
    if d == 1:
        return np.array([[1.0], [-1.0]])
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_directions, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def compute_b(problem: BassProblem, n_directions: int, seed: int) -> float:
    """Monte Carlo estimate of b = min_{x in supp nu0, |e|=1} E_{nu1}[(e.(Y-x))_+].

    The expectation over nu1 is the weighted average over `problem.y_samples`;
    the minimization is over the finite grid of nu0 atoms and sampled unit
    directions (Section 3.1).
    """
    directions = unit_directions(problem.d, n_directions, seed)
    y = problem.y_samples
    w = problem.y_weights
    best = np.inf
    for x in problem.x_atoms:
        diff = y - x  # (n_y, d)
        proj = diff @ directions.T  # (n_y, n_directions)
        pos = np.clip(proj, 0.0, None)
        expectations = w @ pos  # (n_directions,)
        best = min(best, expectations.min())
    if not np.isfinite(best) or best <= 0:
        raise ValueError(f"computed b is not strictly positive: b={best}")
    return float(best)


def compute_L_and_mR(M: float, b: float, R: float, d: int) -> dict:
    """L and m_R from Theorem 1.1, given M, b, R, d.

    m_R = b * exp(-(L+R)^2) / (2 L 2^{d/2}) involves exp(-(L+R)^2) with
    (L+R)^2 of order 10^2-10^3 for the O(1)-scale distributions used here,
    which underflows float64 (exp(-x) is representable only for x < ~745).
    `log_m_R` evaluates the same quantity in log-space, so its sign and its
    monotonic decrease in R -- the qualitative content of Theorem 1.1 -- stay
    checkable even where the plain value `m_R` numerically underflows to 0.0.
    """
    L = 4.0 * M * (2.0 * R + np.sqrt(d)) / b
    log_m_R = np.log(b) - (L + R) ** 2 - np.log(2.0 * L) - (d / 2.0) * np.log(2.0)
    m_R = float(np.exp(log_m_R)) if log_m_R > -700 else 0.0
    return {"L": L, "m_R": m_R, "log_m_R": float(log_m_R)}


def lemma_3_1_bound(
    problem: BassProblem,
    Q: np.ndarray,
    gamma_for_Q: np.ndarray,
    x: np.ndarray,
    e: np.ndarray,
    rho: float,
    sigma: float,
    b: float,
    M: float,
    n_mc: int,
    seed: int,
) -> dict:
    """Monte Carlo check of the directional-derivative lower bound (Lemma 3.1).

    Fits v_Q (the Brenier potential for Z=Q), then estimates
    E[e . grad v_Q(rho*e + sigma*Gamma)] by averaging over `n_mc` fresh
    Gaussian draws, and compares with the analytic lower bound
    b - M(2E|Q|+sqrt(d))/rho, valid when rho exceeds the threshold
    M(2E|Q|+sqrt(d))/b.
    """
    brenier = brenier_map_for_Z(problem, Q, gamma_for_Q)
    d = problem.d
    rng = np.random.default_rng(seed)
    fresh_gamma = rng.normal(size=(n_mc, d))
    points = rho * e[None, :] + sigma * fresh_gamma
    grads = brenier.gradient(points)
    directional_derivative = grads @ e
    mc_estimate = float(directional_derivative.mean()) - float(e @ x)

    e_abs_Q = float(np.sum(problem.x_weights * np.linalg.norm(Q, axis=1)))  # E|Q| = sum_i w_i |Q_i|
    threshold = M * (2.0 * e_abs_Q + np.sqrt(d)) / b
    analytic_lower_bound = b - M * (2.0 * e_abs_Q + np.sqrt(d)) / rho

    return {
        "mc_estimate": mc_estimate,
        "analytic_lower_bound": analytic_lower_bound,
        "rho": rho,
        "rho_threshold": threshold,
        "rho_above_threshold": bool(rho > threshold),
        "e_abs_Q": e_abs_Q,
    }
