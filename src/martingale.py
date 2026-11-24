"""The Bass martingale built from one gradient-flow iterate (paper Section 5).

Given Z_t, the Brenier potential v_{Z_t} pushes the law of Z_t + Gamma to nu1
(Property 1.2). Replacing the single Gaussian shift Gamma by a standard
Brownian motion B and setting W_s = Z_t + B_s (for a fixed atom value Z_t(x))
defines the Bass martingale

    M_s = E[grad v_{Z_t}(W_1) | W_s],   s in [0, 1],

whose terminal law M_1 = grad v_{Z_t}(Z_t + B_1) is nu1 exactly (B_1 ~
N(0,I_d) has the same law as Gamma), and whose initial value M_0 = E[Y_{Z_t} |
X] is the conditional expectation appearing in g(Z_t) = E[Y_{Z_t}|X] - X
(paper Section 5, "Error in the initial value of the martingale").
"""

from __future__ import annotations

import numpy as np

from src.optimal_transport import BrenierPotential


def bass_martingale_paths(
    brenier: BrenierPotential,
    z_atom: np.ndarray,
    s_grid: np.ndarray,
    n_outer: int,
    n_inner: int,
    seed: int,
) -> dict:
    """Estimate M_s = E[grad v_{Z_t}(W_1) | W_s] along `n_outer` outer paths.

    For each outer path, an independent Brownian motion gives W_s = z_atom +
    B_s at every grid time. At s = 1, M_1 = grad v_{Z_t}(W_1) is exact (no
    conditioning needed). At s < 1, the conditional expectation is estimated
    by nested Monte Carlo: `n_inner` independent completions B_1 - B_s ~
    N(0, (1-s) I_d) are drawn and grad v_{Z_t}(W_s + (B_1-B_s)) is averaged.

    Returns
    -------
    dict with 'W_s' (n_outer, len(s_grid), d), 'M_s' (same shape), and
    'M_1_exact' (n_outer, d), the exact (un-averaged) terminal value.
    """
    s_grid = np.asarray(s_grid, dtype=float)
    if s_grid[0] != 0.0 or s_grid[-1] != 1.0 or np.any(np.diff(s_grid) <= 0):
        raise ValueError("s_grid must start at 0, end at 1, and be strictly increasing")
    d = z_atom.shape[0]
    rng = np.random.default_rng(seed)

    outer_increments = rng.normal(size=(n_outer, len(s_grid) - 1, d))
    sqrt_dt = np.sqrt(np.diff(s_grid))
    B_s = np.zeros((n_outer, len(s_grid), d))
    B_s[:, 1:, :] = np.cumsum(outer_increments * sqrt_dt[None, :, None], axis=1)
    W_s = z_atom[None, None, :] + B_s

    M_1_exact = brenier.gradient(W_s[:, -1, :])

    M_s = np.empty_like(W_s)
    for k, s in enumerate(s_grid):
        if s >= 1.0 - 1e-12:
            M_s[:, k, :] = M_1_exact
            continue
        residual_std = np.sqrt(1.0 - s)
        inner = rng.normal(scale=residual_std, size=(n_outer, n_inner, d))
        completions = W_s[:, k, None, :] + inner  # (n_outer, n_inner, d)
        flat = completions.reshape(n_outer * n_inner, d)
        grads = brenier.gradient(flat).reshape(n_outer, n_inner, d)
        M_s[:, k, :] = grads.mean(axis=1)

    return {"s_grid": s_grid, "W_s": W_s, "M_s": M_s, "M_1_exact": M_1_exact}
