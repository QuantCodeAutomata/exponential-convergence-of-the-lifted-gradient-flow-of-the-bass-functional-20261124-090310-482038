"""Euler discretization of the gradient flow dZ_t/dt = -g(Z_t) (Theorem 1.2).

At each step, g(Z_t) = E[Y_{Z_t}|X] - X is recomputed by refitting the
Brenier potential v_{Z_t} from fresh common-random-number Gaussian replicates
(the same replicate array is reused within one step for the value V(Z_t) and
the gradient g(Z_t), but freshly redrawn at the next step, exactly like
resampling Gamma along the flow).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bass_functional import (
    BassProblem,
    bass_functional_value,
    bass_gradient,
    brenier_map_for_Z,
    inner_product,
    l2_norm,
)


@dataclass
class FlowHistory:
    times: np.ndarray
    Z_path: np.ndarray          # (n_steps+1, n_x, d)
    V_path: np.ndarray          # (n_steps+1,)
    g_norm_path: np.ndarray     # (n_steps+1,)
    mean_path: np.ndarray       # (n_steps+1, d), E[Z_t]
    linf_path: np.ndarray       # (n_steps+1,), ||Z_t||_inf


def run_gradient_flow(
    problem: BassProblem,
    Z0: np.ndarray,
    dt: float,
    n_steps: int,
    k_gamma: int,
    seed: int,
) -> FlowHistory:
    """Explicit Euler integration of dZ_t/dt = -g(Z_t) = -(E[Y_Z|X] - X).

    One fresh set of K=k_gamma Gaussian replicates per atom is drawn at every
    step to estimate g(Z_t); the step itself is a forward (explicit) Euler
    update Z_{t+dt} = Z_t - dt * g(Z_t).
    """
    n_x, d = problem.x_atoms.shape
    Z = Z0.copy()
    rng = np.random.default_rng(seed)

    times = dt * np.arange(n_steps + 1)
    Z_path = np.empty((n_steps + 1, n_x, d))
    V_path = np.empty(n_steps + 1)
    g_norm_path = np.empty(n_steps + 1)
    mean_path = np.empty((n_steps + 1, d))
    linf_path = np.empty(n_steps + 1)

    for step in range(n_steps + 1):
        gamma = rng.normal(size=(n_x, k_gamma, d))
        brenier = brenier_map_for_Z(problem, Z, gamma)
        V = bass_functional_value(problem, Z, gamma, brenier)
        g = bass_gradient(problem, Z, gamma, brenier)

        Z_path[step] = Z
        V_path[step] = V
        g_norm_path[step] = l2_norm(g, problem.x_weights)
        mean_path[step] = np.sum(problem.x_weights[:, None] * Z, axis=0)
        linf_path[step] = np.max(np.abs(Z))

        if step < n_steps:
            Z = Z - dt * g

    return FlowHistory(
        times=times,
        Z_path=Z_path,
        V_path=V_path,
        g_norm_path=g_norm_path,
        mean_path=mean_path,
        linf_path=linf_path,
    )


def fit_exponential_rate(times: np.ndarray, values: np.ndarray, skip_fraction: float) -> dict:
    """Fit log(values) ~ a - rate*t by least squares, after discarding an
    initial `skip_fraction` of points (transient removal), and after
    discarding non-positive values (floating-point noise near convergence).
    Returns the fitted decay rate (positive means decay) and intercept.
    """
    n = len(times)
    skip = int(n * skip_fraction)
    t = times[skip:]
    v = values[skip:]
    mask = v > 1e-14
    t, v = t[mask], v[mask]
    if len(t) < 2:
        raise ValueError("not enough positive samples to fit an exponential rate")
    log_v = np.log(v)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, log_v, rcond=None)[0]
    return {"rate": float(-slope), "intercept": float(intercept), "n_points": int(len(t))}
