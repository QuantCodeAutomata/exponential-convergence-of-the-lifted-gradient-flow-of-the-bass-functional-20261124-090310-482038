"""Euler discretization of the gradient flow dZ_t/dt = -g(Z_t) (Theorem 1.2).

A single batch of Gaussian replicates Gamma is drawn once, before integration
starts, and reused at every Euler step (common random numbers): at each step,
g(Z_t) = E[Y_{Z_t}|X] - X is obtained by refitting the Brenier potential
v_{Z_t} against that same fixed replicate array, which is also reused to
evaluate V(Z_t). Reusing the replicates keeps g(Z_t) a deterministic,
Lipschitz (Lemma 2.5) function of Z_t throughout the flow, as required for
the ODE dZ_t/dt = -g(Z_t) to be deterministic; see `run_gradient_flow`.
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

    dZ_t/dt = -g(Z_t) is a deterministic ODE (Section 1.2): g is a fixed
    functional of Z, not a stochastic one. One set of K=k_gamma Gaussian
    replicates per atom is drawn once, before integration starts, and reused
    at every step (common random numbers), so that g(Z_t) is the same
    deterministic, Lipschitz (Lemma 2.5) function of Z throughout the flow;
    redrawing Gamma at every step would instead add fresh i.i.d. Monte Carlo
    noise of order 1/sqrt(k_gamma) to every step, which for small k_gamma
    swamps the true gradient and turns the deterministic descent into a
    noisy walk that neither V nor ||g|| decreases along. The step itself is
    a forward (explicit) Euler update Z_{t+dt} = Z_t - dt * g(Z_t).
    """
    n_x, d = problem.x_atoms.shape
    Z = Z0.copy()
    rng = np.random.default_rng(seed)
    gamma = rng.normal(size=(n_x, k_gamma, d))

    times = dt * np.arange(n_steps + 1)
    Z_path = np.empty((n_steps + 1, n_x, d))
    V_path = np.empty(n_steps + 1)
    g_norm_path = np.empty(n_steps + 1)
    mean_path = np.empty((n_steps + 1, d))
    linf_path = np.empty(n_steps + 1)

    for step in range(n_steps + 1):
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


def fit_exponential_rate(times: np.ndarray, values: np.ndarray, skip_fraction: float,
                          noise_floor: float | None = None) -> dict:
    """Fit log(values) ~ a - rate*t by least squares, after discarding an
    initial `skip_fraction` of points (transient removal), and after
    discarding non-positive values (floating-point noise near convergence).

    `V(Z_t)` is estimated from the discretized (finite-N) Bass problem, so
    `V(Z_t) - V(Z*)` has an irreducible discretization floor (the empirical
    optimal-transport value only approximates the population value up to a
    finite-sample bias) that does not shrink as `t` grows, unlike the true
    population gap. Once the signal drops to that floor the residual
    fluctuates near zero or goes slightly negative and no longer reflects the
    exponential rate, so when `noise_floor` is given the fit keeps only the
    leading, still-decaying run of points with `values > noise_floor`.
    Returns the fitted decay rate (positive means decay) and intercept.
    """
    n = len(times)
    skip = int(n * skip_fraction)
    t_full, v_full = times[skip:], values[skip:]
    t, v = t_full, v_full
    if noise_floor is not None:
        above = v_full > noise_floor
        cutoff = int(np.argmax(~above)) if not above.all() else len(above)
        t, v = t_full[:cutoff], v_full[:cutoff]
        # If decay already fell below the floor within (or right at) the
        # transient-removal skip, the floor-truncated prefix can be too short
        # (or empty) to fit even though real decay did occur earlier in the
        # run: fall back to the full post-skip window and let the v>1e-14
        # mask below discard the flat/negative floor noise directly.
        if np.count_nonzero(v > 1e-14) < 2:
            t, v = t_full, v_full
    mask = v > 1e-14
    t, v = t[mask], v[mask]
    if len(t) < 2:
        raise ValueError("not enough positive samples to fit an exponential rate")
    log_v = np.log(v)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, log_v, rcond=None)[0]
    return {"rate": float(-slope), "intercept": float(intercept), "n_points": int(len(t))}
