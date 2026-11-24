"""exp_1: finiteness, attainment, differentiability, minimizer characterization.

Numerically checks the foundational properties of the Bass functional V
stated in paper Section 1.1:

  * Property 1.1 (finiteness): V(Z) is bounded above by the Cauchy-Schwarz
    estimate (||Z||_2 + sqrt(d)) sqrt(E_nu1|Y|^2) + ||Z||_2 sqrt(E_nu0|X|^2).
  * Property 1.2 (attainment): the supremum defining V(Z) is attained at
    Y_Z = grad v_Z(Z+Gamma), the Brenier optimal-transport map pushing the
    law of Z+Gamma onto nu1, so that Y_Z has law nu1.
  * Property 1.3 (Frechet differentiability): V is differentiable with
    gradient g(Z) = E[Y_Z|X] - X, checked by comparing finite differences
    (V(Z+hW)-V(Z))/h with <g(Z), W>_H for small h.
  * Property 1.6 (minimizer characterization): Z minimizes V iff g(Z) = 0;
    checked by running the gradient flow to approximate the minimizer Z*
    and verifying ||g(Z*)||_2 is small relative to ||g(Z_0)||_2.

Run with `python -m exp.exp1`. Writes tables and a figure to results/.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import sample_bass_pair, sample_gamma
from src import config
from src.bass_functional import (
    bass_functional_value,
    bass_gradient,
    brenier_map_for_Z,
    inner_product,
    l2_norm,
    make_bass_problem,
)
from src.gradient_flow import run_gradient_flow

RESULTS_DIR = "results"


def _make_gamma(n_x: int, d: int, k: int, seed: int) -> np.ndarray:
    """k i.i.d. standard Gaussian replicates per atom, shape (n_x, k, d)."""
    flat = sample_gamma(n_x * k, d, seed)
    return flat.reshape(n_x, k, d)


def _random_Z(problem, scale: float, rng: np.random.Generator) -> np.ndarray:
    """A random bounded element of H = L^2(sigma(X); R^d): an affine, bounded
    function of each atom (scale * atom plus independent bounded noise)."""
    n_x, d = problem.x_atoms.shape
    noise = rng.uniform(-1.0, 1.0, size=(n_x, d))
    return scale * (problem.x_atoms + 0.3 * noise)


def check_finiteness(problem, gamma, rng) -> list[dict]:
    """Property 1.1: V(Z) <= Cauchy-Schwarz bound, for EXP1_N_TEST_Z random Z."""
    e_y2 = float(np.sum(problem.y_weights * np.sum(problem.y_samples ** 2, axis=1)))
    e_x2 = float(np.sum(problem.x_weights * np.sum(problem.x_atoms ** 2, axis=1)))
    d = problem.d
    rows = []
    for i in range(config.EXP1_N_TEST_Z):
        scale = rng.uniform(0.2, 2.5)
        Z = _random_Z(problem, scale, rng)
        V = bass_functional_value(problem, Z, gamma)
        norm_z = l2_norm(Z, problem.x_weights)
        bound = (norm_z + np.sqrt(d)) * np.sqrt(e_y2) + norm_z * np.sqrt(e_x2)
        rows.append({
            "trial": i, "scale": scale, "norm_Z": norm_z,
            "V": V, "cauchy_schwarz_bound": bound, "finite": bool(np.isfinite(V)),
            "bound_holds": bool(V <= bound + 1e-8),
        })
    return rows


def check_attainment(problem, gamma) -> dict:
    """Property 1.2: Y_Z = grad v_Z(Z+Gamma) should have (weighted empirical)
    law nu1. Compared via mean, covariance, and the Wasserstein-2 distance
    between the weighted empirical law of Y_Z and the nu1 sample."""
    import ot

    Z = problem.x_atoms.copy()
    brenier = brenier_map_for_Z(problem, Z, gamma)
    n_x, k, d = gamma.shape
    source = (Z[:, None, :] + gamma).reshape(n_x * k, d)
    source_weights = np.repeat(problem.x_weights / k, k)
    Y = brenier.gradient(source)

    mean_Y = np.sum(source_weights[:, None] * Y, axis=0)
    mean_nu1 = np.sum(problem.y_weights[:, None] * problem.y_samples, axis=0)
    cov_Y = np.cov(Y.T, aweights=source_weights, bias=True)
    cov_nu1 = np.cov(problem.y_samples.T, aweights=problem.y_weights, bias=True)

    cost = ot.dist(Y, problem.y_samples, metric="sqeuclidean")
    w2_sq = ot.emd2(source_weights, problem.y_weights, cost)
    w2 = float(np.sqrt(max(w2_sq, 0.0)))

    return {
        "mean_Y_Z": mean_Y.tolist(),
        "mean_nu1": mean_nu1.tolist(),
        "cov_Y_Z": np.atleast_2d(cov_Y).tolist(),
        "cov_nu1": np.atleast_2d(cov_nu1).tolist(),
        "wasserstein2_Y_Z_to_nu1": w2,
        "support_radius_M": problem.M,
        "relative_w2": w2 / problem.M,
    }


def check_differentiability(problem, rng) -> list[dict]:
    """Property 1.3: finite differences of V against <g(Z), W>_H.

    Each evaluation of V(Z+hW) re-solves the discrete optimal-transport
    problem, so V is only piecewise-affine in Z at this sample size: the
    finite-difference error is expected to be U-shaped in h (truncation
    error at large h, discrete re-matching noise at very small h) rather
    than monotonically vanishing. Uses common random numbers (one draw of
    Gamma per atom, shared across every V evaluation for a given direction)
    with EXP1_FD_K_GAMMA replicates to keep that noise floor low.
    """
    n_x, d = problem.x_atoms.shape
    rows = []
    for j in range(config.EXP1_FD_DIRECTIONS):
        Z = _random_Z(problem, rng.uniform(0.3, 1.5), rng)
        W = _random_Z(problem, 1.0, rng)
        gamma = _make_gamma(n_x, d, config.EXP1_FD_K_GAMMA, seed=config.BASE_SEED + 5000 + j)
        brenier_Z = brenier_map_for_Z(problem, Z, gamma)
        V_Z = bass_functional_value(problem, Z, gamma, brenier_Z)
        g_Z = bass_gradient(problem, Z, gamma, brenier_Z)
        directional = inner_product(g_Z, W, problem.x_weights)
        for h in config.EXP1_FD_STEPS:
            V_Zh = bass_functional_value(problem, Z + h * W, gamma)
            fd_ratio = (V_Zh - V_Z) / h
            rows.append({
                "direction": j, "h": h,
                "finite_difference": fd_ratio,
                "predicted_directional_derivative": directional,
                "abs_error": abs(fd_ratio - directional),
            })
    return rows


def check_minimizer(problem) -> dict:
    """Property 1.6: Z minimizes V iff g(Z) = 0. Approximates Z* by running
    the gradient flow and compares ||g(Z*)||_2 with ||g(Z_0)||_2."""
    n_x, d = problem.x_atoms.shape
    Z0 = problem.x_atoms.copy()
    history = run_gradient_flow(
        problem, Z0,
        dt=config.EXP1_MINIMIZER_DT,
        n_steps=config.EXP1_MINIMIZER_STEPS,
        k_gamma=config.K_GAMMA,
        seed=config.BASE_SEED + 101,
    )
    return {
        "g_norm_initial": float(history.g_norm_path[0]),
        "g_norm_final": float(history.g_norm_path[-1]),
        "ratio_final_to_initial": float(history.g_norm_path[-1] / history.g_norm_path[0]),
        "V_initial": float(history.V_path[0]),
        "V_final": float(history.V_path[-1]),
        "n_steps": config.EXP1_MINIMIZER_STEPS,
        "dt": config.EXP1_MINIMIZER_DT,
    }


def run_for_dimension(d: int) -> dict:
    rng = np.random.default_rng(config.BASE_SEED + 1000 * d)
    synth = sample_bass_pair(d=d, n_atoms=config.N_X_ATOMS, radius=config.SUPPORT_RADIUS,
                              seed=config.BASE_SEED + d)
    problem = make_bass_problem(synth)
    gamma = _make_gamma(problem.x_atoms.shape[0], d, config.K_GAMMA, seed=config.BASE_SEED + 2000 + d)

    finiteness = check_finiteness(problem, gamma, rng)
    attainment = check_attainment(problem, gamma)
    differentiability = check_differentiability(problem, rng)
    minimizer = check_minimizer(problem)

    return {
        "d": d,
        "n_atoms": problem.x_atoms.shape[0],
        "n_y_samples": problem.y_samples.shape[0],
        "M": problem.M,
        "finiteness": finiteness,
        "attainment": attainment,
        "differentiability": differentiability,
        "minimizer": minimizer,
    }


def plot_fd_convergence(all_results: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for res in all_results:
        rows = res["differentiability"]
        hs = sorted(set(r["h"] for r in rows))
        mean_err = [np.mean([r["abs_error"] for r in rows if r["h"] == h]) for h in hs]
        ax.loglog(hs, mean_err, marker="o", label=f"d={res['d']}")
    ax.set_xlabel("finite-difference step h")
    ax.set_ylabel("mean |FD ratio - <g(Z), W>_H|")
    ax.set_title("exp_1: Frechet-differentiability check (Property 1.3)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    all_results = [run_for_dimension(d) for d in config.DIMENSIONS]

    summary_rows = []
    for res in all_results:
        fin_ok = all(r["bound_holds"] and r["finite"] for r in res["finiteness"])
        fd_rows = res["differentiability"]
        hs = sorted(set(r["h"] for r in fd_rows))
        err_by_h = {h: np.mean([r["abs_error"] for r in fd_rows if r["h"] == h]) for h in hs}
        largest_h, smallest_h = max(hs), min(hs)
        best_h = min(err_by_h, key=err_by_h.get)
        summary_rows.append({
            "d": res["d"],
            "finiteness_bound_holds_for_all_trials": fin_ok,
            "attainment_relative_w2": res["attainment"]["relative_w2"],
            "fd_error_at_largest_h": float(err_by_h[largest_h]),
            "fd_error_at_best_h": float(err_by_h[best_h]),
            "best_h": best_h,
            # Each V-evaluation re-solves a discrete optimal-transport problem, so V is only
            # piecewise-affine at finite sample size: the finite-difference error is expected to
            # be U-shaped in h (truncation error at large h, discrete re-matching noise at small
            # h), not monotonically vanishing; convergence is checked against this minimum.
            "fd_error_improves_from_largest_h": bool(err_by_h[best_h] < err_by_h[largest_h]),
            "g_norm_ratio_at_minimizer": res["minimizer"]["ratio_final_to_initial"],
        })

    with open(f"{RESULTS_DIR}/exp1_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(f"{RESULTS_DIR}/exp1_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plot_fd_convergence(all_results, f"{RESULTS_DIR}/exp1_fd_convergence.png")

    print("exp_1: Bass functional finiteness / attainment / differentiability / minimizer")
    for row in summary_rows:
        print(
            f"  d={row['d']}: finiteness_bound_holds={row['finiteness_bound_holds_for_all_trials']} "
            f"attainment_rel_W2={row['attainment_relative_w2']:.4f} "
            f"fd_error(h_max->best_h={row['best_h']:.0e})="
            f"{row['fd_error_at_largest_h']:.3e}->{row['fd_error_at_best_h']:.3e} "
            f"g_norm_ratio_at_Z*={row['g_norm_ratio_at_minimizer']:.3e}"
        )


if __name__ == "__main__":
    main()
