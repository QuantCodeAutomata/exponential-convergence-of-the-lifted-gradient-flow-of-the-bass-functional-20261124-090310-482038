"""exp_5: error in the initial value of the Bass martingale (Section 5).

Builds, from a gradient-flow iterate Z_t (Theorem 1.2), the Bass martingale

    W_s = Z_t + B_s,    M_s = E[grad v_{Z_t}(W_1) | W_s],    s in [0, 1],

where B is a standard Brownian motion and v_{Z_t} is the Brenier potential
pushing the law of Z_t + Gamma to nu1 (Property 1.2). By construction the
terminal value M_1 = grad v_{Z_t}(Z_t + B_1) has law nu1 exactly, while the
initial value M_0 = E[Y_{Z_t} | X] differs from X by g(Z_t) = E[Y_{Z_t}|X] - X,
the gradient of the Bass functional at Z_t (Section 5, "Error in the initial
value of the martingale"). Combining the Lipschitz bound of Lemma 2.5
(||g(W)-g(Z)||_2 <= M||W-Z||_2) with g(Z*) = 0 at the minimizer and the
exponential convergence of Theorem 1.2 gives

    ||g(Z_t)||_2 = ||E[Y_{Z_t}|X] - X||_2 <= M * exp(-m*t) * ||Z_0 - Z*||_2.

This experiment reuses the same gradient flow as exp_4 (same seeds, same
Z_0, Z*, R, b, m_R) and checks, at the stopping times EXP5_EVAL_TIMES:

  * the decay of ||g(Z_t)||_2 along the flow and the (literal, and
    log-linear-fit) comparison with the bound above, exactly mirroring
    exp_4's treatment of the strong-convexity constant m_R (which underflows
    to 0.0 in float64 here, see src/strong_convexity.py);
  * that the terminal value M_1, reconstructed at a few representative
    atoms and times via nested Brownian-bridge Monte Carlo, lies in the
    compact support of nu1 and its empirical mean/std approach those of the
    nu1 sample;
  * that the initial value M_0 approaches the corresponding nu0 atom as t
    grows, with error decaying at the same rate as ||g(Z_t)||_2.

Run with `python -m exp.exp5` (reduced step count) or
`python -m exp.exp5 --full` (paper-scale step count, see src/config.py).
Writes tables and a figure to results/.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import sample_bass_pair
from src import config
from src.bass_functional import brenier_map_for_Z, l2_norm, make_bass_problem
from src.gradient_flow import fit_exponential_rate, run_gradient_flow
from src.martingale import bass_martingale_paths
from src.strong_convexity import compute_b, compute_L_and_mR

RESULTS_DIR = "results"


def _nearest_step(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


def run_for_dimension(d: int, n_steps: int, dt: float) -> dict:
    """Run the gradient flow of Theorem 1.2 and check the martingale error
    bound of Section 5 along it."""
    synth = sample_bass_pair(
        d=d, n_atoms=config.N_X_ATOMS, radius=config.SUPPORT_RADIUS, seed=config.BASE_SEED + d
    )
    problem = make_bass_problem(synth)
    Z0 = config.EXP4_Z0_SCALE * problem.x_atoms  # identical to exp_4's initial vector

    history = run_gradient_flow(
        problem, Z0, dt=dt, n_steps=n_steps, k_gamma=config.EXP4_FLOW_K_GAMMA,
        seed=config.BASE_SEED + 6000 + d,
    )
    Z_star = history.Z_path[-1]
    diff0 = l2_norm(Z0 - Z_star, problem.x_weights)

    R = float(history.linf_path.max())
    b = compute_b(problem, config.EXP3_B_DIRECTIONS, seed=config.BASE_SEED + 7000 + d)
    mR = compute_L_and_mR(M=problem.M, b=b, R=R, d=d)
    m, M = mR["m_R"], problem.M

    eval_times = [t for t in config.EXP5_EVAL_TIMES if t <= history.times[-1]]
    eval_steps = [_nearest_step(history.times, t) for t in eval_times]
    g_norm_at_eval = [float(history.g_norm_path[s]) for s in eval_steps]
    bound_at_eval = [M * np.exp(-m * t) * diff0 for t in eval_times]
    # literal_bound_holds is vacuous whenever m_R underflows to 0 (bound=M*diff0,
    # a constant): the right-hand side is then just the (non-increasing, by
    # Lemma 2.5 + Theorem 1.2) initial-gap constant, exactly as in exp_4.
    literal_bound_holds = [bool(g <= bd + 1e-9) for g, bd in zip(g_norm_at_eval, bound_at_eval)]

    rate_fit_g = fit_exponential_rate(history.times, history.g_norm_path, config.EXP4_FIT_SKIP_FRACTION)

    # Representative atoms and times for the Bass-martingale reconstruction
    # (nested Brownian-bridge Monte Carlo): three atoms spanning the support
    # of nu0, evaluated at the first, middle, and last feasible stopping time.
    n_x = problem.x_atoms.shape[0]
    atom_indices = sorted({0, n_x // 2, n_x - 1})
    martingale_times = sorted({eval_times[0], eval_times[len(eval_times) // 2], eval_times[-1]})
    s_grid = np.array(config.EXP5_BRIDGE_S_GRID)

    martingale_checks = []
    rng_seed_base = config.BASE_SEED + 8000 + d
    fresh_rng = np.random.default_rng(rng_seed_base)
    for t in martingale_times:
        step = _nearest_step(history.times, t)
        Z_t = history.Z_path[step]
        gamma = fresh_rng.normal(size=(n_x, config.K_GAMMA, d))
        brenier = brenier_map_for_Z(problem, Z_t, gamma)
        for atom_idx in atom_indices:
            z_atom = Z_t[atom_idx]
            x_atom = problem.x_atoms[atom_idx]
            paths = bass_martingale_paths(
                brenier, z_atom, s_grid,
                n_outer=config.EXP5_N_OUTER_PATHS, n_inner=config.EXP5_N_BRIDGE_PATHS,
                seed=rng_seed_base + 100 * step + atom_idx,
            )
            M1 = paths["M_1_exact"]  # (n_outer, d), exact draws from the reconstructed nu1 map
            M0 = paths["M_s"][:, 0, :].mean(axis=0)  # Monte Carlo estimate of E[Y_{Z_t}|X=z_atom]

            m1_norms = np.linalg.norm(M1, axis=1)
            terminal_in_support = bool(np.all(m1_norms <= M * (1.0 + 1e-6)))
            terminal_is_target_point = bool(
                np.all(np.any(np.all(np.isclose(M1[:, None, :], problem.y_samples[None, :, :]), axis=2), axis=1))
            )
            initial_error = float(np.linalg.norm(M0 - x_atom))

            martingale_checks.append({
                "t": t,
                "atom_idx": int(atom_idx),
                "x_atom": x_atom.tolist(),
                "z_atom": z_atom.tolist(),
                "M1_mean": M1.mean(axis=0).tolist(),
                "M1_max_norm": float(m1_norms.max()),
                "terminal_in_support": terminal_in_support,
                "terminal_is_target_point": terminal_is_target_point,
                "M0_mean": M0.tolist(),
                "initial_error_vs_x_atom": initial_error,
            })
    # population-level target-distribution sanity (mean/std of nu1 sample)
    nu1_mean = (problem.y_weights[:, None] * problem.y_samples).sum(axis=0)
    nu1_std = np.sqrt((problem.y_weights[:, None] * (problem.y_samples - nu1_mean) ** 2).sum(axis=0))

    return {
        "d": d,
        "n_steps": n_steps,
        "dt": dt,
        "R": R,
        "b": b,
        "M": M,
        "m_R": m,
        "log_m_R": mR["log_m_R"],
        "diff0": diff0,
        "eval_times": eval_times,
        "g_norm_at_eval": g_norm_at_eval,
        "bound_at_eval": bound_at_eval,
        "literal_bound_holds": literal_bound_holds,
        "rate_fit_g": rate_fit_g,
        "martingale_checks": martingale_checks,
        "nu1_mean": nu1_mean.tolist(),
        "nu1_std": nu1_std.tolist(),
        "times": history.times.tolist(),
        "g_norm_path": history.g_norm_path.tolist(),
    }


def plot_error(all_results: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for res in all_results:
        times = np.array(res["times"])
        g_norm = np.maximum(np.array(res["g_norm_path"]), 1e-14)
        ax.semilogy(times, g_norm, label=f"||g(Z_t)||_2, d={res['d']}")
        ax.semilogy(res["eval_times"], np.maximum(res["bound_at_eval"], 1e-14), "--",
                    label=f"M exp(-m t) ||Z0-Z*||_2, d={res['d']}")
    ax.set_xlabel("t")
    ax.set_ylabel("initial-value error of the Bass martingale")
    ax.set_title("exp_5: error in the initial value of the Bass martingale (Section 5)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper-scale step count EXP5_N_STEPS_PAPER instead of EXP5_N_STEPS_RUN")
    args = parser.parse_args()
    n_steps = config.EXP5_N_STEPS_PAPER if args.full else config.EXP5_N_STEPS_RUN

    all_results = [run_for_dimension(d, n_steps=n_steps, dt=config.EXP5_DT) for d in config.DIMENSIONS]

    summary_rows = []
    for res in all_results:
        errors_vs_x = [c["initial_error_vs_x_atom"] for c in res["martingale_checks"]]
        in_support = all(c["terminal_in_support"] for c in res["martingale_checks"])
        is_target = all(c["terminal_is_target_point"] for c in res["martingale_checks"])
        summary_rows.append({
            "d": res["d"],
            "n_steps": res["n_steps"],
            "R": res["R"],
            "b": res["b"],
            "m_R": res["m_R"],
            "log_m_R": res["log_m_R"],
            "fitted_rate_g": res["rate_fit_g"]["rate"],
            "literal_bound_holds_all": all(res["literal_bound_holds"]),
            "martingale_terminal_in_support_all": in_support,
            "martingale_terminal_is_target_point_all": is_target,
            "martingale_initial_error_min_max": [min(errors_vs_x), max(errors_vs_x)],
            "g_norm_first": res["g_norm_at_eval"][0],
            "g_norm_last": res["g_norm_at_eval"][-1],
        })

    with open(f"{RESULTS_DIR}/exp5_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(f"{RESULTS_DIR}/exp5_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plot_error(all_results, f"{RESULTS_DIR}/exp5_martingale_error.png")

    print(f"exp_5: error in the initial value of the Bass martingale (Section 5), n_steps={n_steps}, dt={config.EXP5_DT}")
    for row in summary_rows:
        print(
            f"  d={row['d']}: R={row['R']:.3f} b={row['b']:.4f} log_m_R={row['log_m_R']:.2f} "
            f"fitted_rate_g={row['fitted_rate_g']:.4f} literal_bound_holds={row['literal_bound_holds_all']} "
            f"terminal_in_support={row['martingale_terminal_in_support_all']} "
            f"terminal_is_target_point={row['martingale_terminal_is_target_point_all']} "
            f"initial_error(min,max)=({row['martingale_initial_error_min_max'][0]:.4f},"
            f"{row['martingale_initial_error_min_max'][1]:.4f}) "
            f"g_norm(first,last)=({row['g_norm_first']:.4f},{row['g_norm_last']:.4f})"
        )


if __name__ == "__main__":
    main()
