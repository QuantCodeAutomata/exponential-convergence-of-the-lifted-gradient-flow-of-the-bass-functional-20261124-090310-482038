"""exp_3: strong convexity of V (Theorem 1.1) and the directional-derivative
lower bound (Lemma 3.1).

Numerically checks:

  * Section 3.1's constant b = min_{x in supp(nu0), |e|=1} E_nu1[(e.(Y-x))_+]
    (via `src.strong_convexity.compute_b`), reported strictly positive.
  * Theorem 1.1: V is strongly convex with modulus m_R on the L^infinity ball
    of radius R intersected with a fixed weighted mean, i.e. for Z, W in H
    with ||Z||_inf, ||W||_inf <= R and E[Z] = E[W]:

        V(W) >= V(Z) + <g(Z), W-Z>_H + (m_R/2)||W-Z||_2^2,

    where L = 4M(2R+sqrt(d))/b and m_R = b*exp(-(L+R)^2)/(2L*2^(d/2)).
    As documented in `src.strong_convexity.compute_L_and_mR`, m_R underflows
    to exactly 0.0 in float64 at the O(1) scales used here; `log_m_R` stays
    finite and lets us check the qualitative content of the theorem (m_R > 0
    in principle, strictly decreasing in R) even when the plain value is 0.
    The residual check below is accordingly only numerically meaningful as a
    plain-convexity check whenever m_R == 0.0 exactly.
  * Lemma 3.1 (a proof ingredient for Theorem 1.1): for Q in H, sigma in
    (0,1], x in supp(nu0), unit vector e, and rho above the threshold
    M(2E|Q|+sqrt(d))/b,

        E[e.grad v_Q(rho*e + sigma*Gamma)] - e.x >= b - M(2E|Q|+sqrt(d))/rho > 0.

Run with `python -m exp.exp3`. Writes tables and a figure to results/.
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
    BassProblem,
    bass_functional_value,
    bass_gradient,
    brenier_map_for_Z,
    inner_product,
    l2_norm,
    make_bass_problem,
)
from src.strong_convexity import compute_b, compute_L_and_mR, lemma_3_1_bound, unit_directions

RESULTS_DIR = "results"

# Relative tolerance for the strong-convexity residual check: V(Z), V(W) and
# g(Z) are each only a K_GAMMA-replicate Monte Carlo / finite-OT estimate
# (same noise floor discussed in exp2.py's TOL_CONVEXITY_REL), so a small
# relative slack is allowed for that estimation noise rather than for the
# underlying inequality.
TOL_SC_REL = 0.02

# Lemma 3.1's lower bound is checked up to a Monte Carlo tolerance: the
# directional derivative e.grad v_Q is bounded in absolute value by M (it is
# e.y for some sampled y in supp(nu1)), so by a CLT-type argument its sample
# mean over n_mc i.i.d. draws has standard error of order M/sqrt(n_mc); 5
# such standard errors gives a two-sided band with negligible false-rejection
# rate. A tiny absolute floor guards against the degenerate M=0 case.
def _lemma31_tol(M: float, n_mc: int) -> float:
    return max(5.0 * M / np.sqrt(n_mc), 1e-6)


def _make_gamma(n_x: int, d: int, k: int, seed: int) -> np.ndarray:
    """k i.i.d. standard Gaussian replicates per atom, shape (n_x, k, d)."""
    flat = sample_gamma(n_x * k, d, seed)
    return flat.reshape(n_x, k, d)


def _random_Z(problem: BassProblem, scale: float, rng: np.random.Generator) -> np.ndarray:
    """A random bounded element of H = L^2(sigma(X); R^d): an affine, bounded
    function of each atom (scale * atom plus independent bounded noise)."""
    n_x, d = problem.x_atoms.shape
    noise = rng.uniform(-1.0, 1.0, size=(n_x, d))
    return scale * (problem.x_atoms + 0.3 * noise)


def _make_mean_matched_pair(
    problem: BassProblem,
    R: float,
    rng: np.random.Generator,
    max_tries: int = 50,
    mean_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """A random pair (Z, W) in H with ||Z||_inf, ||W||_inf <= R and equal
    weighted means E[Z] = E[W], for the strong-convexity check of Theorem 1.1.

    The paper does not specify a sampling scheme for such pairs, so this is
    our own ad hoc construction: start from a random bounded Z clipped to
    [-R, R]; perturb it by a random `delta` whose own weighted mean is first
    removed (so adding it to Z cannot change E[Z]); clip the result to
    [-R, R] (clipping can reintroduce a small mean mismatch whenever some
    coordinates saturate at +-R); then repeatedly shift by the residual mean
    difference and clip again -- a fixed-point iteration that converges
    quickly unless a large fraction of coordinates are saturated. If it has
    not converged to within `mean_tol` after a handful of shift-clip
    iterations, the whole pair is redrawn (fresh draws from `rng`) up to
    `max_tries` times; Z's scale and delta's magnitude are kept well inside
    [-R, R] (a 0.6*R margin) so saturation, and hence redraws, are rare.
    """
    n_x, d = problem.x_atoms.shape
    x_weights = problem.x_weights
    for _ in range(max_tries):
        scale = rng.uniform(0.1, 0.4) * R
        Z = np.clip(_random_Z(problem, scale, rng), -R, R)
        mean_Z = np.sum(x_weights[:, None] * Z, axis=0)

        delta = 0.4 * R * rng.uniform(-1.0, 1.0, size=(n_x, d))
        delta -= np.sum(x_weights[:, None] * delta, axis=0)  # zero weighted mean
        W = np.clip(Z + delta, -R, R)

        for _ in range(20):
            mean_diff = np.sum(x_weights[:, None] * W, axis=0) - mean_Z
            if np.linalg.norm(mean_diff) < mean_tol:
                return Z, W
            W = np.clip(W - mean_diff, -R, R)
    raise RuntimeError(
        f"failed to construct a mean-matched pair within tol={mean_tol} after {max_tries} tries"
    )


def strong_convexity_residual(
    problem: BassProblem,
    Z: np.ndarray,
    W: np.ndarray,
    gamma: np.ndarray,
    m_R: float,
) -> dict:
    """Theorem 1.1's residual V(W) - V(Z) - <g(Z),W-Z>_H - (m_R/2)||W-Z||_2^2,
    which should be >= 0 (up to the estimation-noise tolerance TOL_SC_REL).

    Also reports the "empirical modulus" 2*(V(W)-V(Z)-<g(Z),W-Z>_H)/||W-Z||_2^2
    (None if ||W-Z||_2 ~ 0), a numerically observable quantity to compare in
    order-of-magnitude terms with m_R (meaningful even when m_R underflows to
    0.0 and the residual check reduces to plain convexity of V).

    `gamma` is shared between Z and W (common random numbers), as in
    exp2.py's g_lipschitz_ratio / f_convexity_residual, to keep the
    discrete-OT re-matching noise in V(W)-V(Z) and g(Z) as small as possible.
    """
    brenier_Z = brenier_map_for_Z(problem, Z, gamma)
    V_Z = bass_functional_value(problem, Z, gamma, brenier_Z)
    g_Z = bass_gradient(problem, Z, gamma, brenier_Z)
    V_W = bass_functional_value(problem, W, gamma)

    linear_term = inner_product(g_Z, W - Z, problem.x_weights)
    diff_norm_sq = l2_norm(W - Z, problem.x_weights) ** 2
    convexity_gap = V_W - V_Z - linear_term
    residual = convexity_gap - 0.5 * m_R * diff_norm_sq
    empirical_modulus = (2.0 * convexity_gap / diff_norm_sq) if diff_norm_sq > 1e-12 else None

    scale = max(abs(V_W), abs(V_Z), abs(linear_term), 1.0)
    tol = TOL_SC_REL * scale
    return {
        "V_Z": V_Z,
        "V_W": V_W,
        "linear_term": linear_term,
        "diff_norm_sq": diff_norm_sq,
        "convexity_gap": convexity_gap,
        "m_R": m_R,
        "residual": residual,
        "empirical_modulus": empirical_modulus,
        "holds": bool(residual >= -tol),
    }


def run_lemma_3_1_checks(
    problem: BassProblem,
    b: float,
    M: float,
    seed: int,
    n_mc: int = config.EXP3_LEMMA31_N_MC,
    max_atoms: int = 8,
    max_directions: int = 8,
) -> list[dict]:
    """Lemma 3.1's directional-derivative check over a subset of (x, e)
    combinations, for one fixed random Q.

    Lemma 3.1's statement does not depend on R, so this is run once per
    dimension and the resulting rows are reused across every R entry in
    `run_for_dimension` (each `lemma_3_1_bound` call re-solves a discrete OT
    problem; repeating all combinations per R would be purely redundant
    work). `rho` is fixed above the Lemma 3.1 threshold by
    `config.EXP3_LEMMA31_RHO_FACTOR`, computed from the same E|Q| used
    internally by `lemma_3_1_bound`, so `rho_above_threshold` holds by
    construction for every combination below.

    Only `max_atoms` atoms (the first `max_atoms` of `problem.x_atoms`,
    deterministic for reproducibility) and `max_directions` unit directions
    (the first `max_directions` of `unit_directions(..., config.EXP3_B_DIRECTIONS, ...)`)
    are used, to keep the number of OT re-solves (one per combination)
    bounded; not specified in the paper.
    """
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(seed)
    Q = _random_Z(problem, scale=0.3, rng=rng)
    gamma_for_Q = _make_gamma(n_x, d, config.K_GAMMA, seed=seed + 1)

    e_abs_Q = float(np.sum(problem.x_weights * np.linalg.norm(Q, axis=1)))
    rho = config.EXP3_LEMMA31_RHO_FACTOR * M * (2.0 * e_abs_Q + np.sqrt(d)) / b
    sigma = config.EXP3_LEMMA31_SIGMA

    n_atom_subset = min(max_atoms, n_x)
    atom_indices = np.arange(n_atom_subset)

    directions = unit_directions(d, config.EXP3_B_DIRECTIONS, seed=seed + 2)
    n_dir_subset = min(max_directions, directions.shape[0])
    directions_subset = directions[:n_dir_subset]

    tol = _lemma31_tol(M, n_mc)

    rows: list[dict] = []
    for i in atom_indices:
        x = problem.x_atoms[i]
        for j, e in enumerate(directions_subset):
            result = lemma_3_1_bound(
                problem, Q, gamma_for_Q, x, e, rho, sigma, b, M, n_mc,
                seed=seed + 1000 + int(i) * 100 + j,
            )
            rows.append({
                "atom_index": int(i),
                "direction_index": j,
                "mc_estimate": result["mc_estimate"],
                "analytic_lower_bound": result["analytic_lower_bound"],
                "rho": result["rho"],
                "rho_threshold": result["rho_threshold"],
                "rho_above_threshold": bool(result["rho_above_threshold"]),
                "holds": bool(result["mc_estimate"] >= result["analytic_lower_bound"] - tol),
            })
    return rows


def run_for_dimension(d: int) -> dict:
    # Seed offsets are our own choice (distinct from exp1/exp2's offsets, to
    # avoid colliding with other experiments' seeds): +0 for the synthetic
    # pair, +5000 for b, +7000 for the Lemma 3.1 check, +8000/+9000 for the
    # per-R pair / Gamma draws below.
    synth = sample_bass_pair(
        d=d, n_atoms=config.N_X_ATOMS, radius=config.SUPPORT_RADIUS,
        seed=config.BASE_SEED + d,
    )
    problem = make_bass_problem(synth)
    n_x = problem.x_atoms.shape[0]
    M = problem.M

    b = compute_b(problem, config.EXP3_B_DIRECTIONS, seed=config.BASE_SEED + 5000 + d)
    assert b > 0.0, f"computed b must be strictly positive, got {b}"

    lemma31_rows = run_lemma_3_1_checks(problem, b, M, seed=config.BASE_SEED + 7000 + d)
    lemma31_holds_all = all(r["holds"] for r in lemma31_rows)
    rho_above_threshold_all = all(r["rho_above_threshold"] for r in lemma31_rows)

    r_results = []
    for R in sorted(config.EXP3_R_VALUES):
        mR_info = compute_L_and_mR(M, b, R, d)
        r_tag = int(round(R * 100))
        pair_rng = np.random.default_rng(config.BASE_SEED + 8000 + d * 1000 + r_tag)

        sc_rows = []
        for p in range(config.EXP3_N_SC_PAIRS):
            Z, W = _make_mean_matched_pair(problem, R, pair_rng)
            gamma = _make_gamma(
                n_x, d, config.K_GAMMA,
                seed=config.BASE_SEED + 9000 + d * 10000 + r_tag * 100 + p,
            )
            residual_info = strong_convexity_residual(problem, Z, W, gamma, mR_info["m_R"])
            sc_rows.append({"pair": p, **residual_info})

        residuals = [row["residual"] for row in sc_rows]
        r_results.append({
            "R": R,
            "L": mR_info["L"],
            "m_R": mR_info["m_R"],
            "log_m_R": mR_info["log_m_R"],
            "sc_rows": sc_rows,
            "sc_min_residual": float(min(residuals)),
            "sc_mean_residual": float(np.mean(residuals)),
            "sc_holds_all": bool(all(row["holds"] for row in sc_rows)),
            "lemma31_rows": lemma31_rows,
            "lemma31_holds_all": bool(lemma31_holds_all),
            "rho_above_threshold_all": bool(rho_above_threshold_all),
        })

    log_m_Rs = [row["log_m_R"] for row in r_results]
    log_m_R_strictly_decreasing = bool(
        all(log_m_Rs[i] > log_m_Rs[i + 1] for i in range(len(log_m_Rs) - 1))
    )

    return {
        "d": d,
        "n_atoms": n_x,
        "M": M,
        "b": b,
        "R_results": r_results,
        "log_m_R_strictly_decreasing": log_m_R_strictly_decreasing,
    }


def plot_log_mR_vs_R(all_results: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for res in all_results:
        Rs = [r["R"] for r in res["R_results"]]
        log_mRs = [r["log_m_R"] for r in res["R_results"]]
        ax.plot(Rs, log_mRs, marker="o", label=f"d={res['d']}")
    ax.set_xlabel("R (L^infinity bound)")
    ax.set_ylabel("log m_R")
    ax.set_title("exp_3: strong-convexity modulus log(m_R) vs R (Theorem 1.1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    all_results = [run_for_dimension(d) for d in config.DIMENSIONS]

    summary_rows = []
    for res in all_results:
        r_summary = [
            {
                "R": r["R"],
                "L": r["L"],
                "m_R": r["m_R"],
                "log_m_R": r["log_m_R"],
                "sc_min_residual": r["sc_min_residual"],
                "sc_mean_residual": r["sc_mean_residual"],
                "sc_holds_all": r["sc_holds_all"],
                "lemma31_holds_all": r["lemma31_holds_all"],
                "rho_above_threshold_all": r["rho_above_threshold_all"],
            }
            for r in res["R_results"]
        ]
        summary_rows.append({
            "d": res["d"],
            "b": res["b"],
            "M": res["M"],
            "R_summary": r_summary,
            "log_m_R_strictly_decreasing": res["log_m_R_strictly_decreasing"],
        })

    with open(f"{RESULTS_DIR}/exp3_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(f"{RESULTS_DIR}/exp3_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plot_log_mR_vs_R(all_results, f"{RESULTS_DIR}/exp3_m_R_vs_R.png")

    print("exp_3: strong convexity (Theorem 1.1) and directional-derivative bound (Lemma 3.1)")
    for row in summary_rows:
        print(
            f"  d={row['d']}: b={row['b']:.4f} "
            f"log_m_R_strictly_decreasing={row['log_m_R_strictly_decreasing']}"
        )
        for r in row["R_summary"]:
            print(
                f"    R={r['R']:.2f}: L={r['L']:.3f} m_R={r['m_R']:.3e} "
                f"log_m_R={r['log_m_R']:.3f} "
                f"sc_residual(min/mean)={r['sc_min_residual']:.3e}/{r['sc_mean_residual']:.3e} "
                f"sc_holds_all={r['sc_holds_all']} "
                f"lemma31_holds_all={r['lemma31_holds_all']} "
                f"rho_above_threshold_all={r['rho_above_threshold_all']}"
            )


if __name__ == "__main__":
    main()
