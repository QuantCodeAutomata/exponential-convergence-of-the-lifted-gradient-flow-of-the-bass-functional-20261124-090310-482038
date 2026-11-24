"""exp_2: Lipschitz continuity of g and convexity of F.

Numerically checks:

  * Lemma 2.3: a generic fact about convex K-Lipschitz functions f: R^d -> R
    under an independent standard Gaussian shift Gamma -- for any x, h in
    R^d, 0 <= E[Delta_h f(x+Gamma)] <= (K/2)|h|^2, where
    Delta_h f(x) = f(x+h) - f(x) - grad_f(x).h (any subgradient at points of
    non-differentiability). Checked directly on two standalone convex
    1-Lipschitz test functions (f(x)=|x| and f(x)=max_i x_i), with no
    dependence on the Bass-functional machinery.
  * Lemma 2.4: the Brenier-potential identity
    V(Z) = E[v_Z(Z+Gamma)] + E_nu1[v_Z*] - E[Z.X], and the exact M-Lipschitz
    continuity of v_Z (immediate from v_Z being a pointwise max of affine
    maps with slopes y_j, |y_j| <= M).
  * Lemma 2.5: g(Z) = E[Y_Z|X] - X is M-Lipschitz: ||g(W)-g(Z)||_2 <= M
    ||W-Z||_2 for random pairs Z, W in H.
  * Baillon-Haddad / convexity of F(Z) = (M/2)||Z||_2^2 - V(Z): the
    subgradient inequality F(W) >= F(Z) + <grad F(Z), W-Z>_H, equivalent to
    Lemma 2.5 via the Baillon-Haddad theorem (M-Lipschitz gradient of a
    convex functional <=> (1/M)-co-coercivity <=> convexity of F).
  * Property 1.2 (attainment), reused from exp_1: Y_Z has (weighted
    empirical) law approximately nu1, measured via Wasserstein-2 distance.

Run with `python -m exp.exp2`. Writes tables and a figure to results/.
"""

from __future__ import annotations

import json
from typing import Callable

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
    bass_value_identity_terms,
    brenier_map_for_Z,
    inner_product,
    l2_norm,
    make_bass_problem,
)
from src.optimal_transport import BrenierPotential

RESULTS_DIR = "results"

# Tolerance for the v_Z M-Lipschitz check: v_Z is *exactly* a pointwise max of
# affine maps y_j . w - v_Z*(y_j) with |y_j| <= M, so the only slack expected
# is floating-point round-off (mirrors tests/test_optimal_transport.py, which
# uses the same 1e-8 margin for the identical mathematical fact).
TOL_V_LIPSCHITZ = 1e-8

# Tolerance for the g(.) M-Lipschitz ratio and for the convexity residual of
# F: both are exact theorems (Lemma 2.5, Baillon-Haddad), but g(Z) is itself
# a Monte Carlo / finite-K_GAMMA estimate of a conditional expectation, so a
# small relative slack (here 2% of M, resp. of the typical scale of F) is
# allowed for that estimation noise rather than for the underlying math.
TOL_G_LIPSCHITZ_REL = 0.02
TOL_CONVEXITY_REL = 0.02


# ---------------------------------------------------------------------------
# Lemma 2.3: convex K-Lipschitz functions under a Gaussian shift
# ---------------------------------------------------------------------------

def f_norm(x: np.ndarray) -> np.ndarray:
    """f(x) = |x| (Euclidean norm), 1-Lipschitz and convex. x: (..., d)."""
    return np.linalg.norm(x, axis=-1)


def subgrad_norm(x: np.ndarray) -> np.ndarray:
    """A subgradient of f_norm at x != 0 (negligible chance of x == 0 here
    since x is always drawn from a continuous distribution)."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def f_max(x: np.ndarray) -> np.ndarray:
    """f(x) = max(x_1, ..., x_d), 1-Lipschitz (w.r.t. the Euclidean norm,
    since |max(x)-max(y)| <= ||x-y||_inf <= ||x-y||_2) and convex."""
    return np.max(x, axis=-1)


def subgrad_max(x: np.ndarray) -> np.ndarray:
    """A subgradient e_{i*} of f_max at x (negligible chance of an exact tie
    for i* since x is always drawn from a continuous distribution)."""
    idx = np.argmax(x, axis=-1)
    g = np.zeros_like(x)
    rows = np.arange(x.shape[0])
    g[rows, idx] = 1.0
    return g


def delta_h_mc(
    f: Callable[[np.ndarray], np.ndarray],
    subgrad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """One Monte Carlo draw of Delta_h f(x+Gamma) per row of `gamma`.

    Delta_h f(z) = f(z+h) - f(z) - subgrad_f(z).h, z = x + Gamma.
    gamma: (n_mc, d) i.i.d. draws of Gamma ~ N(0, I_d).
    """
    z = x[None, :] + gamma
    return f(z + h[None, :]) - f(z) - subgrad(z) @ h


def lemma_2_3_mc_estimate(
    f: Callable[[np.ndarray], np.ndarray],
    subgrad: Callable[[np.ndarray], np.ndarray],
    K: float,
    x: np.ndarray,
    h: np.ndarray,
    gamma: np.ndarray,
) -> dict:
    """Monte Carlo estimate of E[Delta_h f(x+Gamma)] and the Lemma 2.3 check
    0 <= estimate <= (K/2)|h|^2, up to `tol` Monte Carlo standard errors."""
    delta = delta_h_mc(f, subgrad, x, h, gamma)
    n_mc = delta.shape[0]
    estimate = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / np.sqrt(n_mc)) if n_mc > 1 else 0.0
    h_mag = float(np.linalg.norm(h))
    bound = 0.5 * K * h_mag ** 2
    # Tolerance: 5 Monte Carlo standard errors gives a two-sided band with a
    # negligible false-rejection rate (~1e-6) under the CLT, while being tiny
    # relative to `bound` for the sample sizes used here (n_mc ~ 1e4-2e4).
    # Floating-point floor: for some (f, x, d) combinations (e.g. f_max in
    # d=1, where f is literally affine and the subgradient is a deterministic
    # constant) Delta_h f is *exactly* 0 in exact arithmetic, so every draw
    # agrees up to ~1e-16 round-off and se itself collapses to the
    # floating-point noise floor (~1e-19), making 5*se tighter than the
    # round-off it is meant to absorb. An absolute floor of 1e-9 -- many
    # orders of magnitude above float64 round-off at the O(1-10) magnitudes
    # used here, yet "a small numerical tolerance" relative to `bound` -- is
    # added to cover this degenerate case.
    tol = max(5.0 * se, 1e-9)
    return {
        "estimate": estimate,
        "se": se,
        "h_mag": h_mag,
        "bound": bound,
        "tol": tol,
        "lower_holds": bool(estimate >= -tol),
        "upper_holds": bool(estimate <= bound + tol),
        "holds": bool(estimate >= -tol and estimate <= bound + tol),
    }


LEMMA23_FUNCTIONS: tuple[tuple[str, Callable, Callable, float], ...] = (
    ("euclidean_norm", f_norm, subgrad_norm, 1.0),
    ("max_coordinate", f_max, subgrad_max, 1.0),
)


def check_lemma_2_3(
    d: int,
    rng: np.random.Generator,
    n_mc: int,
    h_grid: tuple[float, ...],
    n_x_trials: int = 5,
) -> list[dict]:
    """Run the Lemma 2.3 Monte Carlo check for both test functions, several
    random x (avoiding exact kinks by construction) and |h| in `h_grid`."""
    rows: list[dict] = []
    for name, f, subgrad, K in LEMMA23_FUNCTIONS:
        for trial in range(n_x_trials):
            # Continuous draw: probability 0 of landing exactly at a kink
            # (x == 0 for f_norm, a coordinate tie for f_max).
            x = rng.normal(size=d) + rng.choice([-1.0, 1.0]) * 0.7
            for h_mag in h_grid:
                direction = rng.normal(size=d)
                direction = direction / np.linalg.norm(direction)
                h = h_mag * direction
                gamma = rng.normal(size=(n_mc, d))
                result = lemma_2_3_mc_estimate(f, subgrad, K, x, h, gamma)
                rows.append({
                    "function": name, "d": d, "trial": trial,
                    "x": x.tolist(), "h_mag": h_mag, **result,
                })
    return rows


# ---------------------------------------------------------------------------
# Lemma 2.4 / 2.5 / convexity of F: Bass-functional machinery
# ---------------------------------------------------------------------------

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


def identity_discrepancy(
    problem: BassProblem,
    Z: np.ndarray,
    gamma: np.ndarray,
    brenier: BrenierPotential | None = None,
) -> dict:
    """Lemma 2.4: compare V(Z) (primal, `bass_functional_value`) against the
    identity V(Z) = E[v_Z(Z+Gamma)] + E_nu1[v_Z*] - E[Z.X]
    (`bass_value_identity_terms`), two computations from the same OT
    solution that should agree up to a tiny numerical discrepancy."""
    if brenier is None:
        brenier = brenier_map_for_Z(problem, Z, gamma)
    V = bass_functional_value(problem, Z, gamma, brenier)
    terms = bass_value_identity_terms(problem, Z, gamma, brenier)
    abs_diff = abs(V - terms["identity_value"])
    rel_diff = abs_diff / (abs(V) + 1e-9)
    return {
        "V": V, "identity_value": terms["identity_value"],
        "abs_diff": abs_diff, "rel_diff": rel_diff,
    }


def brenier_lipschitz_check(
    brenier: BrenierPotential,
    M: float,
    rng: np.random.Generator,
    n_pairs: int = 8,
    tol: float = TOL_V_LIPSCHITZ,
) -> list[dict]:
    """Lemma 2.4: |v_Z(w1) - v_Z(w2)| <= M|w1-w2|, for random query-point
    pairs drawn from the source cloud used to fit `brenier`."""
    source = brenier.source
    n = source.shape[0]
    rows = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        w1, w2 = source[i], source[j]
        v1 = float(brenier.value(w1[None, :])[0])
        v2 = float(brenier.value(w2[None, :])[0])
        dist = float(np.linalg.norm(w1 - w2))
        diff = abs(v1 - v2)
        bound = M * dist
        rows.append({
            "diff": diff, "dist": dist, "bound": bound,
            "holds": bool(diff <= bound + tol),
        })
    return rows


def g_lipschitz_ratio(
    problem: BassProblem,
    Z: np.ndarray,
    W: np.ndarray,
    gamma_Z: np.ndarray,
    gamma_W: np.ndarray,
) -> dict:
    """Lemma 2.5: ||g(W)-g(Z)||_2 / ||W-Z||_2 <= M, each with its own freshly
    fit Brenier potential.

    `gamma_Z` and `gamma_W` are ordinarily the *same* array (common random
    numbers): g(Z) and g(W) are each only a K_GAMMA-replicate Monte Carlo
    estimate of E[Y_.|X], and the discrete-OT re-matching that builds Y_.
    is a combinatorial (piecewise-constant) function of the source cloud
    (same discreteness noted in exp1.py's differentiability check), so
    independent Gamma draws for Z and W inject independent re-matching noise
    into the *difference* g(W)-g(Z) that swamps the K_GAMMA=6 signal and can
    push the empirical ratio several times above M. Reusing the same Gamma
    draws for both evaluations cancels most of that shared noise and leaves
    the ratio comfortably below M, exactly as exp1.py reuses one Gamma draw
    across every V evaluation along a finite-difference direction."""
    g_Z = bass_gradient(problem, Z, gamma_Z)
    g_W = bass_gradient(problem, W, gamma_W)
    diff_g = l2_norm(g_W - g_Z, problem.x_weights)
    diff_zw = l2_norm(W - Z, problem.x_weights)
    ratio = diff_g / diff_zw if diff_zw > 1e-12 else 0.0
    return {
        "diff_g_norm": diff_g, "diff_zw_norm": diff_zw, "ratio": ratio,
        "M": problem.M, "holds": bool(ratio <= problem.M * (1.0 + TOL_G_LIPSCHITZ_REL)),
    }


def f_convexity_residual(
    problem: BassProblem,
    Z: np.ndarray,
    W: np.ndarray,
    gamma_Z: np.ndarray,
    gamma_W: np.ndarray,
) -> dict:
    """Baillon-Haddad / convexity of F(Z) = (M/2)||Z||_2^2 - V(Z): the
    subgradient inequality F(W) >= F(Z) + <grad F(Z), W-Z>_H, equivalent to
    the M-Lipschitz continuity of g (Lemma 2.5).

    `gamma_Z` and `gamma_W` are ordinarily the same array: see the common
    random numbers note in `g_lipschitz_ratio`, which applies identically
    here since F and grad F are built from the same V(Z), g(Z) estimates."""
    M = problem.M
    x_weights = problem.x_weights
    V_Z = bass_functional_value(problem, Z, gamma_Z)
    V_W = bass_functional_value(problem, W, gamma_W)
    g_Z = bass_gradient(problem, Z, gamma_Z)
    F_Z = 0.5 * M * l2_norm(Z, x_weights) ** 2 - V_Z
    F_W = 0.5 * M * l2_norm(W, x_weights) ** 2 - V_W
    grad_F_Z = M * Z - g_Z
    linear_term = inner_product(grad_F_Z, W - Z, x_weights)
    residual = F_W - F_Z - linear_term
    scale = max(abs(F_W), abs(F_Z), abs(linear_term), 1.0)
    tol = TOL_CONVEXITY_REL * scale
    return {
        "F_Z": F_Z, "F_W": F_W, "linear_term": linear_term,
        "residual": residual, "holds": bool(residual >= -tol),
    }


def check_attainment(problem: BassProblem, gamma: np.ndarray) -> dict:
    """Property 1.2, reused from exp_1: Y_Z = grad v_Z(Z+Gamma) should have
    (weighted empirical) law nu1, compared via the Wasserstein-2 distance
    between the weighted empirical law of Y_Z and the nu1 sample. Here
    `gamma` carries k = max(1, EXP2_N_W1_CHECK // n_atoms) replicates per
    atom (not K_GAMMA) since this check alone benefits from a larger,
    dedicated sample to resolve nu1's law accurately; this sample size is
    not specified in the paper."""
    import ot

    Z = problem.x_atoms.copy()
    brenier = brenier_map_for_Z(problem, Z, gamma)
    n_x, k, d = gamma.shape
    source = (Z[:, None, :] + gamma).reshape(n_x * k, d)
    source_weights = np.repeat(problem.x_weights / k, k)
    Y = brenier.gradient(source)

    cost = ot.dist(Y, problem.y_samples, metric="sqeuclidean")
    w2_sq = ot.emd2(source_weights, problem.y_weights, cost)
    w2 = float(np.sqrt(max(w2_sq, 0.0)))
    return {
        "wasserstein2_Y_Z_to_nu1": w2,
        "M": problem.M,
        "relative_w2": w2 / problem.M,
        "k_replicates": k,
        "n_query_points": n_x * k,
    }


def run_pair_checks(
    problem: BassProblem,
    Z: np.ndarray,
    W: np.ndarray,
    gamma: np.ndarray,
    pair_idx: int,
    lipschitz_v_rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    """Run the Lemma 2.4 identity + v_Z-Lipschitz check, the Lemma 2.5
    g-Lipschitz-ratio check, and the F-convexity check, for one (Z, W) pair.

    `gamma` (common random numbers, shared between Z and W) is reused for
    every quantity below: see the note in `g_lipschitz_ratio`."""
    brenier_Z = brenier_map_for_Z(problem, Z, gamma)
    identity = identity_discrepancy(problem, Z, gamma, brenier_Z)
    v_lipschitz_rows = brenier_lipschitz_check(brenier_Z, problem.M, lipschitz_v_rng)

    g_ratio = g_lipschitz_ratio(problem, Z, W, gamma, gamma)
    convexity = f_convexity_residual(problem, Z, W, gamma, gamma)

    row = {
        "pair": pair_idx,
        "identity_abs_diff": identity["abs_diff"],
        "identity_rel_diff": identity["rel_diff"],
        "g_ratio": g_ratio["ratio"],
        "g_ratio_holds": g_ratio["holds"],
        "convexity_residual": convexity["residual"],
        "convexity_holds": convexity["holds"],
    }
    return row, v_lipschitz_rows


def run_for_dimension(d: int) -> dict:
    base = config.BASE_SEED
    synth = sample_bass_pair(d=d, n_atoms=config.N_X_ATOMS, radius=config.SUPPORT_RADIUS,
                              seed=base + 3000 + d)
    problem = make_bass_problem(synth)
    n_x = problem.x_atoms.shape[0]

    lemma23_rng = np.random.default_rng(base + 3010 + d)
    lemma_2_3_rows = check_lemma_2_3(
        d, lemma23_rng, config.EXP2_LEMMA23_N_MC, config.EXP2_LEMMA23_H_GRID,
    )

    pair_rng = np.random.default_rng(base + 3020 + d)
    lipschitz_v_rng = np.random.default_rng(base + 3021 + d)
    pair_rows: list[dict] = []
    lipschitz_v_rows: list[dict] = []
    for p in range(config.EXP2_N_RANDOM_PAIRS):
        Z = _random_Z(problem, pair_rng.uniform(0.3, 1.8), pair_rng)
        W = _random_Z(problem, pair_rng.uniform(0.3, 1.8), pair_rng)
        # Common random numbers: one shared Gamma draw per pair (see the note
        # in g_lipschitz_ratio) rather than independent draws for Z and W.
        gamma = _make_gamma(n_x, d, config.K_GAMMA, seed=base + 4000 + d * 100 + p)
        row, v_rows = run_pair_checks(problem, Z, W, gamma, p, lipschitz_v_rng)
        pair_rows.append(row)
        lipschitz_v_rows.extend(v_rows)

    k_w1 = max(1, config.EXP2_N_W1_CHECK // n_x)
    attainment_gamma = _make_gamma(n_x, d, k_w1, seed=base + 3030 + d)
    attainment = check_attainment(problem, attainment_gamma)

    return {
        "d": d,
        "n_atoms": n_x,
        "M": problem.M,
        "lemma_2_3": lemma_2_3_rows,
        "pairs": pair_rows,
        "lipschitz_v": lipschitz_v_rows,
        "attainment": attainment,
    }


def plot_lipschitz_ratios(all_results: list[dict], path: str) -> None:
    fig, axes = plt.subplots(1, len(all_results), figsize=(5.5 * len(all_results), 4.5), squeeze=False)
    for ax, res in zip(axes[0], all_results):
        ratios = [r["g_ratio"] for r in res["pairs"]]
        M = res["M"]
        ax.scatter(range(len(ratios)), ratios, s=18, label="||g(W)-g(Z)|| / ||W-Z||")
        ax.axhline(M, color="red", linestyle="--", label=f"M = {M:.3f}")
        ax.set_xlabel("pair index")
        ax.set_ylabel("Lipschitz ratio")
        ax.set_title(f"exp_2: Lemma 2.5 check, d={res['d']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    all_results = [run_for_dimension(d) for d in config.DIMENSIONS]

    summary_rows = []
    for res in all_results:
        lemma23_ok = all(r["holds"] for r in res["lemma_2_3"])
        g_ratios = [r["g_ratio"] for r in res["pairs"]]
        max_g_ratio = max(g_ratios)
        g_lipschitz_ok = all(r["g_ratio_holds"] for r in res["pairs"])
        identity_abs = [r["identity_abs_diff"] for r in res["pairs"]]
        identity_rel = [r["identity_rel_diff"] for r in res["pairs"]]
        v_lipschitz_ok = all(r["holds"] for r in res["lipschitz_v"])
        convexity_ok = all(r["convexity_holds"] for r in res["pairs"])
        summary_rows.append({
            "d": res["d"],
            "M": res["M"],
            "lemma_2_3_bound_holds_for_all_trials": lemma23_ok,
            "max_g_lipschitz_ratio": max_g_ratio,
            "g_lipschitz_bound_holds_for_all_pairs": g_lipschitz_ok,
            "max_identity_abs_discrepancy": max(identity_abs),
            "max_identity_rel_discrepancy": max(identity_rel),
            "v_lipschitz_bound_holds_for_all_trials": v_lipschitz_ok,
            "f_convexity_holds_for_all_pairs": convexity_ok,
            "attainment_relative_w2": res["attainment"]["relative_w2"],
        })

    with open(f"{RESULTS_DIR}/exp2_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(f"{RESULTS_DIR}/exp2_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plot_lipschitz_ratios(all_results, f"{RESULTS_DIR}/exp2_lipschitz_ratios.png")

    print("exp_2: Lipschitz continuity of g (Lemmas 2.3-2.5) and convexity of F")
    for row in summary_rows:
        print(
            f"  d={row['d']}: lemma_2_3_holds={row['lemma_2_3_bound_holds_for_all_trials']} "
            f"max_g_ratio/M={row['max_g_lipschitz_ratio']:.4f}/{row['M']:.4f} "
            f"(holds={row['g_lipschitz_bound_holds_for_all_pairs']}) "
            f"max_identity_rel_diff={row['max_identity_rel_discrepancy']:.2e} "
            f"v_lipschitz_holds={row['v_lipschitz_bound_holds_for_all_trials']} "
            f"F_convex_holds={row['f_convexity_holds_for_all_pairs']} "
            f"attainment_rel_W2={row['attainment_relative_w2']:.4f}"
        )


if __name__ == "__main__":
    main()
