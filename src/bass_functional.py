"""The Bass functional V(Z) and its gradient g(Z) (paper Section 1.1).

H = L^2(sigma(X); R^d) is represented by the finite set of atoms of nu0 built
in data/generate_synthetic.py: a vector Z in H is an (n_atoms, d) array giving
one value per atom. The conditional expectation E[.|X] is estimated by
K_GAMMA independent draws of Gamma per atom (common random numbers: the same
Gamma draws are reused for every Z along one experiment so that V and g are
deterministic, smooth functions of Z given the fixed Monte Carlo sample).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.optimal_transport import BrenierPotential, fit_brenier_map


@dataclass
class BassProblem:
    """A discretized instance of the Bass-functional minimization problem."""

    x_atoms: np.ndarray      # (n_x, d), the atoms of nu0
    x_weights: np.ndarray    # (n_x,), probabilities of the atoms, sum to 1
    y_samples: np.ndarray    # (n_y, d), i.i.d. draws approximating nu1
    y_weights: np.ndarray    # (n_y,), uniform weights, sum to 1
    M: float                 # support radius of nu1 (Section 1.1, item 1)
    d: int                   # ambient dimension


def make_bass_problem(synthetic: dict) -> BassProblem:
    """Wrap the output of data.generate_synthetic.sample_bass_pair."""
    x_atoms = synthetic["x_atoms"]
    y_samples = synthetic["y_samples"]
    n_y = y_samples.shape[0]
    return BassProblem(
        x_atoms=x_atoms,
        x_weights=synthetic["x_weights"],
        y_samples=y_samples,
        y_weights=np.full(n_y, 1.0 / n_y),
        M=synthetic["M"],
        d=x_atoms.shape[1],
    )


def _source_points_and_weights(problem: BassProblem, Z: np.ndarray, gamma: np.ndarray):
    """Flatten the (atom, replicate) grid of Z_i + Gamma_{i,k} into one point
    cloud with weights x_weights[i] / K spread over the K replicates."""
    n_x, k, d = gamma.shape
    if Z.shape != (n_x, d):
        raise ValueError(f"Z must have shape {(n_x, d)}, got {Z.shape}")
    source = (Z[:, None, :] + gamma).reshape(n_x * k, d)
    weights = np.repeat(problem.x_weights / k, k)
    return source, weights


def brenier_map_for_Z(problem: BassProblem, Z: np.ndarray, gamma: np.ndarray) -> BrenierPotential:
    """Fit v_Z: the Brenier potential pushing the law of Z + Gamma to nu1.

    gamma has shape (n_x, k, d): k independent standard Gaussian draws per atom.
    """
    source, weights = _source_points_and_weights(problem, Z, gamma)
    return fit_brenier_map(source, weights, problem.y_samples, problem.y_weights)


def bass_functional_value(problem: BassProblem, Z: np.ndarray, gamma: np.ndarray,
                           brenier: BrenierPotential | None = None) -> float:
    """V(Z) via the defining supremum (paper Section 1.1):

        V(Z) = E[(Z + Gamma) . Y_Z] - E[Z . X],

    evaluated at the maximizer Y_Z = grad v_Z(Z + Gamma) (Property 1.2), with
    the expectations replaced by the weighted empirical averages of the
    discretized problem.
    """
    if brenier is None:
        brenier = brenier_map_for_Z(problem, Z, gamma)
    n_x, k, d = gamma.shape
    source, weights = _source_points_and_weights(problem, Z, gamma)
    y_matched = brenier.gradient(source)
    cross_term = np.sum(weights * np.sum(source * y_matched, axis=1))
    mean_term = np.sum(problem.x_weights * np.sum(Z * problem.x_atoms, axis=1))
    return float(cross_term - mean_term)


def bass_gradient(problem: BassProblem, Z: np.ndarray, gamma: np.ndarray,
                   brenier: BrenierPotential | None = None) -> np.ndarray:
    """g(Z) = E[Y_Z | X] - X, shape (n_x, d) (Property 1.3).

    E[Y_Z | X = x_i] is estimated by averaging Y_Z(Z_i + Gamma_{i,k}) over the
    k = 1..K_GAMMA replicates attached to atom i.
    """
    if brenier is None:
        brenier = brenier_map_for_Z(problem, Z, gamma)
    n_x, k, d = gamma.shape
    source = (Z[:, None, :] + gamma).reshape(n_x * k, d)
    y_matched = brenier.gradient(source).reshape(n_x, k, d)
    conditional_mean = y_matched.mean(axis=1)
    return conditional_mean - problem.x_atoms


def bass_value_identity_terms(problem: BassProblem, Z: np.ndarray, gamma: np.ndarray,
                               brenier: BrenierPotential | None = None) -> dict:
    """Terms of the identity V(Z) = E[v_Z(Z+Gamma)] + E_{nu1}[v_Z*] - E[Z.X]
    (paper Lemma 2.4, equation (4)), for cross-checking against the primal
    value computed by `bass_functional_value`.
    """
    if brenier is None:
        brenier = brenier_map_for_Z(problem, Z, gamma)
    source, weights = _source_points_and_weights(problem, Z, gamma)
    e_v_source = float(np.sum(weights * brenier.value(source)))
    e_conjugate_nu1 = float(np.sum(problem.y_weights * brenier.conjugate_values))
    mean_term = float(np.sum(problem.x_weights * np.sum(Z * problem.x_atoms, axis=1)))
    identity_value = e_v_source + e_conjugate_nu1 - mean_term
    return {
        "e_v_source": e_v_source,
        "e_conjugate_nu1": e_conjugate_nu1,
        "mean_term": mean_term,
        "identity_value": identity_value,
    }


def inner_product(U: np.ndarray, Z: np.ndarray, x_weights: np.ndarray) -> float:
    """<U, Z>_H = E[U . Z], the inner product of H = L^2(sigma(X); R^d)."""
    return float(np.sum(x_weights * np.sum(U * Z, axis=1)))


def l2_norm(Z: np.ndarray, x_weights: np.ndarray) -> float:
    """||Z||_2 = (E[|Z|^2])^{1/2}."""
    return float(np.sqrt(np.sum(x_weights * np.sum(Z ** 2, axis=1))))
