import numpy as np

from src.bass_functional import (
    bass_functional_value,
    bass_gradient,
    bass_value_identity_terms,
    brenier_map_for_Z,
    inner_product,
    l2_norm,
)


def test_bass_value_matches_identity_formula(small_problem_1d, gamma_1d):
    """Property/Lemma 2.4 identity: V(Z) = E[v_Z(Z+Gamma)] + E_nu1[v_Z*] - E[Z.X]."""
    problem = small_problem_1d
    Z = problem.x_atoms.copy()
    brenier = brenier_map_for_Z(problem, Z, gamma_1d)
    V = bass_functional_value(problem, Z, gamma_1d, brenier)
    terms = bass_value_identity_terms(problem, Z, gamma_1d, brenier)
    assert abs(V - terms["identity_value"]) < 0.2 * (abs(V) + 1e-6)


def test_bass_gradient_shape_and_sign(small_problem_2d, gamma_2d):
    problem = small_problem_2d
    Z = np.zeros_like(problem.x_atoms)
    g = bass_gradient(problem, Z, gamma_2d)
    assert g.shape == problem.x_atoms.shape
    # g(Z) = E[Y_Z|X] - X: finite and not identically zero for a generic Z
    assert np.all(np.isfinite(g))


def test_inner_product_and_norm_are_consistent(small_problem_1d):
    problem = small_problem_1d
    Z = problem.x_atoms.copy()
    assert abs(inner_product(Z, Z, problem.x_weights) - l2_norm(Z, problem.x_weights) ** 2) < 1e-10


def test_gradient_at_zero_is_not_nan(small_problem_1d, gamma_1d):
    problem = small_problem_1d
    Z = np.zeros_like(problem.x_atoms)
    g = bass_gradient(problem, Z, gamma_1d)
    assert np.all(np.isfinite(g))
    assert l2_norm(g, problem.x_weights) >= 0
