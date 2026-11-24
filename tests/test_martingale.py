import numpy as np

from src.bass_functional import brenier_map_for_Z
from src.martingale import bass_martingale_paths


def test_martingale_terminal_law_matches_target_support(small_problem_1d):
    problem = small_problem_1d
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(5)
    gamma = rng.normal(size=(n_x, 5, d))
    Z = problem.x_atoms.copy()
    brenier = brenier_map_for_Z(problem, Z, gamma)
    result = bass_martingale_paths(
        brenier, Z[0], s_grid=np.array([0.0, 0.5, 1.0]), n_outer=10, n_inner=50, seed=6
    )
    # M_1 = grad v_Z(Z + B_1) must take values in supp(nu1) exactly, since the
    # reconstructed potential is a max of affine pieces indexed by the target
    # sample (Property 1.2: the terminal law of the Bass martingale is nu1)
    for m1 in result["M_1_exact"]:
        assert np.any(np.all(np.isclose(problem.y_samples, m1), axis=1))


def test_martingale_shapes_and_validation(small_problem_1d):
    problem = small_problem_1d
    n_x, d = problem.x_atoms.shape
    rng = np.random.default_rng(5)
    gamma = rng.normal(size=(n_x, 5, d))
    brenier = brenier_map_for_Z(problem, problem.x_atoms, gamma)
    s_grid = np.array([0.0, 0.3, 1.0])
    result = bass_martingale_paths(brenier, problem.x_atoms[0], s_grid, n_outer=4, n_inner=20, seed=1)
    assert result["M_s"].shape == (4, 3, d)
    assert result["W_s"].shape == (4, 3, d)

    import pytest

    with pytest.raises(ValueError):
        bass_martingale_paths(brenier, problem.x_atoms[0], np.array([0.1, 1.0]), 4, 20, 1)
