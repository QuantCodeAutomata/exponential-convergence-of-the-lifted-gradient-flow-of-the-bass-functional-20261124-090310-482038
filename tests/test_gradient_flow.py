import numpy as np

from src.gradient_flow import fit_exponential_rate, run_gradient_flow


def test_gradient_flow_conserves_mean(small_problem_1d):
    problem = small_problem_1d
    Z0 = 0.5 * problem.x_atoms
    history = run_gradient_flow(problem, Z0, dt=0.1, n_steps=15, k_gamma=4, seed=3)
    mean0 = history.mean_path[0]
    # Property 1.5: E[Z_t] = E[Z_0] for all t (the gradient g(Z) has zero mean
    # by construction, since E[Y_Z] = E[X] under the matching constraint)
    assert np.allclose(history.mean_path, mean0, atol=0.25)


def test_gradient_flow_reduces_functional_value(small_problem_1d):
    problem = small_problem_1d
    Z0 = 1.5 * problem.x_atoms
    history = run_gradient_flow(problem, Z0, dt=0.05, n_steps=40, k_gamma=4, seed=3)
    # V(Z_t) must be non-increasing along a (sub)gradient-descent flow, up to
    # Monte Carlo noise from resampling Gamma at every step
    assert history.V_path[-1] <= history.V_path[0] + 0.5


def test_fit_exponential_rate_recovers_known_rate():
    t = np.linspace(0, 10, 50)
    values = 3.0 * np.exp(-0.4 * t)
    fit = fit_exponential_rate(t, values, skip_fraction=0.0)
    assert abs(fit["rate"] - 0.4) < 1e-6


def test_fit_exponential_rate_rejects_degenerate_input():
    import pytest

    with pytest.raises(ValueError):
        fit_exponential_rate(np.array([0.0]), np.array([1.0]), skip_fraction=0.0)
