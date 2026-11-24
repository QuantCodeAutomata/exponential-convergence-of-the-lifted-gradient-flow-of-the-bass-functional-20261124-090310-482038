import numpy as np
import pytest

from src.optimal_transport import fit_brenier_map


def test_brenier_potential_is_M_lipschitz_1d():
    """v_Z = max_j (w.y_j - c_j) is pointwise max of affine maps with slope
    y_j, so it is exactly M-Lipschitz with M = max|y_j| (paper Lemma 2.4)."""
    rng = np.random.default_rng(0)
    source = rng.normal(size=(40, 1))
    target = rng.uniform(-1, 1, size=(30, 1))
    M = np.max(np.abs(target))
    bp = fit_brenier_map(source, np.ones(40), target, np.ones(30))
    w1 = rng.uniform(-2, 2, size=(100, 1))
    w2 = rng.uniform(-2, 2, size=(100, 1))
    v1, v2 = bp.value(w1), bp.value(w2)
    ratios = np.abs(v1 - v2) / np.linalg.norm(w1 - w2, axis=1)
    assert np.all(ratios <= M + 1e-8)


def test_brenier_gradient_values_lie_in_target_support():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(50, 2))
    target = rng.normal(size=(35, 2))
    bp = fit_brenier_map(source, np.ones(50), target, np.ones(35))
    grads = bp.gradient(source)
    # every returned gradient value must be one of the target atoms
    for g in grads:
        assert np.any(np.all(np.isclose(target, g), axis=1))


def test_fit_brenier_map_rejects_mismatched_shapes():
    with pytest.raises(Exception):
        fit_brenier_map(np.zeros((3, 1)), np.ones(3), np.zeros((4, 1)), np.ones(5))
