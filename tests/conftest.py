"""Shared fixtures for the Bass-functional test suite.

This paper's experiments use only synthetic data generated in
data/generate_synthetic.py (no market data is involved), so no
`market_data` marker is required here.
"""

import numpy as np
import pytest

from data.generate_synthetic import sample_bass_pair
from src.bass_functional import make_bass_problem


@pytest.fixture
def small_problem_1d():
    """A small d=1 Bass problem: fast enough for unit tests."""
    synth = sample_bass_pair(d=1, n_atoms=10, radius=1.0, seed=42)
    return make_bass_problem(synth)


@pytest.fixture
def small_problem_2d():
    """A small d=2 Bass problem: fast enough for unit tests."""
    synth = sample_bass_pair(d=2, n_atoms=10, radius=1.0, seed=42)
    return make_bass_problem(synth)


@pytest.fixture
def gamma_1d(small_problem_1d):
    n_x, d = small_problem_1d.x_atoms.shape
    rng = np.random.default_rng(7)
    return rng.normal(size=(n_x, 4, d))


@pytest.fixture
def gamma_2d(small_problem_2d):
    n_x, d = small_problem_2d.x_atoms.shape
    rng = np.random.default_rng(7)
    return rng.normal(size=(n_x, 4, d))
