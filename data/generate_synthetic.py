"""Synthetic (nu0, nu1) pairs obeying the standing assumptions of Section 1.1.

The paper states its properties and theorems for any pair of laws (nu0, nu1)
on R^d with:

  1. supp(nu1) compact, with M = max_{y in supp nu1} |y|;
  2. nu0 <=_cx nu1 (nu0 dominated by nu1 in convex order);
  3. supp(nu0) contained in the interior of conv(supp nu1);
  4. (nu0, nu1) irreducible (Section 1.1, item 4).

It gives no concrete numerical example, so this module builds one explicit,
assumption-compliant instance per dimension using the standard sufficient
condition for the convex order: if X has a bounded law and U is an
independent, bounded, zero-mean perturbation, then X <=_cx X+U, because
Jensen's inequality applied to the conditional expectation E[X+U | X] = X
gives E[f(X)] = E[f(E[X+U|X])] <= E[f(X+U)] for every convex f.

* nu0 = law(X), with X uniform on the ball of radius `core_radius` (a small
  fraction of `radius`), discretized as `n_atoms` i.i.d. atoms with equal
  weight.
* nu1 = law(X+U), with U uniform on the ball of radius `radius - core_radius`,
  independent of X and of mean zero; `n_rep` independent replicates of U are
  drawn per atom and pooled into the empirical nu1 sample. This gives
  nu0 <=_cx nu1 by the Jensen argument above (item 2) and, since every atom
  of nu0 lies in the ball of radius `core_radius` while nu1's samples reach
  out to approximately `radius`, supp(nu0) sits with a margin of at least
  `radius - core_radius` inside conv(supp nu1) (item 3) -- unlike a
  K-means/conditional-mean split of a single nu1 sample, whose cell means can
  sit arbitrarily close to the boundary of supp(nu1) and collapse the
  constant b of Theorem 1.1 (Section 3.1) to a numerically negligible value.
* Irreducibility (item 4) is not verified numerically: the paper notes this
  condition is hard to check for arbitrary discrete laws. Every atom of nu0
  is linked to every region of nu1 with positive probability through the
  shared noise U, which is the natural candidate witness for irreducibility,
  but we do not prove the defining property here.
* M = max_{y in supp nu1} |y| is read off the realized sample (<= radius by
  construction, with equality approached as n_rep grows).

H = L^2(sigma(X); R^d) is represented by the finite set of nu0 atoms; a
vector Z in H is an array of one R^d-value per atom.
"""

from __future__ import annotations

import numpy as np


def _sample_ball(rng: np.random.Generator, n: int, d: int, radius: float) -> np.ndarray:
    """n i.i.d. draws from the uniform law on the ball of given radius in R^d."""
    directions = rng.normal(size=(n, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.uniform(size=n) ** (1.0 / d)
    return directions * radii[:, None]


def sample_bass_pair(
    d: int,
    n_atoms: int,
    radius: float,
    seed: int,
    core_fraction: float = 0.15,
    n_rep: int = 40,
) -> dict:
    """Build one (nu0, nu1) instance in R^d obeying the standing assumptions.

    Parameters
    ----------
    d : dimension of the ambient space R^d.
    n_atoms : number of atoms of nu0.
    radius : target compact-support radius of nu1 (M is realized <= radius).
    seed : random seed.
    core_fraction : supp(nu0)'s ball radius as a fraction of `radius`
                    (not specified in the paper; chosen so that nu0's support
                    sits with a comfortable margin inside conv(supp nu1)).
    n_rep : independent noise replicates per atom pooled into the nu1 sample
            (not specified in the paper; our discretization of nu1).

    Returns
    -------
    dict with keys:
      'y_samples'  : (n_atoms * n_rep, d) array, i.i.d. draws from nu1.
      'x_atoms'    : (n_atoms, d) array, the atoms of nu0.
      'x_weights'  : (n_atoms,) array of probabilities, summing to 1.
      'assignment' : (n_atoms * n_rep,) int array, the atom index of each
                     y_sample's conditioning atom (the shared noise coupling).
      'M'          : float, max |y| realized over `y_samples`.
    """
    if d < 1 or n_atoms < 2 or radius <= 0 or not (0 < core_fraction < 1) or n_rep < 1:
        raise ValueError("invalid sample_bass_pair arguments")
    rng = np.random.default_rng(seed)
    core_radius = core_fraction * radius
    noise_radius = radius - core_radius

    x_atoms = _sample_ball(rng, n_atoms, d, core_radius)
    x_weights = np.full(n_atoms, 1.0 / n_atoms)

    noise = _sample_ball(rng, n_atoms * n_rep, d, noise_radius).reshape(n_atoms, n_rep, d)
    y_samples = (x_atoms[:, None, :] + noise).reshape(n_atoms * n_rep, d)
    assignment = np.repeat(np.arange(n_atoms), n_rep)

    # nu0 <=_cx nu1 forces the two *population* means to coincide (Jensen's
    # inequality applied to f(z) = +-e.z for every direction e); the noise
    # added above has mean zero by construction, so this holds for the exact
    # population laws. At finite sample size, however, the empirical mean of
    # `y_samples` need not exactly equal the empirical mean of `x_atoms` (the
    # realized average of `noise` over n_atoms*n_rep draws is itself only
    # approximately zero). Re-centering `y_samples` here makes the *finite
    # discretization* obey the exact-mean consequence of the convex order,
    # which the gradient g(Z) = E[Y_Z|X]-X otherwise violates by a small but
    # systematic constant (a discretization artifact, not a property of the
    # flow): left uncorrected, Euler-integrating this constant bias over many
    # gradient-flow steps (Theorem 1.2, exp_4) produces a spurious drift in
    # E[Z_t] that numerically masks the mean-conservation property (Property
    # 1.5) the flow should exhibit exactly.
    mean_x = np.sum(x_weights[:, None] * x_atoms, axis=0)
    mean_y = y_samples.mean(axis=0)
    y_samples = y_samples - mean_y + mean_x
    M = float(np.linalg.norm(y_samples, axis=1).max())

    return {
        "y_samples": y_samples,
        "x_atoms": x_atoms,
        "x_weights": x_weights,
        "assignment": assignment,
        "M": M,
    }


def sample_gamma(n: int, d: int, seed: int) -> np.ndarray:
    """n i.i.d. standard Gaussian vectors in R^d, shape (n, d) (paper: Gamma)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def sample_brownian_increments(n_paths: int, d: int, times: np.ndarray, seed: int) -> np.ndarray:
    """Standard d-dimensional Brownian motion sampled at `times`.

    `times` must start at 0 and be increasing. Returns an array of shape
    (len(times), n_paths, d) with B_0 = 0 (paper Property 1.7, Section 5).
    """
    times = np.asarray(times, dtype=float)
    if times[0] != 0.0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must start at 0 and be strictly increasing")
    rng = np.random.default_rng(seed)
    increments = np.diff(times)
    paths = np.zeros((len(times), n_paths, d))
    for k, dt in enumerate(increments):
        step = rng.normal(scale=np.sqrt(dt), size=(n_paths, d))
        paths[k + 1] = paths[k] + step
    return paths
