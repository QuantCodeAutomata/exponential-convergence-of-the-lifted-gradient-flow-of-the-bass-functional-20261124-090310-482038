"""Discrete optimal transport and Brenier-potential reconstruction.

The Bass functional (paper Section 1.1) is built from the Brenier optimal
transport map pushing the law of Z + Gamma forward to nu1 (Property 1.2):
there is a convex function v_Z such that Y_Z = grad v_Z(Z + Gamma) ~ nu1.
This module solves the discrete analogue of that transport problem with the
POT library (exact network-simplex solver, `ot.emd`) and reconstructs v_Z
from the Kantorovich dual potentials using the quadratic-cost conjugacy

    v_Z(w) = sup_y [ w . y - v_Z*(y) ],    v_Z*(y_j) = (1/2)|y_j|^2 - v_j,

where v_j is the dual potential on the target side of the discrete optimal
transport problem with cost C_ij = (1/2)|s_i - y_j|^2 (McCann/Brenier duality,
e.g. Villani, "Topics in Optimal Transportation", Theorem 2.9). Restricting
the supremum to the finite set of sampled target points {y_j} gives the
"Laguerre cell" / power-diagram reconstruction of the convex potential that is
standard in computational optimal transport; grad v_Z(w) is then y_{j*(w)}
for the maximizing index j*(w) (a piecewise-constant approximation of the
true gradient map, consistent with Property 1.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot  # POT: Python Optimal Transport, exact discrete OT via ot.emd


@dataclass
class BrenierPotential:
    """A convex function reconstructed as the max of N affine pieces.

    Attributes
    ----------
    target : (n_t, d) array, the points y_j used to build the affine pieces.
    conjugate_values : (n_t,) array, v_Z*(y_j) for each target point.
    source : (n_s, d) array, the source points used to fit the potential.
    source_potential : (n_s,) array, v_Z(source_i) from the dual solution.
    matched_index : (n_s,) int array, the index j matched to source_i in the
        discrete optimal transport plan (the Monge map at the source points).
    """

    target: np.ndarray
    conjugate_values: np.ndarray
    source: np.ndarray
    source_potential: np.ndarray
    matched_index: np.ndarray

    def value(self, w: np.ndarray) -> np.ndarray:
        """v_Z(w) for query points w, shape (Q, d) -> (Q,)."""
        w = np.atleast_2d(w)
        scores = w @ self.target.T - self.conjugate_values[None, :]
        return scores.max(axis=1)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """grad v_Z(w) = Y_Z(w), the Brenier map at query points w, shape (Q, d)."""
        w = np.atleast_2d(w)
        scores = w @ self.target.T - self.conjugate_values[None, :]
        return self.target[scores.argmax(axis=1)]

    def value_and_gradient(self, w: np.ndarray):
        w = np.atleast_2d(w)
        scores = w @ self.target.T - self.conjugate_values[None, :]
        idx = scores.argmax(axis=1)
        return scores[np.arange(len(w)), idx], self.target[idx]


def fit_brenier_map(
    source: np.ndarray,
    source_weights: np.ndarray,
    target: np.ndarray,
    target_weights: np.ndarray,
) -> BrenierPotential:
    """Solve the discrete quadratic-cost optimal transport problem and build
    the Brenier potential v_Z pushing `source` (weighted by `source_weights`)
    forward to `target` (weighted by `target_weights`).

    `source` plays the role of (a discretization of) the law of Z + Gamma and
    `target` the role of (a discretization of) nu1, as in Property 1.2.
    """
    source = np.atleast_2d(source).astype(float)
    target = np.atleast_2d(target).astype(float)
    source_weights = np.asarray(source_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)
    source_weights = source_weights / source_weights.sum()
    target_weights = target_weights / target_weights.sum()

    cost = ot.dist(source, target, metric="sqeuclidean") / 2.0
    # numItermax scales with problem size: the network-simplex solver needs
    # more pivots than its 100000 default once source/target grow past a few
    # hundred points each.
    iter_max = max(200_000, 200 * (len(source_weights) + len(target_weights)))
    plan, log = ot.emd(source_weights, target_weights, cost, numItermax=iter_max, log=True)
    if log.get("warning"):
        raise RuntimeError(f"ot.emd did not converge cleanly: {log['warning']}")

    u = log["u"]  # dual potential on the source side
    v = log["v"]  # dual potential on the target side
    conjugate_values = 0.5 * np.sum(target ** 2, axis=1) - v  # v_Z*(y_j)
    source_potential = 0.5 * np.sum(source ** 2, axis=1) - u  # v_Z(s_i)
    matched_index = plan.argmax(axis=1)

    return BrenierPotential(
        target=target,
        conjugate_values=conjugate_values,
        source=source,
        source_potential=source_potential,
        matched_index=matched_index,
    )
