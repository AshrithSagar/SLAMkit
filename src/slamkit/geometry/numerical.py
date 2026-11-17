"""
Numerical differentiation
=========================
Central-difference Jacobians between manifolds.

Every analytic Jacobian in slamkit is tested against this. It is also the
quickest way to get a new model working before deriving its Jacobians by hand.
"""
# src/slamkit/geometry/numerical.py

from collections.abc import Callable
from typing import Any

import numpy as np

from slamkit.core.arrays import Matrix, Vector
from slamkit.geometry.manifold import Manifold

__all__ = [
    "NumericalJacobian",
]


class NumericalJacobian:
    """Computes ``∂ f(x) / ∂ δ`` where ``f(x ⊕ δ) ⊖ f(x) ≈ J δ``."""

    def __init__(self, step: float = 1e-6) -> None:
        if step <= 0.0:
            raise ValueError("step must be positive")
        self._step: float = step

    def __call__[X: Manifold[Any], Y: Manifold[Any]](
        self, function: Callable[[X], Y], at: X
    ) -> Matrix:
        reference = function(at)
        columns: list[Vector] = []
        for i in range(at.DOF):
            delta = np.zeros(at.DOF)
            delta[i] = self._step
            forward = function(at.oplus(Vector(delta))).ominus(reference)
            backward = function(at.oplus(Vector(-delta))).ominus(reference)
            columns.append(Vector((forward - backward) / (2.0 * self._step)))
        return Matrix(np.column_stack(columns))
