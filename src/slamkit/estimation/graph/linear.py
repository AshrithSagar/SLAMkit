"""
Linear system
=============
The Gauss-Newton normal equations of a factor graph, ``H δ = -g`` with
``H = Jᵀ J`` and ``g = Jᵀ r`` over the stacked whitened factors.

Dense, which keeps the code obvious and is fine for the problem sizes the
examples use. A sparse (or Schur-complement, or incremental) solver would
replace this class without touching the factors or the optimisers' logic.
"""
# src/slamkit/estimation/graph/linear.py

from collections.abc import Iterable, Mapping

import numpy as np

from slamkit.core.arrays import Matrix, Vector
from slamkit.estimation.graph.graph import FactorGraph
from slamkit.estimation.graph.values import Key, Values

__all__ = [
    "NormalEquations",
    "Ordering",
]


class Ordering:
    """Assigns each variable a contiguous block of columns."""

    def __init__(self, keys: Iterable[Key]) -> None:
        self._offsets: dict[Key, int] = {}
        offset = 0
        for key in keys:
            self._offsets[key] = offset
            offset += key.DOF
        self._dim = offset

    @classmethod
    def of(cls, values: Values) -> Ordering:
        return cls(iter(values))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def keys(self) -> Iterable[Key]:
        return self._offsets.keys()

    def __getitem__(self, key: Key) -> slice:
        offset = self._offsets[key]
        return slice(offset, offset + key.DOF)

    def indices(self, *keys: Key) -> list[int]:
        """Flat column indices of ``keys``' blocks, in the given order."""
        return [
            i
            for key in keys
            for i in range(self._offsets[key], self._offsets[key] + key.DOF)
        ]

    def split(self, vector: Vector) -> Mapping[Key, Vector]:
        """Cut a full-length vector into per-key blocks."""
        return {key: Vector(vector[self[key]]) for key in self._offsets}


class NormalEquations:
    """``H δ = -g`` for a graph linearised at ``values``."""

    def __init__(
        self, graph: FactorGraph, values: Values, ordering: Ordering | None = None
    ) -> None:
        self._ordering = ordering if ordering is not None else Ordering.of(values)
        n = self._ordering.dim
        hessian = np.zeros((n, n))
        gradient = np.zeros(n)
        for factor in graph:
            linearized = factor.linearize(values)
            blocks = [self._ordering[key] for key in linearized.keys]
            for bi, ji in zip(blocks, linearized.jacobians, strict=True):
                gradient[bi] += ji.T @ linearized.residual
                for bj, jj in zip(blocks, linearized.jacobians, strict=True):
                    hessian[bi, bj] += ji.T @ jj
        self._hessian = Matrix(hessian)
        self._gradient = Vector(gradient)

    @property
    def ordering(self) -> Ordering:
        return self._ordering

    @property
    def hessian(self) -> Matrix:
        return self._hessian

    @property
    def gradient(self) -> Vector:
        return self._gradient

    def solve(self, damping: float = 0.0) -> Mapping[Key, Vector]:
        """Solve ``(H + λ·diag(H)) δ = -g``; ``λ = 0`` is the Gauss-Newton step."""
        system = self._hessian + damping * np.diag(np.diag(self._hessian))
        return self._ordering.split(Vector(np.linalg.solve(system, -self._gradient)))

    def covariance(self) -> Matrix:
        """``H⁻¹``: the Laplace approximation of the joint posterior covariance."""
        return Matrix(np.linalg.inv(self._hessian))
