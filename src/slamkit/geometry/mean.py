"""
Manifold mean
=============
The weighted mean of points on a manifold.
"""
# src/slamkit/geometry/mean.py

from collections.abc import Sequence
from typing import Any

import numpy as np

from slamkit.core.arrays import Vector
from slamkit.geometry.manifold import Manifold

__all__ = [
    "ManifoldMean",
]


class ManifoldMean:
    """Iterates ``μ ← μ ⊕ Σ wᵢ (xᵢ ⊖ μ)`` to a fixed point.

    On a vector space this is the ordinary weighted mean after one step; on a
    curved space (headings, SE(2)) it is the point at which the weighted
    tangent-space deviations balance -- what "the mean pose" has to mean.
    """

    def __init__(self, max_iterations: int = 20, tolerance: float = 1e-12) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def __call__[M: Manifold[Any]](self, points: Sequence[M], weights: Vector) -> M:
        if len(points) != weights.shape[0] or not points:
            raise ValueError("need one weight per point, and at least one point")
        mean = points[0]
        for _ in range(self._max_iterations):
            step = Vector(
                np.sum(
                    [w * p.ominus(mean) for p, w in zip(points, weights, strict=True)],
                    axis=0,
                )
            )
            mean = mean.oplus(step)
            if float(np.linalg.norm(step)) < self._tolerance:
                break
        return mean
