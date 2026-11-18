"""
Ray
===
A half-line: the locus of points a direction-only measurement allows.
"""
# src/slamkit/geometry/ray.py

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Vec2, Vector
from slamkit.core.frames import Frame
from slamkit.geometry.point import Point2

__all__ = [
    "Ray",
]


@dataclass(frozen=True, slots=True)
class Ray[F: Frame]:
    """``origin + t · direction`` for ``t ≥ 0``, in frame ``F``."""

    origin: Point2[F]
    direction: Vec2
    """Unit length."""

    def __post_init__(self) -> None:
        norm = float(np.linalg.norm(self.direction))
        if not math.isclose(norm, 1.0, rel_tol=1e-9):
            object.__setattr__(self, "direction", Vec2(self.direction / norm))

    def point_at(self, distance: float) -> Point2[F]:
        return self.origin.oplus(Vector(distance * self.direction))

    def depth_of(self, point: Point2[F]) -> float:
        """Signed distance of ``point``'s projection along the ray."""
        return float(point.ominus(self.origin) @ self.direction)

    def angle_to(self, other: Ray[F]) -> float:
        """The angle between the two directions, in ``[0, π]``."""
        cosine = float(np.clip(self.direction @ other.direction, -1.0, 1.0))
        return math.acos(cosine)

    @staticmethod
    def closest_point[G: Frame](rays: Sequence[Ray[G]]) -> Point2[G] | None:
        """The point minimising the summed squared distances to the rays' lines,
        or ``None`` if they are (near-)parallel."""
        system = np.zeros((2, 2))
        target = np.zeros(2)
        for ray in rays:
            projector = np.eye(2) - np.outer(ray.direction, ray.direction)
            system += projector
            target += projector @ ray.origin.to_vector()
        if np.linalg.cond(system) > 1e12:
            return None
        solution = np.linalg.solve(system, target)
        return rays[0].origin.from_vector(Vector(solution))
