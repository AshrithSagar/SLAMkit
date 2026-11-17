"""
Point2
======
A frame-tagged 2D point.
"""
# src/slamkit/geometry/point.py

import math
from dataclasses import dataclass, replace
from typing import ClassVar, Self

from slamkit.core.arrays import Vec2, Vector
from slamkit.core.frames import Frame
from slamkit.geometry.manifold import Manifold

__all__ = [
    "Point2",
]


@dataclass(frozen=True, slots=True)
class Point2[F: Frame](Manifold["Point2[F]"]):
    """A point ``(x, y)`` expressed in frame ``F``.

    ``Point2[WorldFrame]`` and ``Point2[SensorFrame]`` are distinct types; the
    only way to move a point between frames is through an ``SE2`` transform
    whose frame parameters line up.
    """

    x: float
    y: float

    DOF: ClassVar[int] = 2

    @classmethod
    def from_vector(cls, vector: Vector) -> Point2[F]:
        if vector.shape != (2,):
            raise ValueError(f"Point2 expects a 2-vector, got shape {vector.shape}")
        return cls(float(vector[0]), float(vector[1]))

    def to_vector(self) -> Vec2:
        return Vec2([self.x, self.y])

    def oplus(self, delta: Vector) -> Self:
        return replace(self, x=self.x + float(delta[0]), y=self.y + float(delta[1]))

    def ominus(self, other: Point2[F]) -> Vector:
        return Vector([self.x - other.x, self.y - other.y])

    def distance_to(self, other: Point2[F]) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
