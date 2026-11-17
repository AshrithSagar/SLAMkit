"""
World
=====
Ground-truth landmark layouts.
"""
# src/slamkit/simulation/world.py

import math
from collections.abc import Iterator, Mapping

import numpy as np

from slamkit.core.frames import WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2

__all__ = [
    "LandmarkWorld",
]


class LandmarkWorld:
    """A static set of point landmarks with their true identities."""

    def __init__(self, landmarks: Mapping[LandmarkId, Point2[WorldFrame]]) -> None:
        self._landmarks = dict(landmarks)

    @classmethod
    def uniform(
        cls,
        count: int,
        rng: np.random.Generator,
        x_range: tuple[float, float] = (-10.0, 10.0),
        y_range: tuple[float, float] = (-10.0, 10.0),
    ) -> LandmarkWorld:
        xs = rng.uniform(*x_range, size=count)
        ys = rng.uniform(*y_range, size=count)
        return cls(
            {
                LandmarkId(i): Point2[WorldFrame](float(x), float(y))
                for i, (x, y) in enumerate(zip(xs, ys))
            }
        )

    @classmethod
    def ring(
        cls, count: int, radius: float, center: Point2[WorldFrame] | None = None
    ) -> LandmarkWorld:
        c = center if center is not None else Point2[WorldFrame](0.0, 0.0)
        return cls(
            {
                LandmarkId(i): Point2[WorldFrame](
                    c.x + radius * math.cos(2.0 * math.pi * i / count),
                    c.y + radius * math.sin(2.0 * math.pi * i / count),
                )
                for i in range(count)
            }
        )

    @property
    def landmarks(self) -> Mapping[LandmarkId, Point2[WorldFrame]]:
        return self._landmarks

    def __iter__(self) -> Iterator[tuple[LandmarkId, Point2[WorldFrame]]]:
        return iter(self._landmarks.items())

    def __len__(self) -> int:
        return len(self._landmarks)
