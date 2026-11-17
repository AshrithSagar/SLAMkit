"""
Measurements
============
Raw sensor readings as small value types.

A measurement is itself a ``Manifold``: the innovation ``z ⊖ ẑ`` is then just
``ominus``, and any sensor-specific subtlety (bearing wrap-around at ±π) lives
in exactly one place, the measurement type, rather than in every estimator.
"""
# src/slamkit/sensing/measurement.py

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from slamkit.core.angles import Angle
from slamkit.core.arrays import Vector
from slamkit.geometry.manifold import Manifold

__all__ = [
    "Measurement",
    "RangeBearing",
    "RelativePosition",
]


class Measurement[T: "Measurement[Any]"](Manifold[T]):
    """A sensor reading with a vector representation.

    Subclasses provide ``to_vector`` / ``from_vector``; the default manifold
    structure is then plain vector addition, which a subclass overrides only
    where the measurement space is not Euclidean.
    """

    __slots__ = ()

    @abstractmethod
    def to_vector(self) -> Vector: ...

    @classmethod
    @abstractmethod
    def from_vector(cls, vector: Vector) -> Self: ...

    def oplus(self, delta: Vector) -> Self:
        return self.from_vector(Vector(self.to_vector() + delta))

    def ominus(self, other: T) -> Vector:
        return Vector(self.to_vector() - other.to_vector())


@dataclass(frozen=True, slots=True)
class RangeBearing(Measurement["RangeBearing"]):
    """Distance and direction to a point, in the sensor's frame."""

    range: float
    bearing: float

    DOF: ClassVar[int] = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "bearing", Angle.wrap(self.bearing))

    def to_vector(self) -> Vector:
        return Vector([self.range, self.bearing])

    @classmethod
    def from_vector(cls, vector: Vector) -> Self:
        return cls(float(vector[0]), float(vector[1]))

    def ominus(self, other: RangeBearing) -> Vector:
        return Vector(
            [self.range - other.range, Angle.wrap(self.bearing - other.bearing)]
        )


@dataclass(frozen=True, slots=True)
class RelativePosition(Measurement["RelativePosition"]):
    """Cartesian position of a point in the sensor's frame (e.g. a lidar return
    or a stereo-triangulated feature)."""

    x: float
    y: float

    DOF: ClassVar[int] = 2

    def to_vector(self) -> Vector:
        return Vector([self.x, self.y])

    @classmethod
    def from_vector(cls, vector: Vector) -> Self:
        return cls(float(vector[0]), float(vector[1]))
