"""
Manifold
========
The one abstraction every estimated or measured quantity in slamkit shares.

SLAM states are not vectors: headings wrap, rotations compose, bearings live on
a circle. Instead of special-casing that inside each estimator, every value
type exposes the two operators that *locally* turn it into a vector space:

    x ⊕ δ   (``oplus``)   : move from ``x`` along tangent vector ``δ``
    y ⊖ x   (``ominus``)  : the tangent vector ``δ`` with ``x ⊕ δ = y``

EKF corrections, Gauss-Newton steps, particle noise, innovations and numerical
Jacobians are all written once against these two operators and so work
unchanged for poses, landmarks and measurements alike.
"""
# src/slamkit/geometry/manifold.py

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from slamkit.core.arrays import Vector

__all__ = [
    "Manifold",
]


class Manifold[T](ABC):
    """A value with a local vector-space structure of dimension ``DOF``.

    ``T`` is the implementing type itself (F-bounded), e.g.
    ``class Point2[F](Manifold["Point2[F]"])``. That is what lets ``ominus``
    demand an argument of *exactly* the same type, frames included -- a plain
    ``Self``-typed parameter is rejected by pyright as an unsafe override.
    """

    __slots__ = ()

    DOF: ClassVar[int]
    """Dimension of the tangent space (degrees of freedom)."""

    @abstractmethod
    def oplus(self, delta: Vector) -> Self:
        """``self ⊕ delta``: retract a tangent vector onto the manifold."""

    @abstractmethod
    def ominus(self, other: T) -> Vector:
        """``self ⊖ other``: the tangent vector ``δ`` such that ``other ⊕ δ == self``."""
