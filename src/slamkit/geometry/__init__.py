"""
Geometry
========
Manifold-valued, frame-typed geometric primitives.
"""
# src/slamkit/geometry/__init__.py

from slamkit.geometry.manifold import Manifold
from slamkit.geometry.numerical import NumericalJacobian
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2, TransformedPoint

__all__ = [
    "SE2",
    "Manifold",
    "NumericalJacobian",
    "Point2",
    "TransformedPoint",
]
