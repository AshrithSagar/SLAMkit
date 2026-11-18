"""
Geometry
========
Manifold-valued, frame-typed geometric primitives.
"""
# src/slamkit/geometry/__init__.py

from slamkit.geometry.manifold import Manifold
from slamkit.geometry.mean import ManifoldMean
from slamkit.geometry.numerical import NumericalJacobian
from slamkit.geometry.point import Point2
from slamkit.geometry.ray import Ray
from slamkit.geometry.se2 import SE2, TransformedPoint

__all__ = [
    "SE2",
    "Manifold",
    "ManifoldMean",
    "NumericalJacobian",
    "Point2",
    "Ray",
    "TransformedPoint",
]
