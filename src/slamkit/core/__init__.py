"""
Core
====
Primitive, dependency-free building blocks: typed arrays, frame tags, ids, angles.
"""
# src/slamkit/core/__init__.py

from slamkit.core.angles import Angle
from slamkit.core.arrays import Float64, Mat2, Mat2x3, Mat3, Matrix, Vec2, Vec3, Vector
from slamkit.core.frames import BodyFrame, Frame, SensorFrame, WorldFrame
from slamkit.core.ids import LandmarkId, LandmarkIdAllocator
from slamkit.core.namespace import Namespace

__all__ = [
    "Angle",
    "BodyFrame",
    "Float64",
    "Frame",
    "LandmarkId",
    "LandmarkIdAllocator",
    "Mat2",
    "Mat2x3",
    "Mat3",
    "Matrix",
    "Namespace",
    "SensorFrame",
    "Vec2",
    "Vec3",
    "Vector",
    "WorldFrame",
]
