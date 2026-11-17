"""
Arrays
======
Shape-typed array aliases, built on ``typingkit``.

Every alias is *callable*: ``Vec3(values)`` constructs the array and validates
its shape and dtype at runtime, while the alias itself narrows the static type.
That is the single idiom used across slamkit for crossing from "some numpy
expression" back into a typed value::

    def predict(self) -> Vec3:
        return Vec3(self.rotation @ v + t)   # checked statically *and* at runtime

Fixed-size aliases (``Vec2``, ``Mat2x3``, ...) cover the 2D geometry, whose
dimensions are known up front. ``Vector`` / ``Matrix`` are the rank-checked,
size-relaxed escape hatch for quantities whose size is only known at runtime
(joint SLAM states that grow with the map, measurement spaces of arbitrary
dimension). A fixed alias is assignable to the dynamic one, never the reverse.
"""
# src/slamkit/core/arrays.py

# ruff: noqa: UP040

from typing import Literal, TypeAlias

import numpy as np
from typingkit.numpy import TypedNDArray

__all__ = [
    "Float64",
    "Mat2",
    "Mat2x3",
    "Mat3",
    "Matrix",
    "Vec2",
    "Vec3",
    "Vector",
]

Float64: TypeAlias = np.dtype[np.float64]

# ── Dynamically-sized ────────────────────────────────────────────────────────
Vector: TypeAlias = TypedNDArray[tuple[int], Float64]
Matrix: TypeAlias = TypedNDArray[tuple[int, int], Float64]

# ── Fixed-size (2D geometry) ─────────────────────────────────────────────────
Vec2: TypeAlias = TypedNDArray[tuple[Literal[2]], Float64]
Vec3: TypeAlias = TypedNDArray[tuple[Literal[3]], Float64]
Mat2: TypeAlias = TypedNDArray[tuple[Literal[2], Literal[2]], Float64]
Mat3: TypeAlias = TypedNDArray[tuple[Literal[3], Literal[3]], Float64]
Mat2x3: TypeAlias = TypedNDArray[tuple[Literal[2], Literal[3]], Float64]
