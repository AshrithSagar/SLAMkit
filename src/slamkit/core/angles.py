"""
Angles
======
Scalar functions on angles, grouped as a non-instantiable namespace.
"""
# src/slamkit/core/angles.py

import math
from typing import Final, final

from slamkit.core.namespace import Namespace

__all__ = [
    "Angle",
]


@final
class Angle(Namespace):
    """Angle utilities: wrapping and the numerically-stable small-angle series
    that SE(2)'s exponential map and Jacobians are built from."""

    SMALL: Final[float] = 1e-6
    """Below this magnitude, ``sin(θ)/θ`` switches to its Taylor series."""

    SERIES: Final[float] = 1e-2
    """Below this magnitude, ``(θ - sin θ)/θ²`` (which cancels catastrophically in
    closed form) switches to its Taylor series, accurate to ~1e-15 here."""

    @staticmethod
    def wrap(theta: float) -> float:
        """Wrap to the half-open interval ``[-π, π)``."""
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def sinc(theta: float) -> float:
        """``sin(θ) / θ``, continuous at 0."""
        if abs(theta) < Angle.SMALL:
            return 1.0 - theta * theta / 6.0
        return math.sin(theta) / theta

    @staticmethod
    def cosc(theta: float) -> float:
        """``(1 - cos(θ)) / θ``, continuous at 0."""
        return theta * Angle.versine_quotient(theta)

    @staticmethod
    def versine_quotient(theta: float) -> float:
        """``(1 - cos(θ)) / θ²``, continuous at 0 (→ ½).

        Uses ``1 - cos θ = 2 sin²(θ/2)``, which has no cancellation.
        """
        half = 0.5 * Angle.sinc(0.5 * theta)
        return 2.0 * half * half

    @staticmethod
    def sine_defect_quotient(theta: float) -> float:
        """``(θ - sin(θ)) / θ²``, continuous at 0 (→ 0)."""
        if abs(theta) < Angle.SERIES:
            t2 = theta * theta
            return theta * (1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0)
        return (theta - math.sin(theta)) / (theta * theta)
