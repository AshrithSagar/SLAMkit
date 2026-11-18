"""
Chi-square
==========
Quantiles of the χ² distribution, for gating Mahalanobis distances.
"""
# src/slamkit/probability/chi_square.py

import math
from statistics import NormalDist
from typing import final

from slamkit.core.namespace import Namespace

__all__ = [
    "ChiSquare",
]


@final
class ChiSquare(Namespace):
    """χ² quantiles without SciPy.

    Exact for one and two degrees of freedom -- the dimensions of bearing and
    of range-bearing / point innovations -- and the Wilson-Hilferty
    approximation beyond (within ~0.5% at three dofs, improving with more).
    """

    @staticmethod
    def quantile(probability: float, dof: int) -> float:
        """The value ``x`` with ``P(χ²_dof ≤ x) = probability``."""
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie in (0, 1)")
        if dof < 1:
            raise ValueError("dof must be at least 1")
        if dof == 1:  # χ²₁ is a squared standard normal
            return NormalDist().inv_cdf(0.5 + probability / 2.0) ** 2
        if dof == 2:  # χ²₂ is exponential with mean 2
            return -2.0 * math.log1p(-probability)
        z = NormalDist().inv_cdf(probability)
        spread = 2.0 / (9.0 * dof)
        return dof * (1.0 - spread + z * math.sqrt(spread)) ** 3
