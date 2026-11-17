"""
Covariance
==========
The handful of covariance manipulations every estimator repeats.
"""
# src/slamkit/probability/covariance.py

from typing import final

import numpy as np

from slamkit.core.arrays import Matrix
from slamkit.core.namespace import Namespace

__all__ = [
    "Covariance",
]


@final
class Covariance(Namespace):
    """Namespace of covariance operations."""

    @staticmethod
    def symmetrize(covariance: Matrix) -> Matrix:
        """Remove the antisymmetric part accumulated by floating-point error."""
        return Matrix(0.5 * (covariance + covariance.T))

    @staticmethod
    def propagate(jacobian: Matrix, covariance: Matrix) -> Matrix:
        """First-order propagation through a linear(ised) map: ``J Σ Jᵀ``."""
        return Matrix(jacobian @ covariance @ jacobian.T)

    @staticmethod
    def diagonal(*standard_deviations: float) -> Matrix:
        return Matrix(
            np.diag(np.square(np.asarray(standard_deviations, dtype=np.float64)))
        )

    @staticmethod
    def validate(covariance: Matrix, dim: int) -> None:
        if covariance.shape != (dim, dim):
            raise ValueError(
                f"expected a {dim}x{dim} covariance, got shape {covariance.shape}"
            )
        if float(np.max(np.abs(covariance - covariance.T), initial=0.0)) > 1e-9:
            raise ValueError("covariance must be symmetric")
