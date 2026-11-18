"""
Unscented transform
===================
Deterministic sigma points that capture a Gaussian's mean and covariance.

Sigma points are produced as *tangent-space offsets* ``δᵢ``, not as states:
the caller retracts them onto whatever manifold it is working on
(``x ⊕ δᵢ``). That keeps the transform itself purely Euclidean and lets it
serve poses, landmarks and joint states alike.
"""
# src/slamkit/probability/unscented.py

from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Matrix, Vector
from slamkit.probability.covariance import Covariance

__all__ = [
    "SigmaPoints",
    "UnscentedTransform",
]


@dataclass(frozen=True, slots=True)
class SigmaPoints:
    """``2n + 1`` offsets with their mean and covariance weights."""

    offsets: tuple[Vector, ...]
    mean_weights: Vector
    covariance_weights: Vector

    def __len__(self) -> int:
        return len(self.offsets)

    def weighted_mean(self, samples: Matrix) -> Vector:
        """``Σ wᵐᵢ yᵢ`` for samples stacked as rows."""
        return Vector(self.mean_weights @ samples)

    def cross_covariance(self, left: Matrix, right: Matrix) -> Matrix:
        """``Σ wᶜᵢ aᵢ bᵢᵀ`` for deviations stacked as rows."""
        return Matrix((left * self.covariance_weights[:, None]).T @ right)


class UnscentedTransform:
    """The scaled unscented transform with parameters ``(α, β, κ)``.

    The defaults (``α = 1, β = 2, κ = 0``) place the sigma points at
    ``±√n σ`` with no weight on the centre for the mean, which stays
    well-conditioned for the moderate state sizes SLAM sub-problems have.
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 2.0, kappa: float = 0.0
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

    def sigma_points(self, covariance: Matrix) -> SigmaPoints:
        Covariance.validate(covariance, covariance.shape[0])
        n = covariance.shape[0]
        lam = self._alpha**2 * (n + self._kappa) - n
        scale = n + lam
        if scale <= 0.0:
            raise ValueError("(alpha, kappa) give a non-positive sigma-point spread")
        root = self._matrix_square_root(Matrix(scale * covariance))
        offsets = (
            Vector(np.zeros(n)),
            *(Vector(root[:, i]) for i in range(n)),
            *(Vector(-root[:, i]) for i in range(n)),
        )
        mean_weights = np.full(2 * n + 1, 1.0 / (2.0 * scale))
        mean_weights[0] = lam / scale
        covariance_weights = mean_weights.copy()
        covariance_weights[0] += 1.0 - self._alpha**2 + self._beta
        return SigmaPoints(offsets, Vector(mean_weights), Vector(covariance_weights))

    @staticmethod
    def _matrix_square_root(matrix: Matrix) -> Matrix:
        """A square root ``L`` with ``L Lᵀ = matrix``, via the symmetric
        eigendecomposition (unlike Cholesky, robust to semi-definiteness)."""
        eigenvalues, eigenvectors = np.linalg.eigh(Covariance.symmetrize(matrix))
        return Matrix(eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None)))
