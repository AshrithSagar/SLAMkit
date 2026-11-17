"""
Gaussians
=========
Multivariate normal distributions in the three roles SLAM uses them:

* ``Gaussian``      : a belief or a noise distribution (mean + covariance);
* ``Innovation``    : a residual paired with its predicted covariance, i.e.
                      the statistic data association gates on and particle
                      filters weight by;
* ``GaussianNoise`` : a noise model in *square-root information* form, i.e. the
                      whitening that turns a weighted least-squares factor into
                      an ordinary one.
"""
# src/slamkit/probability/gaussian.py

import math
from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Matrix, Vector
from slamkit.probability.covariance import Covariance

__all__ = [
    "Gaussian",
    "GaussianNoise",
    "Innovation",
]


@dataclass(frozen=True, slots=True)
class Gaussian:
    """``N(mean, covariance)`` over a vector space."""

    mean: Vector
    covariance: Matrix

    def __post_init__(self) -> None:
        Covariance.validate(self.covariance, self.dim)

    @classmethod
    def zero_mean(cls, covariance: Matrix) -> Gaussian:
        return cls(Vector(np.zeros(covariance.shape[0])), covariance)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def sample(self, rng: np.random.Generator) -> Vector:
        return Vector(rng.multivariate_normal(self.mean, self.covariance))


@dataclass(frozen=True, slots=True)
class Innovation:
    """A measurement residual ``ν = z ⊖ ẑ`` with its predicted covariance ``S``."""

    residual: Vector
    covariance: Matrix

    def __post_init__(self) -> None:
        Covariance.validate(self.covariance, self.dim)

    @property
    def dim(self) -> int:
        return self.residual.shape[0]

    def mahalanobis_squared(self) -> float:
        """``νᵀ S⁻¹ ν``; χ²-distributed with ``dim`` dofs if the model is right."""
        return float(self.residual @ np.linalg.solve(self.covariance, self.residual))

    def log_likelihood(self) -> float:
        """``log N(ν; 0, S)``."""
        _, log_det = np.linalg.slogdet(self.covariance)
        return -0.5 * (
            self.mahalanobis_squared()
            + float(log_det)
            + self.dim * math.log(2.0 * math.pi)
        )


class GaussianNoise:
    """Zero-mean Gaussian noise held as its square-root information matrix.

    With ``Σ⁻¹ = L Lᵀ``, ``‖r‖²_Σ = ‖Lᵀ r‖²``: whitening by ``Lᵀ`` turns the
    Mahalanobis cost into a plain squared norm, which is what least-squares
    solvers minimise.
    """

    def __init__(self, covariance: Matrix) -> None:
        Covariance.validate(covariance, covariance.shape[0])
        self._covariance = covariance
        self._sqrt_information_t = Matrix(
            np.linalg.cholesky(np.linalg.inv(covariance)).T
        )

    @classmethod
    def from_standard_deviations(cls, *standard_deviations: float) -> GaussianNoise:
        return cls(Covariance.diagonal(*standard_deviations))

    @property
    def dim(self) -> int:
        return self._covariance.shape[0]

    @property
    def covariance(self) -> Matrix:
        return self._covariance

    def whiten(self, residual: Vector) -> Vector:
        return Vector(self._sqrt_information_t @ residual)

    def whiten_jacobian(self, jacobian: Matrix) -> Matrix:
        return Matrix(self._sqrt_information_t @ jacobian)
