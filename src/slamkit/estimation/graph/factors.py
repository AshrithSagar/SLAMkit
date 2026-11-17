"""
Factors
=======
Probabilistic constraints between graph variables.

A factor is a residual ``r(x)`` with Gaussian noise ``Σ``; the graph minimises
``½ Σ_f ‖r_f(x)‖²_Σf``. Each concrete factor implements only ``evaluate``
(its raw residual and right-perturbation Jacobians); whitening and
linearisation are shared.

The three factor types needed for landmark SLAM reuse the very same building
blocks the filters use:

* ``PosePriorFactor``          : anchors the gauge (the first pose);
* ``BetweenFactor``            : a motion model's increment ``Δ(u)`` and ``Q(u)``;
* ``LandmarkObservationFactor``: an ``Observation``'s own ``linearize``.
"""
# src/slamkit/estimation/graph/factors.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

from slamkit.core.arrays import Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.estimation.graph.values import Key, LandmarkKey, PoseKey, Values
from slamkit.geometry.se2 import SE2
from slamkit.probability.gaussian import GaussianNoise
from slamkit.sensing.observation import Observation

__all__ = [
    "BetweenFactor",
    "Factor",
    "LandmarkObservationFactor",
    "LinearizedFactor",
    "PosePriorFactor",
]


@dataclass(frozen=True, slots=True)
class LinearizedFactor:
    """Whitened ``r + Σ_k J_k δ_k`` at the current values."""

    keys: tuple[Key, ...]
    residual: Vector
    jacobians: tuple[Matrix, ...]


class Factor(ABC):
    """A residual over ``keys`` with Gaussian ``noise``."""

    def __init__(self, keys: tuple[Key, ...], noise: GaussianNoise) -> None:
        self._keys = keys
        self._noise = noise

    @property
    def keys(self) -> tuple[Key, ...]:
        return self._keys

    @property
    def noise(self) -> GaussianNoise:
        return self._noise

    @abstractmethod
    def evaluate(self, values: Values) -> tuple[Vector, tuple[Matrix, ...]]:
        """Unwhitened residual and its Jacobians w.r.t. each key, in ``keys`` order."""

    def linearize(self, values: Values) -> LinearizedFactor:
        residual, jacobians = self.evaluate(values)
        return LinearizedFactor(
            keys=self._keys,
            residual=self._noise.whiten(residual),
            jacobians=tuple(self._noise.whiten_jacobian(j) for j in jacobians),
        )

    def error(self, values: Values) -> float:
        """``½ ‖r‖²_Σ``."""
        residual, _ = self.evaluate(values)
        whitened = self._noise.whiten(residual)
        return 0.5 * float(whitened @ whitened)


class PosePriorFactor(Factor):
    """``r = x ⊖ x̄``: a Gaussian prior on one pose."""

    def __init__(
        self, key: PoseKey, prior: SE2[WorldFrame, BodyFrame], noise: GaussianNoise
    ) -> None:
        super().__init__((key,), noise)
        self._key = key
        self._prior = prior

    def evaluate(self, values: Values) -> tuple[Vector, tuple[Matrix, ...]]:
        residual = values[self._key].ominus(self._prior)
        return residual, (Matrix(SE2.right_jacobian_inverse(residual)),)


class BetweenFactor(Factor):
    """``r = (xᵢ⁻¹ ∘ xⱼ) ⊖ Δ``: a measured relative pose between two timesteps."""

    def __init__(
        self,
        first: PoseKey,
        second: PoseKey,
        measured: SE2[BodyFrame, BodyFrame],
        noise: GaussianNoise,
    ) -> None:
        super().__init__((first, second), noise)
        self._first = first
        self._second = second
        self._measured = measured

    def evaluate(self, values: Values) -> tuple[Vector, tuple[Matrix, ...]]:
        xi, xj = values[self._first], values[self._second]
        residual = xi.between(xj).ominus(self._measured)
        jr_inv = SE2.right_jacobian_inverse(residual)
        wrt_first = -(jr_inv @ xj.between(xi).adjoint())
        return residual, (Matrix(wrt_first), Matrix(jr_inv))


class LandmarkObservationFactor(Factor):
    """``r = h(x, l) ⊖ z``: one observation of one landmark from one pose."""

    def __init__(
        self, pose: PoseKey, landmark: LandmarkKey, observation: Observation
    ) -> None:
        super().__init__((pose, landmark), GaussianNoise(observation.noise_covariance))
        self._pose = pose
        self._landmark = landmark
        self._observation = observation

    def evaluate(self, values: Values) -> tuple[Vector, tuple[Matrix, ...]]:
        linearization = self._observation.linearize(
            values[self._pose], values[self._landmark]
        )
        return Vector(-linearization.innovation), (
            linearization.wrt_pose,
            linearization.wrt_landmark,
        )
