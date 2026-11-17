"""
SE2
===
Rigid 2D transforms, typed by the frames they map between.

``SE2[To, From]`` maps coordinates expressed in ``From`` into ``To``. The robot
pose is therefore ``SE2[WorldFrame, BodyFrame]`` and a sensor's extrinsic
calibration is ``SE2[BodyFrame, SensorFrame]``. Composition and inversion keep
track of the frames statically::

    pose: SE2[WorldFrame, BodyFrame]
    extrinsic: SE2[BodyFrame, SensorFrame]
    sensor_pose = pose @ extrinsic          # SE2[WorldFrame, SensorFrame]
    extrinsic @ pose                        # type error: frames don't chain

Lie-group conventions (following Solà et al., "A micro Lie theory"):

* tangent vectors are ``τ = (ρx, ρy, φ)``;
* perturbations are on the *right*: ``T ⊕ δ = T ∘ Exp(δ)``, so uncertainty is
  expressed in the body (``From``) frame of the transform;
* every Jacobian in this module is with respect to that right perturbation.
"""
# src/slamkit/geometry/se2.py

import math
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np

from slamkit.core.angles import Angle
from slamkit.core.arrays import Mat2, Mat2x3, Mat3, Vec2, Vec3, Vector
from slamkit.core.frames import Frame
from slamkit.geometry.manifold import Manifold
from slamkit.geometry.point import Point2

__all__ = [
    "SE2",
    "TransformedPoint",
]

_SKEW: Mat2 = Mat2([[0.0, -1.0], [1.0, 0.0]])
"""The generator of 2D rotations: ``d/dθ R(θ) = R(θ) @ _SKEW``."""


@dataclass(frozen=True, slots=True)
class TransformedPoint[F: Frame]:
    """A transformed point together with its first-order sensitivities."""

    point: Point2[F]
    wrt_transform: Mat2x3
    """∂ point / ∂ δ, for a right perturbation ``δ`` of the transform."""
    wrt_point: Mat2
    """∂ point / ∂ (input point)."""


@dataclass(frozen=True, slots=True)
class SE2[To: Frame, From: Frame](Manifold["SE2[To, From]"]):
    """A rigid transform ``(x, y, θ)`` from frame ``From`` into frame ``To``.

    ``θ`` is normalised to ``[-π, π)`` on construction.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    DOF: ClassVar[int] = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", Angle.wrap(self.theta))

    # ── Construction / conversion ─────────────────────────────────────────────

    @classmethod
    def identity(cls) -> SE2[To, From]:
        return cls()

    @classmethod
    def from_vector(cls, vector: Vector) -> SE2[To, From]:
        """From ``[x, y, θ]`` (note: *not* a tangent vector; see ``exp``)."""
        if vector.shape != (3,):
            raise ValueError(f"SE2 expects a 3-vector, got shape {vector.shape}")
        return cls(float(vector[0]), float(vector[1]), float(vector[2]))

    def to_vector(self) -> Vec3:
        return Vec3([self.x, self.y, self.theta])

    @property
    def translation(self) -> Vec2:
        return Vec2([self.x, self.y])

    @property
    def rotation(self) -> Mat2:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Mat2([[c, -s], [s, c]])

    # ── Group operations ──────────────────────────────────────────────────────

    def compose[X: Frame](self, other: SE2[From, X]) -> SE2[To, X]:
        """``self ∘ other``: first apply ``other`` (X → From), then ``self`` (From → To)."""
        t = self.rotation @ other.translation + self.translation
        return SE2[To, X](float(t[0]), float(t[1]), self.theta + other.theta)

    def __matmul__[X: Frame](self, other: SE2[From, X]) -> SE2[To, X]:
        return self.compose(other)

    def inverse(self) -> SE2[From, To]:
        t = -(self.rotation.T @ self.translation)
        return SE2[From, To](float(t[0]), float(t[1]), -self.theta)

    def between[X: Frame](self, other: SE2[To, X]) -> SE2[From, X]:
        """``self⁻¹ ∘ other``: ``other`` expressed relative to ``self``."""
        return self.inverse().compose(other)

    # ── Group actions on points ───────────────────────────────────────────────

    def transform(self, point: Point2[From]) -> Point2[To]:
        return self.transform_with_jacobians(point).point

    def transform_with_jacobians(self, point: Point2[From]) -> TransformedPoint[To]:
        """``q = R p + t`` with ``∂q/∂δ = R [I | J p]`` and ``∂q/∂p = R``."""
        p = point.to_vector()
        rotation = self.rotation
        q = rotation @ p + self.translation
        wrt_transform = rotation @ np.column_stack((np.eye(2), _SKEW @ p))
        return TransformedPoint[To](
            point=Point2[To](float(q[0]), float(q[1])),
            wrt_transform=Mat2x3(wrt_transform),
            wrt_point=rotation,
        )

    def inverse_transform(self, point: Point2[To]) -> Point2[From]:
        return self.inverse_transform_with_jacobians(point).point

    def inverse_transform_with_jacobians(
        self, point: Point2[To]
    ) -> TransformedPoint[From]:
        """``p = Rᵀ (q - t)`` with ``∂p/∂δ = [-I | -J p]`` and ``∂p/∂q = Rᵀ``."""
        rotation_t = self.rotation.T
        p = rotation_t @ (point.to_vector() - self.translation)
        wrt_transform = np.column_stack((-np.eye(2), -(_SKEW @ p)))
        return TransformedPoint[From](
            point=Point2[From](float(p[0]), float(p[1])),
            wrt_transform=Mat2x3(wrt_transform),
            wrt_point=Mat2(rotation_t),
        )

    # ── Lie algebra ───────────────────────────────────────────────────────────

    @classmethod
    def exp(cls, tau: Vector) -> SE2[To, From]:
        """Exponential map ``se(2) → SE(2)`` of ``τ = (ρx, ρy, φ)``."""
        rho, phi = tau[:2], float(tau[2])
        t = cls._v_matrix(phi) @ rho
        return cls(float(t[0]), float(t[1]), phi)

    def log(self) -> Vec3:
        """Logarithmic map ``SE(2) → se(2)``; inverse of ``exp``."""
        rho = np.linalg.solve(self._v_matrix(self.theta), self.translation)
        return Vec3([float(rho[0]), float(rho[1]), self.theta])

    def adjoint(self) -> Mat3:
        """``Ad_T``, with ``T ∘ Exp(δ) = Exp(Ad_T δ) ∘ T``."""
        ad = np.eye(3)
        ad[:2, :2] = self.rotation
        ad[:2, 2] = [self.y, -self.x]
        return Mat3(ad)

    @staticmethod
    def right_jacobian(tau: Vector) -> Mat3:
        """``J_r(τ)``, with ``Exp(τ + δ) ≈ Exp(τ) ∘ Exp(J_r(τ) δ)``."""
        r1, r2, phi = float(tau[0]), float(tau[1]), float(tau[2])
        a, b = Angle.sinc(phi), Angle.cosc(phi)
        c, d = Angle.sine_defect_quotient(phi), Angle.versine_quotient(phi)
        jr = np.eye(3)
        jr[0, :] = [a, b, r1 * c - r2 * d]
        jr[1, :] = [-b, a, r1 * d + r2 * c]
        return Mat3(jr)

    @staticmethod
    def right_jacobian_inverse(tau: Vector) -> Mat3:
        return Mat3(np.linalg.inv(SE2.right_jacobian(tau)))

    @staticmethod
    def _v_matrix(phi: float) -> Mat2:
        a, b = Angle.sinc(phi), Angle.cosc(phi)
        return Mat2([[a, -b], [b, a]])

    # ── Manifold ──────────────────────────────────────────────────────────────

    def oplus(self, delta: Vector) -> Self:
        increment = SE2[From, From].exp(delta)
        t = self.rotation @ increment.translation + self.translation
        return type(self)(float(t[0]), float(t[1]), self.theta + increment.theta)

    def ominus(self, other: SE2[To, From]) -> Vector:
        return Vector(other.between(self).log())
