import math

import numpy as np
import pytest

from slamkit.core import Angle, BodyFrame, SensorFrame, Vector, WorldFrame
from slamkit.geometry import SE2, NumericalJacobian, Point2

POSE = SE2[WorldFrame, BodyFrame](1.0, -2.0, 0.7)
TANGENTS = [
    Vector([0.3, -0.2, 1.1]),
    Vector([1.0, 2.0, -2.5]),
    Vector([0.4, 0.1, 1e-9]),  # small-angle branch
]


class TestAngle:
    @pytest.mark.parametrize(
        "theta", [0.0, math.pi, -math.pi, 3 * math.pi, -7.5, 100.0]
    )
    def test_wrap_lands_in_half_open_interval(self, theta: float) -> None:
        wrapped = Angle.wrap(theta)
        assert -math.pi <= wrapped < math.pi
        assert math.isclose(math.cos(wrapped), math.cos(theta), abs_tol=1e-9)

    def test_namespace_is_not_instantiable(self) -> None:
        with pytest.raises(TypeError):
            Angle()


class TestSE2Group:
    def test_compose_with_inverse_is_identity(self) -> None:
        identity = POSE @ POSE.inverse()
        assert np.allclose(identity.to_vector(), 0.0, atol=1e-12)

    def test_transform_and_inverse_transform_round_trip(self) -> None:
        point = Point2[BodyFrame](0.5, -1.2)
        assert np.allclose(
            POSE.inverse_transform(POSE.transform(point)).to_vector(), point.to_vector()
        )

    def test_composition_chains_frames(self) -> None:
        extrinsic = SE2[BodyFrame, SensorFrame](0.2, 0.1, -0.4)
        point = Point2[SensorFrame](3.0, 1.0)
        chained = (POSE @ extrinsic).transform(point)
        assert np.allclose(
            chained.to_vector(), POSE.transform(extrinsic.transform(point)).to_vector()
        )

    @pytest.mark.parametrize("tau", TANGENTS)
    def test_exp_log_round_trip(self, tau: Vector) -> None:
        assert np.allclose(SE2[BodyFrame, BodyFrame].exp(tau).log(), tau)

    @pytest.mark.parametrize("tau", TANGENTS)
    def test_oplus_ominus_round_trip(self, tau: Vector) -> None:
        assert np.allclose(POSE.oplus(tau).ominus(POSE), tau)

    def test_adjoint_moves_perturbation_from_right_to_left(self) -> None:
        delta = Vector([0.1, 0.2, -0.3])
        right = POSE @ SE2[BodyFrame, BodyFrame].exp(delta)
        left = SE2[WorldFrame, WorldFrame].exp(Vector(POSE.adjoint() @ delta)) @ POSE
        assert np.allclose(right.to_vector(), left.to_vector())


class TestSE2Jacobians:
    jacobian = NumericalJacobian()

    @pytest.mark.parametrize("tau", TANGENTS)
    def test_right_jacobian(self, tau: Vector) -> None:
        # J_r is ∂/∂τ of Exp(τ) under right perturbation: a Jacobian *from the
        # tangent space* (a Euclidean Point2-like 3-vector) *to* SE(2).
        base = SE2[BodyFrame, BodyFrame].exp(tau)
        h = 1e-6
        columns: list[Vector] = []
        for i in range(3):
            d = np.zeros(3)
            d[i] = h
            plus = SE2[BodyFrame, BodyFrame].exp(Vector(tau + d)).ominus(base)
            minus = SE2[BodyFrame, BodyFrame].exp(Vector(tau - d)).ominus(base)
            columns.append(Vector((plus - minus) / (2 * h)))
        assert np.allclose(np.column_stack(columns), SE2.right_jacobian(tau), atol=1e-6)

    def test_transform_jacobians(self) -> None:
        point = Point2[BodyFrame](0.5, -1.2)
        result = POSE.transform_with_jacobians(point)
        assert np.allclose(
            self.jacobian(lambda t: t.transform(point), POSE),
            result.wrt_transform,
            atol=1e-6,
        )
        assert np.allclose(
            self.jacobian(lambda p: POSE.transform(p), point),
            result.wrt_point,
            atol=1e-6,
        )

    def test_inverse_transform_jacobians(self) -> None:
        point = Point2[WorldFrame](3.0, 1.0)
        result = POSE.inverse_transform_with_jacobians(point)
        assert np.allclose(
            self.jacobian(lambda t: t.inverse_transform(point), POSE),
            result.wrt_transform,
            atol=1e-6,
        )
        assert np.allclose(
            self.jacobian(lambda p: POSE.inverse_transform(p), point),
            result.wrt_point,
            atol=1e-6,
        )
