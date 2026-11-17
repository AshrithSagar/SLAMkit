import math
from typing import Any

import numpy as np
import pytest

from slamkit.core import BodyFrame, WorldFrame
from slamkit.geometry import SE2, NumericalJacobian
from slamkit.motion import (
    Control,
    MotionModel,
    Odometry,
    OdometryMotionModel,
    UnicycleMotionModel,
    VelocityCommand,
)
from slamkit.simulation import CircularPath

POSE = SE2[WorldFrame, BodyFrame](1.0, -2.0, 0.7)

CASES: list[tuple[MotionModel[Any], Control]] = [
    (OdometryMotionModel(), Odometry(0.8, 0.1, 0.3)),
    (UnicycleMotionModel(), VelocityCommand(1.2, 0.4, 0.5)),
]


@pytest.mark.parametrize(("model", "control"), CASES, ids=["odometry", "unicycle"])
class TestMotionModel:
    def test_state_jacobian(
        self, model: MotionModel[Control], control: Control
    ) -> None:
        linearization = model.linearize(POSE, control)
        numeric = NumericalJacobian()(lambda x: model.predict(x, control), POSE)
        assert np.allclose(numeric, linearization.wrt_pose, atol=1e-6)

    def test_noise_covariance_is_positive_definite(
        self, model: MotionModel[Control], control: Control
    ) -> None:
        assert np.all(np.linalg.eigvalsh(model.increment_covariance(control)) > 0.0)

    def test_samples_scatter_with_the_model_covariance(
        self, model: MotionModel[Control], control: Control
    ) -> None:
        rng = np.random.default_rng(0)
        mean = model.predict(POSE, control)
        deltas = np.array(
            [model.sample(POSE, control, rng).ominus(mean) for _ in range(4000)]
        )
        expected = model.increment_covariance(control)
        scale = np.sqrt(
            np.outer(np.diag(expected), np.diag(expected))
        )  # per-entry std scale
        assert np.all(np.abs(np.cov(deltas.T) - expected) <= 0.1 * scale)


class TestPaths:
    PATH = CircularPath(radius=5.0, steps_per_loop=24)

    def test_odometry_lap_closes(self) -> None:
        self._assert_closes(OdometryMotionModel(), self.PATH.odometry())

    def test_velocity_lap_closes(self) -> None:
        self._assert_closes(UnicycleMotionModel(), self.PATH.velocity_commands(dt=0.5))

    @staticmethod
    def _assert_closes[C: Control](model: MotionModel[C], controls: list[C]) -> None:
        pose = SE2[WorldFrame, BodyFrame].identity()
        for control in controls:
            pose = model.predict(pose, control)
        assert math.hypot(pose.x, pose.y) < 1e-9
