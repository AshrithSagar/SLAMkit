import numpy as np
import pytest

from slamkit.core import BodyFrame, LandmarkId, Matrix, Vector, WorldFrame
from slamkit.estimation.graph import (
    BetweenFactor,
    Factor,
    FactorGraph,
    GaussNewton,
    Key,
    LandmarkKey,
    LandmarkObservationFactor,
    LevenbergMarquardt,
    Optimizer,
    Ordering,
    PoseKey,
    PosePriorFactor,
    Values,
)
from slamkit.geometry import SE2, Point2
from slamkit.probability import GaussianNoise
from slamkit.sensing import (
    PointObservation,
    PointSensor,
    RangeBearing,
    RangeBearingModel,
)

X0, X1, L0 = PoseKey(0), PoseKey(1), LandmarkKey(LandmarkId(0))
NOISE3 = GaussianNoise.from_standard_deviations(0.1, 0.1, 0.05)
SENSOR = PointSensor("rb", RangeBearingModel(0.1, 0.02))


def make_values() -> Values:
    values = Values()
    values.set_pose(X0, SE2[WorldFrame, BodyFrame](0.1, -0.2, 0.3))
    values.set_pose(X1, SE2[WorldFrame, BodyFrame](1.3, 0.4, 1.1))
    values.set_landmark(L0, Point2[WorldFrame](3.0, 2.0))
    return values


FACTORS: list[Factor] = [
    PosePriorFactor(X0, SE2[WorldFrame, BodyFrame](0.0, 0.0, 0.2), NOISE3),
    BetweenFactor(X0, X1, SE2[BodyFrame, BodyFrame](1.0, 0.5, 0.7), NOISE3),
    LandmarkObservationFactor(X1, L0, PointObservation(SENSOR, RangeBearing(2.0, 0.1))),
]


class TestFactors:
    @pytest.mark.parametrize("factor", FACTORS, ids=lambda f: type(f).__name__)
    def test_jacobians_match_finite_differences(self, factor: Factor) -> None:
        values = make_values()
        _, jacobians = factor.evaluate(values)
        for key, analytic in zip(factor.keys, jacobians, strict=True):
            assert np.allclose(self._numeric(factor, values, key), analytic, atol=1e-5)

    @staticmethod
    def _numeric(factor: Factor, values: Values, key: Key, h: float = 1e-6) -> Matrix:
        columns: list[Vector] = []
        for i in range(key.DOF):
            d = np.zeros(key.DOF)
            d[i] = h
            plus, _ = factor.evaluate(values.retract({key: Vector(d)}))
            minus, _ = factor.evaluate(values.retract({key: Vector(-d)}))
            columns.append(Vector((plus - minus) / (2 * h)))
        return Matrix(np.column_stack(columns))


class TestOrdering:
    def test_blocks_are_contiguous(self) -> None:
        ordering = Ordering.of(make_values())
        assert ordering.dim == 3 + 3 + 2
        assert ordering.indices(X1, L0) == [3, 4, 5, 6, 7]


@pytest.mark.parametrize(
    "optimizer", [GaussNewton(), LevenbergMarquardt()], ids=["GN", "LM"]
)
class TestOptimizers:
    def test_recovers_consistent_configuration(self, optimizer: Optimizer) -> None:
        truth = SE2[WorldFrame, BodyFrame](1.0, 0.5, 0.7)
        landmark = Point2[WorldFrame](3.0, 2.0)
        graph = FactorGraph(
            [
                PosePriorFactor(X0, SE2[WorldFrame, BodyFrame](), NOISE3),
                BetweenFactor(X0, X1, SE2[BodyFrame, BodyFrame](1.0, 0.5, 0.7), NOISE3),
                LandmarkObservationFactor(
                    X1,
                    L0,
                    PointObservation(
                        SENSOR, SENSOR.predict(truth, landmark).measurement
                    ),
                ),
            ]
        )
        result = optimizer.optimize(graph, make_values())
        assert result.converged
        assert result.final_error < 1e-10
        assert np.allclose(result.values[X1].to_vector(), truth.to_vector(), atol=1e-5)
        assert np.allclose(result.values[L0].to_vector(), landmark.to_vector())
