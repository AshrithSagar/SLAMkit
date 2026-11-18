"""
Compare estimators
==================
Runs EKF-, UKF-, FastSLAM 1.0/2.0- and Graph-SLAM through four scenarios --
known correspondence, nearest-neighbour association, joint-compatibility
association, and a bearing-only camera -- and prints their accuracy against
ground truth.

Every run is assembled from the *same* motion model, sensors, associators and
landmark initialiser; only the estimator differs.
"""
# examples/compare_estimators.py

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from slamkit.association import (
    DataAssociator,
    JointCompatibility,
    KnownCorrespondence,
    NearestNeighbor,
)
from slamkit.core import BodyFrame, SensorFrame
from slamkit.estimation import EkfSlam, FastSlam, GraphSlam, SlamEstimator, UkfSlam
from slamkit.estimation.particle import ObservationProposal
from slamkit.evaluation import Evaluator
from slamkit.geometry import SE2
from slamkit.motion import MotionModel, Odometry, OdometryMotionModel
from slamkit.pipeline import Dataset, GroundTruth, SlamRunner
from slamkit.sensing import (
    BearingModel,
    CartesianModel,
    PointSensor,
    RangeBearingModel,
    SensingRange,
    Sensor,
)
from slamkit.simulation import CircularPath, LandmarkWorld, Simulator

type EstimatorFactory = Callable[
    [MotionModel[Odometry], DataAssociator], SlamEstimator[Odometry]
]


@dataclass(frozen=True, slots=True)
class Experiment:
    """One estimator, run on a scenario's dataset with its associator."""

    name: str
    factory: EstimatorFactory

    def run(
        self,
        dataset: Dataset[Odometry],
        truth: GroundTruth,
        motion_model: MotionModel[Odometry],
        associator: DataAssociator,
        evaluator: Evaluator,
    ) -> str:
        result = SlamRunner(
            self.factory(motion_model, associator), runtime_checks=False
        ).run(dataset)
        report = evaluator.evaluate(result, truth)
        return f"{self.name:<13} {result.elapsed_seconds:6.2f}s | {report}"


@dataclass(frozen=True, slots=True)
class Scenario:
    """Which sensors the robot carries, and how landmarks are identified."""

    title: str
    sensors: Sequence[Sensor]
    tagged: bool
    associator: Callable[[], DataAssociator]
    evaluator: Evaluator


class Demo:
    SEED = 7

    def __init__(self) -> None:
        self._rng = np.random.default_rng(self.SEED)
        self._motion_model = OdometryMotionModel(
            translation_noise=0.05, rotation_noise=0.05, drift_per_meter=0.02
        )
        lidar = PointSensor(
            "lidar",
            RangeBearingModel(range_std=0.15, bearing_std=math.radians(2.0)),
            sensing_range=SensingRange(
                max_range=8.0, field_of_view=math.radians(180.0)
            ),
        )
        stereo = PointSensor(
            "stereo",
            CartesianModel(x_std=0.2, y_std=0.2),
            extrinsic=SE2[BodyFrame, SensorFrame](0.3, 0.0, math.radians(-90.0)),
            sensing_range=SensingRange(max_range=5.0, field_of_view=math.radians(90.0)),
        )
        camera = PointSensor(
            "camera",
            BearingModel(bearing_std=math.radians(1.0)),
            sensing_range=SensingRange(
                max_range=10.0, field_of_view=math.radians(360.0)
            ),
        )
        unknown = Evaluator(match_radius=1.5)
        self._scenarios = (
            Scenario(
                "known correspondence",
                (lidar, stereo),
                True,
                KnownCorrespondence,
                Evaluator(),
            ),
            Scenario(
                "nearest-neighbour association",
                (lidar, stereo),
                False,
                NearestNeighbor,
                unknown,
            ),
            Scenario(
                "joint-compatibility association",
                (lidar, stereo),
                False,
                JointCompatibility,
                unknown,
            ),
            Scenario(
                "omnidirectional bearing-only camera (delayed init)",
                (camera,),
                True,
                KnownCorrespondence,
                Evaluator(),
            ),
        )
        self._world = LandmarkWorld.uniform(
            15, self._rng, x_range=(-12.0, 12.0), y_range=(-4.0, 20.0)
        )
        self._path = CircularPath(radius=8.0, steps_per_loop=40, loops=2)
        self._experiments = (
            Experiment("EKF", lambda m, a: EkfSlam(m, a)),
            Experiment("UKF", lambda m, a: UkfSlam(m, a)),
            Experiment(
                "FastSLAM 1.0",
                lambda m, a: FastSlam(
                    m, a, np.random.default_rng(self.SEED), num_particles=20
                ),
            ),
            Experiment(
                "FastSLAM 2.0",
                lambda m, a: FastSlam(
                    m,
                    a,
                    np.random.default_rng(self.SEED),
                    num_particles=20,
                    proposal=ObservationProposal(),
                ),
            ),
            Experiment("GraphSLAM", lambda m, a: GraphSlam(m, a, optimize_every=20)),
        )

    def run(self) -> None:
        for scenario in self._scenarios:
            dataset = Simulator(
                self._world,
                self._motion_model,
                scenario.sensors,
                np.random.default_rng(self.SEED),
                tag_observations=scenario.tagged,
            ).run(self._path.odometry())
            assert dataset.ground_truth is not None
            print(
                f"\n── {scenario.title} ── "
                + f"({len(dataset)} steps, {len(self._world)} landmarks)"
            )
            print(
                f"{'odometry':<13} {'':>7} | "
                + f"final drift {self._dead_reckoning(dataset):6.3f} m"
            )
            for experiment in self._experiments:
                print(
                    experiment.run(
                        dataset,
                        dataset.ground_truth,
                        self._motion_model,
                        scenario.associator(),
                        scenario.evaluator,
                    ),
                    flush=True,
                )

    def _dead_reckoning(self, dataset: Dataset[Odometry]) -> float:
        assert dataset.ground_truth is not None
        pose = dataset.initial_pose
        for step in dataset:
            pose = self._motion_model.predict(pose, step.control)
        true = dataset.ground_truth.trajectory[-1]
        return math.hypot(pose.x - true.x, pose.y - true.y)


if __name__ == "__main__":
    Demo().run()
