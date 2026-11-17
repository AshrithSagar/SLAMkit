"""
Simulator
=========
Generates datasets with ground truth from a world, a motion model and sensors.
"""
# src/slamkit/simulation/simulator.py

from collections.abc import Sequence

import numpy as np

from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.motion.model import MotionModel
from slamkit.pipeline.dataset import Dataset, GroundTruth, SlamStep
from slamkit.sensing.observation import Observation
from slamkit.sensing.sensor import Sensor
from slamkit.simulation.world import LandmarkWorld

__all__ = [
    "Simulator",
]


class Simulator[C: Control]:
    """Moves a true robot by *sampling* the motion model (so the estimator's model
    is exactly right in expectation), and records each sensor's noisy readings.

    With ``tag_observations`` the true landmark ids are attached, enabling
    known-correspondence association; without it the estimator must associate.
    """

    def __init__(
        self,
        world: LandmarkWorld,
        motion_model: MotionModel[C],
        sensors: Sequence[Sensor],
        rng: np.random.Generator,
        tag_observations: bool = True,
        initial_pose: SE2[WorldFrame, BodyFrame] | None = None,
    ) -> None:
        if not sensors:
            raise ValueError("at least one sensor is required")
        self._world = world
        self._motion_model = motion_model
        self._sensors = tuple(sensors)
        self._rng = rng
        self._tag_observations = tag_observations
        self._initial_pose = (
            initial_pose
            if initial_pose is not None
            else SE2[WorldFrame, BodyFrame].identity()
        )

    def run(self, controls: Sequence[C]) -> Dataset[C]:
        pose = self._initial_pose
        trajectory = [pose]
        steps: list[SlamStep[C]] = []
        for control in controls:
            pose = self._motion_model.sample(pose, control, self._rng)
            trajectory.append(pose)
            steps.append(SlamStep(control, self._observe(pose)))
        truth = GroundTruth(tuple(trajectory), self._world.landmarks)
        return Dataset(self._initial_pose, tuple(steps), truth)

    def _observe(self, pose: SE2[WorldFrame, BodyFrame]) -> tuple[Observation, ...]:
        observations: list[Observation] = []
        for sensor in self._sensors:
            for landmark_id, landmark in self._world:
                tag = landmark_id if self._tag_observations else None
                observation = sensor.observe(pose, landmark, tag, self._rng)
                if observation is not None:
                    observations.append(observation)
        order = self._rng.permutation(
            len(observations)
        )  # sensors don't report in id order
        return tuple(observations[int(i)] for i in order)
