"""
Pose proposals
==============
The one thing that separates FastSLAM 1.0 from FastSLAM 2.0.

Each particle's new pose is drawn from a *proposal* distribution and then
weighted to correct for the proposal not being the true posterior:

* ``MotionProposal``      (FastSLAM 1.0): sample from the motion model alone,
  then weight by how well the observations fit. Simple, but when the sensor
  is much more accurate than odometry nearly all samples land where the
  observations say the robot is not, and the particle set collapses.
* ``ObservationProposal`` (FastSLAM 2.0): first fold the observations of
  already-mapped landmarks into the motion prior (a small per-particle EKF on
  the pose), then sample. Samples land where the observations agree, so far
  fewer particles are needed.
"""
# src/slamkit/estimation/particle/proposal.py

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from slamkit.association.base import DataAssociator, Matched, NewLandmark
from slamkit.estimation.particle.particle import Particle
from slamkit.motion.controls import Control
from slamkit.motion.model import MotionModel
from slamkit.sensing.observation import Observation

__all__ = [
    "MotionProposal",
    "ObservationProposal",
    "PoseProposal",
]


class PoseProposal(ABC):
    """How a particle's pose is predicted, then conditioned and weighted."""

    @abstractmethod
    def predict[C: Control](
        self,
        particle: Particle,
        motion_model: MotionModel[C],
        control: C,
        rng: np.random.Generator,
    ) -> None: ...

    @abstractmethod
    def update(
        self,
        particle: Particle,
        observations: Sequence[Observation],
        associator: DataAssociator,
        rng: np.random.Generator,
    ) -> None:
        """Leave the particle with a known pose, an updated map and its weight."""


class MotionProposal(PoseProposal):
    """FastSLAM 1.0: sample from the motion model, weight by the observations."""

    def predict[C: Control](
        self,
        particle: Particle,
        motion_model: MotionModel[C],
        control: C,
        rng: np.random.Generator,
    ) -> None:
        particle.move_to(motion_model.sample(particle.pose, control, rng))

    def update(
        self,
        particle: Particle,
        observations: Sequence[Observation],
        associator: DataAssociator,
        rng: np.random.Generator,
    ) -> None:
        particle.assimilate(observations, associator)


class ObservationProposal(PoseProposal):
    """FastSLAM 2.0: sample from the motion prior *conditioned on* the
    observations of mapped landmarks."""

    def predict[C: Control](
        self,
        particle: Particle,
        motion_model: MotionModel[C],
        control: C,
        rng: np.random.Generator,
    ) -> None:
        particle.predict_pose(motion_model.linearize(particle.pose, control))

    def update(
        self,
        particle: Particle,
        observations: Sequence[Observation],
        associator: DataAssociator,
        rng: np.random.Generator,
    ) -> None:
        # Associate against the pose *prior*: its uncertainty widens the gates.
        associations = associator.associate(observations, particle)
        for observation, association in zip(observations, associations, strict=True):
            if isinstance(association, Matched):
                particle.condition_pose(observation, association.landmark_id)
        particle.sample_pose(rng)
        for observation, association in zip(observations, associations, strict=True):
            match association:
                case Matched(landmark_id):
                    particle.correct_landmark(observation, landmark_id)
                case NewLandmark(landmark_id):
                    particle.introduce(observation, landmark_id)
                case _:
                    pass
