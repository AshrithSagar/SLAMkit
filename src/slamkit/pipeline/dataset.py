"""
Datasets
========
A SLAM run as data: a sequence of (control, observations) steps, optionally
with ground truth. Simulators produce these; dataset loaders would too.
"""
# src/slamkit/pipeline/dataset.py

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.sensing.observation import Observation

__all__ = [
    "Dataset",
    "GroundTruth",
    "SlamStep",
]


@dataclass(frozen=True, slots=True)
class SlamStep[C: Control]:
    """The robot moved under ``control``, then made ``observations``."""

    control: C
    observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class GroundTruth:
    trajectory: tuple[SE2[WorldFrame, BodyFrame], ...]
    """Every true pose, starting with the initial one (``len(steps) + 1`` poses)."""
    landmarks: Mapping[LandmarkId, Point2[WorldFrame]]


@dataclass(frozen=True, slots=True)
class Dataset[C: Control]:
    initial_pose: SE2[WorldFrame, BodyFrame]
    steps: tuple[SlamStep[C], ...]
    ground_truth: GroundTruth | None = None

    def __post_init__(self) -> None:
        if (
            self.ground_truth is not None
            and len(self.ground_truth.trajectory) != len(self.steps) + 1
        ):
            raise ValueError(
                "ground truth must contain one pose per step plus the initial pose"
            )

    def __iter__(self) -> Iterator[SlamStep[C]]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)
