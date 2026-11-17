"""
Keys and values
===============
Variables of a SLAM factor graph, and an assignment of estimates to them.

Keys are typed by what they name, and ``Values`` is typed accordingly:
``values[PoseKey(3)]`` is statically an ``SE2[WorldFrame, BodyFrame]`` and
``values[LandmarkKey(7)]`` a ``Point2[WorldFrame]``.
"""
# src/slamkit/estimation/graph/values.py

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import overload

from slamkit.core.arrays import Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2

__all__ = [
    "Key",
    "LandmarkKey",
    "PoseKey",
    "Values",
]


@dataclass(frozen=True, slots=True, order=True)
class PoseKey:
    """The robot pose at timestep ``index``."""

    index: int

    DOF = SE2.DOF

    def __str__(self) -> str:
        return f"x{self.index}"


@dataclass(frozen=True, slots=True, order=True)
class LandmarkKey:
    """The position of landmark ``landmark_id``."""

    landmark_id: LandmarkId

    DOF = Point2.DOF

    def __str__(self) -> str:
        return f"l{self.landmark_id}"


type Key = PoseKey | LandmarkKey


class Values:
    """A typed assignment ``key → manifold value``."""

    def __init__(self) -> None:
        self._poses: dict[PoseKey, SE2[WorldFrame, BodyFrame]] = {}
        self._landmarks: dict[LandmarkKey, Point2[WorldFrame]] = {}

    @overload
    def __getitem__(self, key: PoseKey) -> SE2[WorldFrame, BodyFrame]: ...
    @overload
    def __getitem__(self, key: LandmarkKey) -> Point2[WorldFrame]: ...
    def __getitem__(self, key: Key) -> SE2[WorldFrame, BodyFrame] | Point2[WorldFrame]:
        match key:
            case PoseKey():
                return self._poses[key]
            case LandmarkKey():
                return self._landmarks[key]

    def set_pose(self, key: PoseKey, pose: SE2[WorldFrame, BodyFrame]) -> None:
        self._poses[key] = pose

    def set_landmark(self, key: LandmarkKey, landmark: Point2[WorldFrame]) -> None:
        self._landmarks[key] = landmark

    def __contains__(self, key: Key) -> bool:
        return key in self._poses or key in self._landmarks

    def __iter__(self) -> Iterator[Key]:
        yield from sorted(self._poses)
        yield from sorted(self._landmarks)

    def __len__(self) -> int:
        return len(self._poses) + len(self._landmarks)

    @property
    def poses(self) -> Mapping[PoseKey, SE2[WorldFrame, BodyFrame]]:
        return self._poses

    @property
    def landmarks(self) -> Mapping[LandmarkKey, Point2[WorldFrame]]:
        return self._landmarks

    def retract(self, deltas: Mapping[Key, Vector]) -> Values:
        """A new ``Values`` with ``value ⊕ deltas[key]`` for every key in ``deltas``."""
        result = self.copy()
        for key, delta in deltas.items():
            match key:
                case PoseKey():
                    result.set_pose(key, self._poses[key].oplus(delta))
                case LandmarkKey():
                    result.set_landmark(key, self._landmarks[key].oplus(delta))
        return result

    def copy(self) -> Values:
        result = Values()
        result._poses = dict(self._poses)
        result._landmarks = dict(self._landmarks)
        return result
