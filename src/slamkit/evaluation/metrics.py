"""
Metrics
=======
Accuracy of a run against ground truth.
"""
# src/slamkit/evaluation/metrics.py

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.pipeline.dataset import GroundTruth
from slamkit.pipeline.runner import RunResult

__all__ = [
    "EvaluationReport",
    "Evaluator",
]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    position_rmse: float
    """Absolute trajectory error (translation RMSE), in metres."""
    heading_rmse: float
    """Heading RMSE, in radians."""
    final_position_error: float
    landmark_rmse: float
    """RMSE over matched landmarks (``nan`` if none were matched)."""
    landmarks_matched: int
    landmarks_estimated: int
    landmarks_true: int

    def __str__(self) -> str:
        return (
            f"ATE {self.position_rmse:6.3f} m | heading {math.degrees(self.heading_rmse):5.2f}° | "
            f"final {self.final_position_error:6.3f} m | map RMSE {self.landmark_rmse:6.3f} m "
            f"({self.landmarks_matched}/{self.landmarks_true} matched, {self.landmarks_estimated} estimated)"
        )


class Evaluator:
    """Compares a ``RunResult`` with ``GroundTruth``.

    No alignment is applied: every estimator here is anchored at the true
    initial pose, so the gauge is already fixed. Landmarks are matched by id
    when ids are meaningful (known correspondence); otherwise pass
    ``match_radius`` to match each estimate to its nearest unclaimed true
    landmark within that distance.
    """

    def __init__(self, match_radius: float | None = None) -> None:
        self._match_radius = match_radius

    def evaluate(self, result: RunResult, truth: GroundTruth) -> EvaluationReport:
        position_errors, heading_errors = self._trajectory_errors(
            result.trajectory, truth.trajectory
        )
        pairs = self._match_landmarks(result.final.landmarks, truth.landmarks)
        landmark_errors = [estimated.distance_to(true) for estimated, true in pairs]
        return EvaluationReport(
            position_rmse=self._rms(position_errors),
            heading_rmse=self._rms(heading_errors),
            final_position_error=position_errors[-1],
            landmark_rmse=self._rms(landmark_errors),
            landmarks_matched=len(pairs),
            landmarks_estimated=len(result.final.landmarks),
            landmarks_true=len(truth.landmarks),
        )

    @staticmethod
    def _trajectory_errors(
        estimated: Sequence[SE2[WorldFrame, BodyFrame]],
        true: Sequence[SE2[WorldFrame, BodyFrame]],
    ) -> tuple[list[float], list[float]]:
        if len(estimated) != len(true):
            raise ValueError(
                f"trajectory lengths differ: {len(estimated)} vs {len(true)}"
            )
        errors = [e.ominus(t) for e, t in zip(estimated, true, strict=True)]
        positions = [
            math.hypot(e.x - t.x, e.y - t.y)
            for e, t in zip(estimated, true, strict=True)
        ]
        return positions, [abs(float(error[2])) for error in errors]

    def _match_landmarks(
        self,
        estimated: Mapping[LandmarkId, Point2[WorldFrame]],
        true: Mapping[LandmarkId, Point2[WorldFrame]],
    ) -> list[tuple[Point2[WorldFrame], Point2[WorldFrame]]]:
        if self._match_radius is None:
            return [
                (point, true[lid]) for lid, point in estimated.items() if lid in true
            ]
        unclaimed = dict(true)
        pairs: list[tuple[Point2[WorldFrame], Point2[WorldFrame]]] = []
        for point in estimated.values():
            if not unclaimed:
                break
            lid, nearest = min(
                unclaimed.items(), key=lambda item: point.distance_to(item[1])
            )
            if point.distance_to(nearest) <= self._match_radius:
                pairs.append((point, nearest))
                del unclaimed[lid]
        return pairs

    @staticmethod
    def _rms(values: Sequence[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if values else math.nan
