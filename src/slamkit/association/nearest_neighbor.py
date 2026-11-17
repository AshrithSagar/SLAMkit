"""
Nearest-neighbour association
=============================
"""
# src/slamkit/association/nearest_neighbor.py

import math
from collections.abc import Sequence
from dataclasses import dataclass

from slamkit.association.base import (
    Association,
    AssociationContext,
    DataAssociator,
    Matched,
    NewLandmark,
    Unassociated,
)
from slamkit.core.ids import LandmarkId, LandmarkIdAllocator
from slamkit.sensing.observation import Observation

__all__ = [
    "NearestNeighbor",
]


@dataclass(frozen=True, slots=True, order=True)
class _Candidate:
    distance: float
    observation_index: int
    landmark_id: LandmarkId


class NearestNeighbor(DataAssociator):
    """Greedy, mutually-exclusive Mahalanobis nearest-neighbour association.

    Thresholds are squared Mahalanobis distances (χ² values for the
    measurement's dimension; defaults suit 2-D measurements):

    * pairs with ``d² ≤ gate`` are candidates, matched greedily from the closest,
      each landmark used at most once per batch;
    * an unmatched observation whose nearest landmark has ``d² > new_landmark``
      starts a new landmark; one in between is too ambiguous and is dropped.

    Observation tags are ignored. New ids come from ``allocator``; share one
    allocator wherever ids must stay unique.
    """

    def __init__(
        self,
        gate: float = 9.21,
        new_landmark: float = 18.42,
        allocator: LandmarkIdAllocator | None = None,
    ) -> None:
        if not 0.0 < gate <= new_landmark:
            raise ValueError("require 0 < gate <= new_landmark")
        self._gate: float = gate
        self._new_landmark: float = new_landmark
        self._allocator: LandmarkIdAllocator = (
            allocator if allocator is not None else LandmarkIdAllocator()
        )

    def associate(
        self, observations: Sequence[Observation], context: AssociationContext
    ) -> list[Association]:
        nearest = [math.inf] * len(observations)
        candidates: list[_Candidate] = []
        for index, observation in enumerate(observations):
            for landmark_id in context.landmark_ids():
                distance = context.innovation(
                    observation, landmark_id
                ).mahalanobis_squared()
                nearest[index] = min(nearest[index], distance)
                if distance <= self._gate:
                    candidates.append(_Candidate(distance, index, landmark_id))

        assigned: dict[int, LandmarkId] = {}
        used: set[LandmarkId] = set()
        for candidate in sorted(candidates):
            if (
                candidate.observation_index not in assigned
                and candidate.landmark_id not in used
            ):
                assigned[candidate.observation_index] = candidate.landmark_id
                used.add(candidate.landmark_id)

        associations: list[Association] = []
        for index in range(len(observations)):
            if index in assigned:
                associations.append(Matched(assigned[index]))
            elif nearest[index] > self._new_landmark:
                associations.append(NewLandmark(self._allocator.allocate()))
            else:
                associations.append(
                    Unassociated(f"ambiguous (nearest d²={nearest[index]:.2f})")
                )
        return associations
