"""
Joint compatibility
===================
JCBB: joint compatibility branch and bound (Neira & Tardós, 2001).
"""
# src/slamkit/association/joint_compatibility.py

import math
from collections.abc import Sequence
from dataclasses import dataclass

from slamkit.association.base import (
    Association,
    AssociationContext,
    DataAssociator,
    Matched,
    NewLandmark,
    Pairing,
    Unassociated,
)
from slamkit.core.ids import LandmarkId, LandmarkIdAllocator
from slamkit.probability.chi_square import ChiSquare
from slamkit.sensing.observation import Observation

__all__ = [
    "JointCompatibility",
]

type Hypothesis = tuple[LandmarkId | None, ...]
"""One landmark (or ``None``) per observation considered so far."""


@dataclass(frozen=True, slots=True, order=True)
class _Candidate:
    distance: float
    landmark_id: LandmarkId


@dataclass(frozen=True, slots=True)
class _Score:
    """Hypotheses are ranked by pairings first, then by joint distance."""

    pairings: int
    distance: float

    def beats(self, other: _Score) -> bool:
        return (self.pairings, -self.distance) > (other.pairings, -other.distance)


class _BranchAndBound:
    """One search over the interpretation tree of a batch of observations."""

    def __init__(
        self,
        observations: Sequence[Observation],
        candidates: Sequence[Sequence[_Candidate]],
        context: AssociationContext,
        confidence: float,
    ) -> None:
        self._observations = observations
        self._candidates = candidates
        self._context = context
        self._confidence = confidence
        self._best: Hypothesis = (None,) * len(observations)
        self._best_score = _Score(0, 0.0)

    def run(self) -> Hypothesis:
        self._descend(())
        return self._best

    def _descend(self, hypothesis: Hypothesis) -> None:
        index = len(hypothesis)
        if index == len(self._observations):
            self._consider(hypothesis)
            return
        used = {lid for lid in hypothesis if lid is not None}
        for candidate in self._candidates[index]:
            if candidate.landmark_id in used:
                continue
            extended = (*hypothesis, candidate.landmark_id)
            if self._joint_distance(extended) is not None:
                self._descend(extended)
        # Leaving this observation unpaired is only worth exploring while the
        # remaining observations could still reach the best pairing count.
        remaining = len(self._observations) - index - 1
        if self._pairings(hypothesis) + remaining >= self._best_score.pairings:
            self._descend((*hypothesis, None))

    def _consider(self, hypothesis: Hypothesis) -> None:
        distance = self._joint_distance(hypothesis)
        score = _Score(
            self._pairings(hypothesis), 0.0 if distance is None else distance
        )
        if score.beats(self._best_score):
            self._best, self._best_score = hypothesis, score

    def _joint_distance(self, hypothesis: Hypothesis) -> float | None:
        """Joint squared Mahalanobis distance if the hypothesis is jointly
        compatible at the configured confidence, else ``None``."""
        pairings = [
            Pairing(self._observations[i], lid)
            for i, lid in enumerate(hypothesis)
            if lid is not None
        ]
        if not pairings:
            return None
        innovation = self._context.joint_innovation(pairings)
        distance = innovation.mahalanobis_squared()
        threshold = ChiSquare.quantile(self._confidence, innovation.dim)
        return distance if distance <= threshold else None

    @staticmethod
    def _pairings(hypothesis: Hypothesis) -> int:
        return sum(lid is not None for lid in hypothesis)


class JointCompatibility(DataAssociator):
    """Associates a whole batch at once, keeping the largest set of pairings that
    is *jointly* consistent with the estimator's uncertainty.

    Individually-gated nearest neighbour treats each observation in isolation,
    so when the pose is uncertain a shifted scan can match every landmark to
    its neighbour. Joint compatibility tests all pairings together, through
    their correlated innovations, and so rejects such shifted interpretations.
    Search is exponential in the worst case but heavily pruned in practice.

    Unpaired observations whose nearest landmark is beyond the
    ``new_landmark_confidence`` χ² gate start new landmarks; the rest are
    discarded as ambiguous. Observation tags are ignored.
    """

    def __init__(
        self,
        confidence: float = 0.99,
        new_landmark_confidence: float = 0.9999,
        allocator: LandmarkIdAllocator | None = None,
    ) -> None:
        if not 0.0 < confidence <= new_landmark_confidence < 1.0:
            raise ValueError("require 0 < confidence <= new_landmark_confidence < 1")
        self._confidence = confidence
        self._new_landmark_confidence = new_landmark_confidence
        self._allocator = allocator if allocator is not None else LandmarkIdAllocator()

    def associate(
        self, observations: Sequence[Observation], context: AssociationContext
    ) -> list[Association]:
        candidates: list[list[_Candidate]] = []
        nearest: list[float] = []
        for observation in observations:
            individual: list[_Candidate] = []
            closest = math.inf
            for landmark_id in context.landmark_ids():
                innovation = context.innovation(observation, landmark_id)
                distance = innovation.mahalanobis_squared()
                closest = min(closest, distance)
                if distance <= ChiSquare.quantile(self._confidence, innovation.dim):
                    individual.append(_Candidate(distance, landmark_id))
            candidates.append(sorted(individual))
            nearest.append(closest)

        hypothesis = _BranchAndBound(
            observations, candidates, context, self._confidence
        ).run()

        associations: list[Association] = []
        for observation, paired, closest in zip(
            observations, hypothesis, nearest, strict=True
        ):
            if paired is not None:
                associations.append(Matched(paired))
            elif closest > self._new_landmark_threshold(observation):
                associations.append(NewLandmark(self._allocator.allocate()))
            else:
                associations.append(
                    Unassociated(f"jointly incompatible (nearest d²={closest:.2f})")
                )
        return associations

    def _new_landmark_threshold(self, observation: Observation) -> float:
        return ChiSquare.quantile(
            self._new_landmark_confidence, observation.noise_covariance.shape[0]
        )
