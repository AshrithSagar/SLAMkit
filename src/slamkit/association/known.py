"""
Known correspondence
====================
"""
# src/slamkit/association/known.py

from collections.abc import Sequence

from slamkit.association.base import (
    Association,
    AssociationContext,
    DataAssociator,
    Matched,
    NewLandmark,
    Unassociated,
)
from slamkit.core.ids import LandmarkId
from slamkit.sensing.observation import Observation

__all__ = [
    "KnownCorrespondence",
]


class KnownCorrespondence(DataAssociator):
    """Trusts each observation's ``tag`` (fiducial markers, simulation ground truth).

    Untagged observations are discarded. A tag seen for the first time becomes a
    ``NewLandmark``; later observations of it in the same batch are ``Matched``,
    since estimators apply associations in order.
    """

    def associate(
        self, observations: Sequence[Observation], context: AssociationContext
    ) -> list[Association]:
        known: set[LandmarkId] = set(context.landmark_ids())
        associations: list[Association] = []
        for observation in observations:
            tag = observation.tag
            if tag is None:
                associations.append(Unassociated("observation carries no tag"))
            elif tag in known:
                associations.append(Matched(tag))
            else:
                known.add(tag)
                associations.append(NewLandmark(tag))
        return associations
