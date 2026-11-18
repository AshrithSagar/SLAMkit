"""
Association
===========
Solving the correspondence problem, independently of the estimator.
"""
# src/slamkit/association/__init__.py

from slamkit.association.base import (
    Association,
    AssociationContext,
    DataAssociator,
    Matched,
    NewLandmark,
    Pairing,
    Unassociated,
)
from slamkit.association.joint_compatibility import JointCompatibility
from slamkit.association.known import KnownCorrespondence
from slamkit.association.nearest_neighbor import NearestNeighbor

__all__ = [
    "Association",
    "AssociationContext",
    "DataAssociator",
    "JointCompatibility",
    "KnownCorrespondence",
    "Matched",
    "NearestNeighbor",
    "NewLandmark",
    "Pairing",
    "Unassociated",
]
