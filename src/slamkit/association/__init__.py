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
    Unassociated,
)
from slamkit.association.known import KnownCorrespondence
from slamkit.association.nearest_neighbor import NearestNeighbor

__all__ = [
    "Association",
    "AssociationContext",
    "DataAssociator",
    "KnownCorrespondence",
    "Matched",
    "NearestNeighbor",
    "NewLandmark",
    "Unassociated",
]
