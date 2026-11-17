"""
Pipeline
========
Datasets and the runner that feeds them to estimators.
"""
# src/slamkit/pipeline/__init__.py

from slamkit.pipeline.dataset import Dataset, GroundTruth, SlamStep
from slamkit.pipeline.runner import RunResult, SlamRunner

__all__ = [
    "Dataset",
    "GroundTruth",
    "RunResult",
    "SlamRunner",
    "SlamStep",
]
