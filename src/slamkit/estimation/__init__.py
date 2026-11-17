"""
Estimation
==========
SLAM backends: different belief representations behind one interface.
"""
# src/slamkit/estimation/__init__.py

from slamkit.estimation.ekf import EkfSlam, EkfSlamState
from slamkit.estimation.estimator import (
    MapHypothesis,
    SlamEstimate,
    SlamEstimator,
    SmoothingSlamEstimator,
)
from slamkit.estimation.graph import GraphSlam
from slamkit.estimation.particle import FastSlam

__all__ = [
    "EkfSlam",
    "EkfSlamState",
    "FastSlam",
    "GraphSlam",
    "MapHypothesis",
    "SlamEstimate",
    "SlamEstimator",
    "SmoothingSlamEstimator",
]
