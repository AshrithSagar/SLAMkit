"""
Gaussian SLAM
=============
EKF- and UKF-SLAM: one joint Gaussian, two ways of linearising.
"""
# src/slamkit/estimation/gaussian/__init__.py

from slamkit.estimation.gaussian.gaussian_slam import EkfSlam, GaussianSlam, UkfSlam
from slamkit.estimation.gaussian.linearizer import (
    AnalyticLinearizer,
    Linearizer,
    UnscentedLinearizer,
)
from slamkit.estimation.gaussian.state import GaussianSlamState

__all__ = [
    "AnalyticLinearizer",
    "EkfSlam",
    "GaussianSlam",
    "GaussianSlamState",
    "Linearizer",
    "UkfSlam",
    "UnscentedLinearizer",
]
