"""
Motion
======
Controls and the motion models they drive.
"""
# src/slamkit/motion/__init__.py

from slamkit.motion.controls import Control, Odometry, VelocityCommand
from slamkit.motion.model import MotionLinearization, MotionModel
from slamkit.motion.odometry import OdometryMotionModel
from slamkit.motion.unicycle import UnicycleMotionModel

__all__ = [
    "Control",
    "MotionLinearization",
    "MotionModel",
    "Odometry",
    "OdometryMotionModel",
    "UnicycleMotionModel",
    "VelocityCommand",
]
