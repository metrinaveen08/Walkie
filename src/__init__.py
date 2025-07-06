"""
Agile Dynamic Robot Control System

A comprehensive robotics framework for agile dynamic robot control with:
- Advanced motion control and trajectory planning
- Multi-sensor fusion and state estimation
- Computer vision and perception
- Real-time safety monitoring
- Hardware abstraction layer
"""

__version__ = "0.1.0"
__author__ = "Robot Developer"
__email__ = "developer@example.com"

from .utils.state import RobotState, ControlCommand, TrajectoryPoint
from .utils.logger import RobotLogger, setup_logger
from .control.robot_controller import RobotController
from .hardware.robot_hardware import RobotHardware
from .sensors.sensor_manager import SensorManager
from .vision.vision_system import VisionSystem
from .planning.path_planner import PathPlanner
from .motion_control.trajectory_controller import TrajectoryController

__all__ = [
    "RobotState",
    "ControlCommand",
    "TrajectoryPoint",
    "RobotLogger",
    "setup_logger",
    "RobotController",
    "RobotHardware",
    "SensorManager",
    "VisionSystem",
    "PathPlanner",
    "TrajectoryController",
]
