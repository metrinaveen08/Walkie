"""
Robot state representation and related utilities.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RobotState:
    """Represents the complete state of the robot."""

    # Position in world coordinates (x, y, z)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Orientation as quaternion or Euler angles (roll, pitch, yaw)
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Linear velocity (vx, vy, vz)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Angular velocity (wx, wy, wz)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Linear acceleration (ax, ay, az)
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Angular acceleration (alpha_x, alpha_y, alpha_z)
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Timestamp of the state
    timestamp: float = field(default_factory=time.time)

    # Confidence/quality of the state estimate (0.0 to 1.0)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.acceleration = np.asarray(self.acceleration, dtype=np.float64)
        self.angular_acceleration = np.asarray(self.angular_acceleration, dtype=np.float64)

    @property
    def x(self) -> float:
        """X position."""
        return float(self.position[0])

    @property
    def y(self) -> float:
        """Y position."""
        return float(self.position[1])

    @property
    def z(self) -> float:
        """Z position."""
        return float(self.position[2])

    @property
    def yaw(self) -> float:
        """Yaw angle (rotation around z-axis)."""
        return float(self.orientation[2])

    @property
    def speed(self) -> float:
        """Current speed magnitude."""
        return float(np.linalg.norm(self.velocity))

    def distance_to(self, other: "RobotState") -> float:
        """
        Calculate Euclidean distance to another state.

        Args:
            other: Another robot state

        Returns:
            Distance in meters
        """
        return float(np.linalg.norm(self.position - other.position))

    def angular_distance_to(self, other: "RobotState") -> float:
        """
        Calculate angular distance to another state.

        Args:
            other: Another robot state

        Returns:
            Angular distance in radians
        """
        angle_diff = self.yaw - other.yaw
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        return abs(angle_diff)

    def copy(self) -> "RobotState":
        """Create a deep copy of the state."""
        return RobotState(
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            acceleration=self.acceleration.copy(),
            angular_acceleration=self.angular_acceleration.copy(),
            timestamp=self.timestamp,
            confidence=self.confidence,
        )


@dataclass
class ControlCommand:
    """Represents a control command for the robot."""

    # Linear velocity command (m/s)
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Angular velocity command (rad/s)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Timestamp of the command
    timestamp: float = field(default_factory=time.time)

    # Command duration/validity (seconds)
    duration: float = 0.1

    def __post_init__(self) -> None:
        """Ensure arrays are numpy arrays."""
        self.linear_velocity = np.asarray(self.linear_velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)


@dataclass
class TrajectoryPoint:
    """Represents a point on a trajectory."""

    # Desired state at this point
    state: RobotState

    # Time to reach this point from trajectory start
    time: float

    # Feedforward control command
    control: Optional[ControlCommand] = None


@dataclass
class SensorData:
    """Base class for sensor data."""

    # Timestamp when data was acquired
    timestamp: float = field(default_factory=time.time)

    # Data quality/confidence (0.0 to 1.0)
    confidence: float = 1.0


@dataclass
class IMUData(SensorData):
    """IMU sensor data."""

    # Linear acceleration (m/s^2)
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Angular velocity (rad/s)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Orientation (roll, pitch, yaw)
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Magnetic field (Tesla)
    magnetic_field: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class OdometryData(SensorData):
    """Wheel odometry data."""

    # Position estimate
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Velocity estimate
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Orientation estimate
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class VisionData(SensorData):
    """Computer vision data."""

    # Pose estimate from visual odometry/SLAM
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Detected objects
    objects: list = field(default_factory=list)

    # Obstacle map
    obstacle_map: Optional[np.ndarray] = None
