"""
Safety monitoring system for the robot.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np

from .state import RobotState, ControlCommand


@dataclass
class SafetyStatus:
    """Represents the safety status of the robot."""

    is_safe: bool = True
    message: str = ""
    requires_emergency_stop: bool = False
    violations: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.violations is None:
            self.violations = []


@dataclass
class SafetyLimits:
    """Safety limits for robot operation."""

    max_linear_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 3.14  # rad/s
    max_acceleration: float = 5.0  # m/s^2
    max_angular_acceleration: float = 10.0  # rad/s^2

    # Workspace limits (x_min, x_max, y_min, y_max, z_min, z_max)
    workspace_limits: Optional[np.ndarray] = None

    # Minimum distance to obstacles (m)
    min_obstacle_distance: float = 0.3

    # Maximum tilt angles (rad)
    max_roll: float = np.pi / 6  # 30 degrees
    max_pitch: float = np.pi / 6  # 30 degrees

    # Battery voltage limits
    min_battery_voltage: float = 11.0  # V
    critical_battery_voltage: float = 10.5  # V

    # Temperature limits (Celsius)
    max_motor_temperature: float = 80.0
    max_controller_temperature: float = 70.0

    def __post_init__(self) -> None:
        if self.workspace_limits is None:
            # Default workspace: 10m x 10m x 2m
            self.workspace_limits = np.array([-5.0, 5.0, -5.0, 5.0, 0.0, 2.0])


class SafetyMonitor:
    """Monitors robot safety and applies protective measures."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety monitor.

        Args:
            config: Safety configuration parameters
        """
        self.logger = logging.getLogger(__name__)

        # Load safety limits from config
        limits_config = config.get("limits", {})
        self.limits = SafetyLimits(**limits_config)

        # Safety state
        self.last_safety_check = 0.0
        self.consecutive_violations = 0
        self.max_consecutive_violations = config.get("max_consecutive_violations", 5)

        self.logger.info("Safety monitor initialized")

    def check_safety(self, state: RobotState, system_status: Dict[str, Any]) -> SafetyStatus:
        """
        Check if the robot is operating safely.

        Args:
            state: Current robot state
            system_status: Hardware system status

        Returns:
            Safety status
        """
        violations = []
        requires_emergency_stop = False

        # Check velocity limits
        speed = np.linalg.norm(state.velocity)
        if speed > self.limits.max_linear_velocity:
            violations.append(
                f"Linear velocity exceeded: {speed:.2f} > {self.limits.max_linear_velocity}"
            )

        angular_speed = np.linalg.norm(state.angular_velocity)
        if angular_speed > self.limits.max_angular_velocity:
            violations.append(
                f"Angular velocity exceeded: {angular_speed:.2f} > "
                f"{self.limits.max_angular_velocity}"
            )

        # Check workspace limits
        if self.limits.workspace_limits is not None:
            pos = state.position
            limits = self.limits.workspace_limits

            if not (
                limits[0] <= pos[0] <= limits[1]
                and limits[2] <= pos[1] <= limits[3]
                and limits[4] <= pos[2] <= limits[5]
            ):
                violations.append(
                    f"Position outside workspace: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                )
                requires_emergency_stop = True

        # Check orientation limits
        roll, pitch, yaw = state.orientation
        if abs(roll) > self.limits.max_roll:
            violations.append(f"Roll angle exceeded: {roll:.2f} > {self.limits.max_roll}")
            requires_emergency_stop = True

        if abs(pitch) > self.limits.max_pitch:
            violations.append(f"Pitch angle exceeded: {pitch:.2f} > {self.limits.max_pitch}")
            requires_emergency_stop = True

        # Check system status
        battery_voltage = system_status.get("battery_voltage", 12.0)
        if battery_voltage < self.limits.critical_battery_voltage:
            violations.append(f"Critical battery voltage: {battery_voltage:.1f}V")
            requires_emergency_stop = True
        elif battery_voltage < self.limits.min_battery_voltage:
            violations.append(f"Low battery voltage: {battery_voltage:.1f}V")

        # Check temperatures
        motor_temp = system_status.get("motor_temperature", 25.0)
        if motor_temp > self.limits.max_motor_temperature:
            violations.append(f"Motor temperature too high: {motor_temp:.1f}°C")
            requires_emergency_stop = True

        controller_temp = system_status.get("controller_temperature", 25.0)
        if controller_temp > self.limits.max_controller_temperature:
            violations.append(f"Controller temperature too high: {controller_temp:.1f}°C")

        # Check for consecutive violations
        if violations:
            self.consecutive_violations += 1
            if self.consecutive_violations >= self.max_consecutive_violations:
                requires_emergency_stop = True
                violations.append("Too many consecutive safety violations")
        else:
            self.consecutive_violations = 0

        # Create safety status
        is_safe = len(violations) == 0
        message = "; ".join(violations) if violations else "All safety checks passed"

        if violations:
            self.logger.warning("Safety violations detected: %s", message)

        return SafetyStatus(
            is_safe=is_safe,
            message=message,
            requires_emergency_stop=requires_emergency_stop,
            violations=violations,
        )

    def apply_limits(self, command: ControlCommand) -> ControlCommand:
        """
        Apply safety limits to a control command.

        Args:
            command: Original control command

        Returns:
            Limited control command
        """
        limited_command = ControlCommand(
            linear_velocity=command.linear_velocity.copy(),
            angular_velocity=command.angular_velocity.copy(),
            timestamp=command.timestamp,
            duration=command.duration,
        )

        # Limit linear velocity
        linear_speed = np.linalg.norm(limited_command.linear_velocity)
        if linear_speed > self.limits.max_linear_velocity:
            scale_factor = self.limits.max_linear_velocity / linear_speed
            limited_command.linear_velocity *= scale_factor
            self.logger.debug("Limited linear velocity by factor %.3f", scale_factor)

        # Limit angular velocity
        angular_speed = np.linalg.norm(limited_command.angular_velocity)
        if angular_speed > self.limits.max_angular_velocity:
            scale_factor = self.limits.max_angular_velocity / angular_speed
            limited_command.angular_velocity *= scale_factor
            self.logger.debug("Limited angular velocity by factor %.3f", scale_factor)

        return limited_command

    def is_collision_imminent(self, state: RobotState, obstacle_map: np.ndarray) -> bool:
        """
        Check if collision is imminent based on current trajectory.

        Args:
            state: Current robot state
            obstacle_map: Occupancy grid or obstacle map

        Returns:
            True if collision is imminent
        """
        # Simplified collision check - project current velocity forward
        # In practice, this would use more sophisticated prediction

        if obstacle_map is None:
            return False

        # Project position forward by velocity * time_horizon
        time_horizon = 1.0  # seconds
        future_position = state.position + state.velocity * time_horizon

        # Check if future position would be in collision
        # This is a simplified implementation - actual implementation would
        # use proper collision detection algorithms

        return False  # Placeholder

    def get_safe_velocity_limits(
        self, state: RobotState, obstacle_map: np.ndarray
    ) -> Dict[str, float]:
        """
        Get dynamic velocity limits based on current situation.

        Args:
            state: Current robot state
            obstacle_map: Current obstacle map

        Returns:
            Dictionary with safe velocity limits
        """
        # Start with base limits
        limits = {
            "max_linear_velocity": self.limits.max_linear_velocity,
            "max_angular_velocity": self.limits.max_angular_velocity,
        }

        # Reduce limits if near obstacles
        if obstacle_map is not None:
            # Simplified distance-based velocity limiting
            # In practice, this would use more sophisticated algorithms
            min_distance = self._get_min_obstacle_distance(state.position, obstacle_map)

            if min_distance < self.limits.min_obstacle_distance * 2:
                # Reduce velocity as we get closer to obstacles
                scale_factor = min_distance / (self.limits.min_obstacle_distance * 2)
                scale_factor = max(0.1, scale_factor)  # Don't go below 10% of max

                limits["max_linear_velocity"] *= scale_factor
                limits["max_angular_velocity"] *= scale_factor

        return limits

    def _get_min_obstacle_distance(self, position: np.ndarray, obstacle_map: np.ndarray) -> float:
        """
        Get minimum distance to obstacles from current position.

        Args:
            position: Current position
            obstacle_map: Obstacle map

        Returns:
            Minimum distance to obstacles
        """
        # Placeholder implementation
        # In practice, this would properly query the obstacle map
        return 1.0  # meters
