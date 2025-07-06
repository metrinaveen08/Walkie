"""
Trajectory controller for smooth motion execution.
"""

from typing import Dict, Any, Optional, List
import time

import numpy as np

from ..utils.state import RobotState, ControlCommand, TrajectoryPoint
from ..utils.logger import RobotLogger


class TrajectoryController:
    """Controller for following planned trajectories."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trajectory controller.

        Args:
            config: Controller configuration
        """
        self.logger = RobotLogger(__name__)
        self.config = config

        # Control parameters
        self.kp_linear = config.get("kp_linear", 2.0)
        self.ki_linear = config.get("ki_linear", 0.1)
        self.kd_linear = config.get("kd_linear", 0.5)

        self.kp_angular = config.get("kp_angular", 3.0)
        self.ki_angular = config.get("ki_angular", 0.1)
        self.kd_angular = config.get("kd_angular", 0.3)

        # Trajectory state
        self.current_trajectory: List[TrajectoryPoint] = []
        self.trajectory_start_time = 0.0
        self.current_waypoint_index = 0

        # PID controller state
        self.linear_error_integral = 0.0
        self.angular_error_integral = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0
        self.last_control_time = 0.0

    def update_path(self, trajectory: List[TrajectoryPoint]) -> None:
        """
        Update the trajectory to follow.

        Args:
            trajectory: New trajectory to follow
        """
        self.current_trajectory = trajectory
        self.trajectory_start_time = time.time()
        self.current_waypoint_index = 0

        # Reset PID state
        self.linear_error_integral = 0.0
        self.angular_error_integral = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0

        self.logger.info(f"Updated trajectory with {len(trajectory)} points")

    def get_current_point(
        self, current_state: RobotState, target_state: RobotState
    ) -> Optional[TrajectoryPoint]:
        """
        Get current trajectory point based on time.

        Args:
            current_state: Current robot state
            target_state: Target state (used if no trajectory)

        Returns:
            Current trajectory point or None
        """
        if not self.current_trajectory:
            # No trajectory - create point towards target
            return self._create_target_point(current_state, target_state)

        current_time = time.time() - self.trajectory_start_time

        # Find appropriate trajectory point
        for i, point in enumerate(self.current_trajectory):
            if point.time >= current_time:
                self.current_waypoint_index = i
                return point

        # Past end of trajectory - return last point
        self.current_waypoint_index = len(self.current_trajectory) - 1
        return self.current_trajectory[-1]

    def compute_control(
        self, current_state: RobotState, target_point: TrajectoryPoint
    ) -> ControlCommand:
        """
        Compute control command to follow trajectory.

        Args:
            current_state: Current robot state
            target_point: Target trajectory point

        Returns:
            Control command
        """
        current_time = time.time()
        dt = current_time - self.last_control_time if self.last_control_time > 0 else 0.01
        self.last_control_time = current_time

        # Position errors
        position_error = target_point.state.position - current_state.position
        linear_error = float(np.linalg.norm(position_error[:2]))  # Distance error

        # Orientation error
        target_yaw = target_point.state.yaw
        current_yaw = current_state.yaw
        angular_error = self._normalize_angle(target_yaw - current_yaw)

        # PID control for linear velocity
        self.linear_error_integral += linear_error * dt
        linear_error_derivative = (linear_error - self.last_linear_error) / dt

        linear_velocity = (
            self.kp_linear * linear_error
            + self.ki_linear * self.linear_error_integral
            + self.kd_linear * linear_error_derivative
        )

        # PID control for angular velocity
        self.angular_error_integral += angular_error * dt
        angular_error_derivative = (angular_error - self.last_angular_error) / dt

        angular_velocity = (
            self.kp_angular * angular_error
            + self.ki_angular * self.angular_error_integral
            + self.kd_angular * angular_error_derivative
        )

        # Add feedforward from trajectory
        if target_point.control:
            linear_velocity += float(np.linalg.norm(target_point.control.linear_velocity[:2]))
            angular_velocity += float(target_point.control.angular_velocity[2])

        # Apply velocity limits
        max_linear = self.config.get("max_linear_velocity", 2.0)
        max_angular = self.config.get("max_angular_velocity", 3.14)

        linear_velocity = np.clip(linear_velocity, -max_linear, max_linear)
        angular_velocity = np.clip(angular_velocity, -max_angular, max_angular)

        # Store errors for next iteration
        self.last_linear_error = linear_error
        self.last_angular_error = angular_error

        # Create control command
        control_cmd = ControlCommand(
            linear_velocity=np.array([linear_velocity, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, angular_velocity]),
            timestamp=current_time,
        )

        self.logger.debug(f"Control: linear={linear_velocity:.3f}, angular={angular_velocity:.3f}")

        return control_cmd

    def _create_target_point(
        self, current_state: RobotState, target_state: RobotState
    ) -> TrajectoryPoint:
        """
        Create trajectory point towards target when no trajectory exists.

        Args:
            current_state: Current robot state
            target_state: Target robot state

        Returns:
            Trajectory point towards target
        """
        # Create intermediate point towards target
        direction = target_state.position - current_state.position
        distance = np.linalg.norm(direction[:2])

        if distance < 0.1:  # Close to target
            return TrajectoryPoint(state=target_state, time=0.0)

        # Create point 1 meter towards target
        max_step = min(1.0, distance)
        unit_direction = direction / np.linalg.norm(direction)

        intermediate_state = RobotState()
        intermediate_state.position = current_state.position + unit_direction * max_step
        intermediate_state.orientation = np.array(
            [0.0, 0.0, np.arctan2(direction[1], direction[0])]
        )

        return TrajectoryPoint(state=intermediate_state, time=0.0)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def is_trajectory_complete(self, current_state: RobotState, tolerance: float = 0.1) -> bool:
        """
        Check if trajectory is complete.

        Args:
            current_state: Current robot state
            tolerance: Position tolerance in meters

        Returns:
            True if trajectory is complete
        """
        if not self.current_trajectory:
            return True

        # Check if we're at the final point
        final_point = self.current_trajectory[-1]
        distance_to_goal = float(
            np.linalg.norm(final_point.state.position[:2] - current_state.position[:2])
        )

        return distance_to_goal < tolerance

    def get_trajectory_progress(self) -> float:
        """
        Get trajectory completion progress.

        Returns:
            Progress as fraction from 0.0 to 1.0
        """
        if not self.current_trajectory:
            return 1.0

        total_points = len(self.current_trajectory)
        return min(1.0, self.current_waypoint_index / max(1, total_points - 1))

    def get_remaining_distance(self, current_state: RobotState) -> float:
        """
        Get remaining distance in trajectory.

        Args:
            current_state: Current robot state

        Returns:
            Remaining distance in meters
        """
        if not self.current_trajectory:
            return 0.0

        total_distance = 0.0
        current_pos = current_state.position[:2]

        # Distance to next waypoint
        if self.current_waypoint_index < len(self.current_trajectory):
            next_point = self.current_trajectory[self.current_waypoint_index]
            total_distance += float(np.linalg.norm(next_point.state.position[:2] - current_pos))

        # Distance between remaining waypoints
        for i in range(self.current_waypoint_index, len(self.current_trajectory) - 1):
            p1 = self.current_trajectory[i].state.position[:2]
            p2 = self.current_trajectory[i + 1].state.position[:2]
            total_distance += float(np.linalg.norm(p2 - p1))

        return total_distance

    def emergency_stop(self) -> ControlCommand:
        """
        Generate emergency stop command.

        Returns:
            Zero velocity control command
        """
        return ControlCommand(
            linear_velocity=np.zeros(3), angular_velocity=np.zeros(3), timestamp=time.time()
        )
