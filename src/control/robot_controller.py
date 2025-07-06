"""
Robot controller implementing the main control loop and coordination between subsystems.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from ..sensors.sensor_manager import SensorManager
from ..vision.vision_system import VisionSystem
from ..hardware.robot_hardware import RobotHardware
from ..planning.path_planner import PathPlanner
from ..motion_control.trajectory_controller import TrajectoryController
from ..utils.state import RobotState, IMUData, OdometryData, VisionData
from ..utils.safety_monitor import SafetyMonitor


@dataclass
class ControlConfig:
    """Configuration for the robot controller."""

    control_frequency: float = 100.0  # Hz
    safety_check_frequency: float = 200.0  # Hz
    state_estimation_frequency: float = 100.0  # Hz
    planning_frequency: float = 10.0  # Hz
    max_linear_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 3.14  # rad/s
    emergency_stop_acceleration: float = -5.0  # m/s^2


class RobotController:
    """Main robot controller coordinating all subsystems."""

    def __init__(
        self,
        hardware: RobotHardware,
        sensor_manager: SensorManager,
        vision_system: VisionSystem,
        path_planner: PathPlanner,
        config: Dict[str, Any],
    ):
        """
        Initialize the robot controller.

        Args:
            hardware: Hardware interface
            sensor_manager: Sensor data manager
            vision_system: Computer vision system
            path_planner: Path planning system
            config: Controller configuration
        """
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.sensor_manager = sensor_manager
        self.vision_system = vision_system
        self.path_planner = path_planner

        # Load configuration
        self.config = ControlConfig(**config)

        # Initialize components
        self.trajectory_controller = TrajectoryController(config.get("trajectory", {}))
        self.safety_monitor = SafetyMonitor(config.get("safety", {}))

        # State variables
        self.current_state = RobotState()
        self.target_state = RobotState()
        self.is_running = False
        self.emergency_stop = False

        # Control loop tasks
        self._control_task: Optional[asyncio.Task] = None
        self._state_estimation_task: Optional[asyncio.Task] = None
        self._planning_task: Optional[asyncio.Task] = None
        self._safety_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the robot controller."""
        self.logger.info("Starting robot controller")

        # Initialize hardware
        await self.hardware.initialize()
        await self.sensor_manager.start()
        await self.vision_system.start()

        self.is_running = True

        # Start control loops
        self._control_task = asyncio.create_task(self._control_loop())
        self._state_estimation_task = asyncio.create_task(self._state_estimation_loop())
        self._planning_task = asyncio.create_task(self._planning_loop())
        self._safety_task = asyncio.create_task(self._safety_loop())

        self.logger.info("Robot controller started successfully")

    async def stop(self) -> None:
        """Stop the robot controller."""
        self.logger.info("Stopping robot controller")

        self.is_running = False

        # Cancel all tasks
        tasks = [
            self._control_task,
            self._state_estimation_task,
            self._planning_task,
            self._safety_task,
        ]

        for task in tasks:
            if task and not task.done():
                task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)

        # Stop subsystems
        await self.vision_system.stop()
        await self.sensor_manager.stop()
        await self.hardware.shutdown()

        self.logger.info("Robot controller stopped")

    async def set_target_pose(self, x: float, y: float, theta: float) -> None:
        """
        Set target pose for the robot.

        Args:
            x: Target x position (m)
            y: Target y position (m)
            theta: Target orientation (rad)
        """
        self.target_state.position = np.array([x, y, 0.0])
        self.target_state.orientation = np.array([0.0, 0.0, theta])
        self.logger.info("New target pose: (%.2f, %.2f, %.2f)", x, y, theta)

    async def emergency_stop_activate(self) -> None:
        """Activate emergency stop."""
        self.emergency_stop = True
        await self.hardware.emergency_stop()
        self.logger.warning("Emergency stop activated")

    async def emergency_stop_release(self) -> None:
        """Release emergency stop."""
        self.emergency_stop = False
        await self.hardware.release_emergency_stop()
        self.logger.info("Emergency stop released")

    async def _control_loop(self) -> None:
        """Main control loop running at high frequency."""
        dt = 1.0 / self.config.control_frequency

        while self.is_running:
            start_time = time.time()

            try:
                if not self.emergency_stop:
                    # Get current trajectory point
                    trajectory_point = self.trajectory_controller.get_current_point(
                        self.current_state, self.target_state
                    )

                    if trajectory_point is not None:
                        # Compute control commands
                        control_cmd = self.trajectory_controller.compute_control(
                            self.current_state, trajectory_point
                        )

                        # Apply safety limits
                        control_cmd = self.safety_monitor.apply_limits(control_cmd)

                        # Send commands to hardware
                        await self.hardware.send_velocity_command(
                            float(control_cmd.linear_velocity[0]),
                            float(control_cmd.angular_velocity[2]),
                        )
                    else:
                        # No trajectory point - send zero velocities
                        await self.hardware.send_velocity_command(0.0, 0.0)
                else:
                    # Emergency stop - send zero velocities
                    await self.hardware.send_velocity_command(0.0, 0.0)

            except Exception as e:
                self.logger.error("Error in control loop: %s", e, exc_info=True)
                await self.emergency_stop_activate()

            # Maintain loop frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            await asyncio.sleep(sleep_time)

    async def _state_estimation_loop(self) -> None:
        """State estimation loop for sensor fusion."""
        dt = 1.0 / self.config.state_estimation_frequency

        while self.is_running:
            start_time = time.time()

            try:
                # Get sensor data
                imu_data = await self.sensor_manager.get_imu_data()
                odometry_data = await self.sensor_manager.get_odometry_data()
                vision_data = await self.vision_system.get_pose_estimate()

                # Perform sensor fusion
                self.current_state = await self._fuse_sensor_data(
                    imu_data, odometry_data, vision_data
                )

            except Exception as e:
                self.logger.error("Error in state estimation: %s", e, exc_info=True)

            # Maintain loop frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            await asyncio.sleep(sleep_time)

    async def _planning_loop(self) -> None:
        """Path planning loop for dynamic replanning."""
        dt = 1.0 / self.config.planning_frequency

        while self.is_running:
            start_time = time.time()

            try:
                # Get obstacle map from vision
                obstacle_map = await self.vision_system.get_obstacle_map()

                # Update path if needed
                if self.path_planner.needs_replanning(
                    self.current_state, self.target_state, obstacle_map
                ):
                    new_path = await self.path_planner.plan_path(
                        self.current_state, self.target_state, obstacle_map
                    )
                    self.trajectory_controller.update_path(new_path)

            except Exception as e:
                self.logger.error("Error in planning loop: %s", e, exc_info=True)

            # Maintain loop frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            await asyncio.sleep(sleep_time)

    async def _safety_loop(self) -> None:
        """Safety monitoring loop."""
        dt = 1.0 / self.config.safety_check_frequency

        while self.is_running:
            start_time = time.time()

            try:
                # Check safety conditions
                safety_status = self.safety_monitor.check_safety(
                    self.current_state, await self.hardware.get_system_status()
                )

                if not safety_status.is_safe:
                    self.logger.warning("Safety violation: %s", safety_status.message)
                    if safety_status.requires_emergency_stop:
                        await self.emergency_stop_activate()

            except Exception as e:
                self.logger.error("Error in safety loop: %s", e, exc_info=True)

            # Maintain loop frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            await asyncio.sleep(sleep_time)

    async def _fuse_sensor_data(
        self,
        imu_data: Optional[IMUData],
        odometry_data: Optional[OdometryData],
        vision_data: Optional[VisionData],
    ) -> RobotState:
        """
        Fuse multiple sensor sources to estimate robot state.

        Args:
            imu_data: IMU sensor data
            odometry_data: Wheel odometry data
            vision_data: Visual odometry/SLAM data

        Returns:
            Fused robot state estimate
        """
        # Simplified sensor fusion - in practice, use Extended Kalman Filter
        # or similar sophisticated fusion algorithm

        state = RobotState()

        # Position from vision if available, otherwise odometry
        if vision_data and vision_data.confidence > 0.7:
            state.position = vision_data.position
        elif odometry_data:
            state.position = odometry_data.position

        # Velocity from odometry
        if odometry_data:
            state.velocity = odometry_data.velocity

        # Orientation from IMU + vision fusion
        if vision_data and vision_data.confidence > 0.5 and imu_data:
            # Weighted average of IMU and vision
            alpha = vision_data.confidence
            state.orientation = alpha * vision_data.orientation + (1 - alpha) * imu_data.orientation
        elif imu_data:
            state.orientation = imu_data.orientation
        elif vision_data:
            state.orientation = vision_data.orientation

        # Angular velocity from IMU
        if imu_data:
            state.angular_velocity = imu_data.angular_velocity

        # Update timestamp
        state.timestamp = time.time()

        return state
