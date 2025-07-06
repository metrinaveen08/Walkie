"""
Sensor management system for data acquisition and fusion.
"""

import asyncio
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import time

import numpy as np

from ..utils.state import SensorData, IMUData, OdometryData
from ..utils.logger import RobotLogger


class SensorInterface(ABC):
    """Abstract base class for sensor interfaces."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sensor interface.

        Args:
            config: Sensor configuration
        """
        self.config = config
        self.logger = RobotLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_active = False
        self.last_data_time = 0.0

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the sensor."""
        pass

    @abstractmethod
    async def read_data(self) -> SensorData:
        """Read data from the sensor."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the sensor."""
        pass

    @property
    def is_healthy(self) -> bool:
        """Check if sensor is healthy."""
        return self.is_active and (time.time() - self.last_data_time) < 1.0


class IMUSensor(SensorInterface):
    """IMU sensor interface."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device_path = config.get("device_path", "/dev/imu0")
        self.sample_rate = config.get("sample_rate", 100)  # Hz
        self.calibration_data = config.get("calibration", {})

    async def initialize(self) -> None:
        """Initialize IMU sensor."""
        self.logger.info(f"Initializing IMU sensor on {self.device_path}")
        # In a real implementation, this would initialize the actual IMU device
        await asyncio.sleep(0.1)  # Simulate initialization delay
        self.is_active = True
        self.logger.info("IMU sensor initialized successfully")

    async def read_data(self) -> IMUData:
        """Read IMU data."""
        if not self.is_active:
            raise RuntimeError("IMU sensor not initialized")

        # Simulate IMU data - in real implementation, read from actual sensor
        current_time = time.time()

        # Generate realistic IMU data with some noise
        linear_acceleration = np.array(
            [
                0.0 + np.random.normal(0, 0.1),  # x
                0.0 + np.random.normal(0, 0.1),  # y
                9.81 + np.random.normal(0, 0.1),  # z (gravity)
            ]
        )

        angular_velocity = np.array(
            [
                np.random.normal(0, 0.01),  # roll rate
                np.random.normal(0, 0.01),  # pitch rate
                np.random.normal(0, 0.01),  # yaw rate
            ]
        )

        orientation = np.array(
            [
                np.random.normal(0, 0.05),  # roll
                np.random.normal(0, 0.05),  # pitch
                np.random.normal(0, 0.1),  # yaw
            ]
        )

        magnetic_field = np.array(
            [
                22.0 + np.random.normal(0, 1.0),  # x
                5.0 + np.random.normal(0, 1.0),  # y
                -45.0 + np.random.normal(0, 1.0),  # z
            ]
        )

        self.last_data_time = current_time

        return IMUData(
            timestamp=current_time,
            linear_acceleration=linear_acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation,
            magnetic_field=magnetic_field,
            confidence=0.95,
        )

    async def shutdown(self) -> None:
        """Shutdown IMU sensor."""
        self.logger.info("Shutting down IMU sensor")
        self.is_active = False


class OdometrySensor(SensorInterface):
    """Wheel odometry sensor interface."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.wheel_diameter = config.get("wheel_diameter", 0.1)  # meters
        self.wheel_base = config.get("wheel_base", 0.3)  # meters
        self.encoder_resolution = config.get("encoder_resolution", 1024)  # ticks/rev

        # Internal state for odometry integration
        self.last_encoder_left = 0
        self.last_encoder_right = 0
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)

    async def initialize(self) -> None:
        """Initialize odometry sensor."""
        self.logger.info("Initializing odometry sensor")
        await asyncio.sleep(0.1)  # Simulate initialization delay
        self.is_active = True
        self.logger.info("Odometry sensor initialized successfully")

    async def read_data(self) -> OdometryData:
        """Read odometry data."""
        if not self.is_active:
            raise RuntimeError("Odometry sensor not initialized")

        current_time = time.time()

        # Simulate encoder readings
        dt = 0.01  # 100 Hz

        # Simulate some motion
        left_encoder_delta = int(np.random.normal(0, 5))
        right_encoder_delta = int(np.random.normal(0, 5))

        # Calculate wheel velocities
        left_velocity = (
            (left_encoder_delta / self.encoder_resolution) * (np.pi * self.wheel_diameter) / dt
        )
        right_velocity = (
            (right_encoder_delta / self.encoder_resolution) * (np.pi * self.wheel_diameter) / dt
        )

        # Calculate robot velocity
        linear_velocity = (left_velocity + right_velocity) / 2.0
        angular_velocity = (right_velocity - left_velocity) / self.wheel_base

        # Integrate position (simplified)
        self.position[0] += linear_velocity * np.cos(self.orientation[2]) * dt
        self.position[1] += linear_velocity * np.sin(self.orientation[2]) * dt
        self.orientation[2] += angular_velocity * dt

        # Normalize yaw angle
        while self.orientation[2] > np.pi:
            self.orientation[2] -= 2 * np.pi
        while self.orientation[2] < -np.pi:
            self.orientation[2] += 2 * np.pi

        velocity = np.array(
            [
                linear_velocity * np.cos(self.orientation[2]),
                linear_velocity * np.sin(self.orientation[2]),
                0.0,
            ]
        )

        self.last_data_time = current_time

        return OdometryData(
            timestamp=current_time,
            position=self.position.copy(),
            velocity=velocity,
            orientation=self.orientation.copy(),
            confidence=0.90,
        )

    async def shutdown(self) -> None:
        """Shutdown odometry sensor."""
        self.logger.info("Shutting down odometry sensor")
        self.is_active = False


class SensorManager:
    """Manages multiple sensors and provides unified data access."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sensor manager.

        Args:
            config: Sensor configuration
        """
        self.logger = RobotLogger(__name__)
        self.config = config
        self.sensors: Dict[str, SensorInterface] = {}
        self.data_cache: Dict[str, SensorData] = {}
        self.is_running = False

        # Data acquisition tasks
        self._acquisition_tasks: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """Start sensor manager and all sensors."""
        self.logger.info("Starting sensor manager")

        # Initialize sensors based on configuration
        sensor_configs = self.config.get("sensors", {})

        for sensor_name, sensor_config in sensor_configs.items():
            sensor_type = sensor_config.get("type")

            sensor: SensorInterface
            if sensor_type == "imu":
                sensor = IMUSensor(sensor_config)
            elif sensor_type == "odometry":
                sensor = OdometrySensor(sensor_config)
            else:
                self.logger.warning(f"Unknown sensor type: {sensor_type}")
                continue

            # Initialize sensor
            await sensor.initialize()
            self.sensors[sensor_name] = sensor

            # Start data acquisition task
            self._acquisition_tasks[sensor_name] = asyncio.create_task(
                self._sensor_acquisition_loop(sensor_name, sensor)
            )

            self.logger.info(f"Started sensor: {sensor_name}")

        self.is_running = True
        self.logger.info("Sensor manager started successfully")

    async def stop(self) -> None:
        """Stop sensor manager and all sensors."""
        self.logger.info("Stopping sensor manager")

        self.is_running = False

        # Cancel acquisition tasks
        for task in self._acquisition_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._acquisition_tasks.values(), return_exceptions=True)

        # Shutdown sensors
        for sensor in self.sensors.values():
            await sensor.shutdown()

        self.logger.info("Sensor manager stopped")

    async def _sensor_acquisition_loop(self, sensor_name: str, sensor: SensorInterface) -> None:
        """Data acquisition loop for a specific sensor."""
        while self.is_running:
            try:
                # Read sensor data
                data = await sensor.read_data()

                # Cache the data
                self.data_cache[sensor_name] = data

                # Log data quality
                self.logger.log_sensor_data(sensor_name, data.confidence)

                # Wait before next reading
                await asyncio.sleep(0.01)  # 100 Hz

            except Exception as e:
                self.logger.error(f"Error reading from sensor {sensor_name}: {e}")
                await asyncio.sleep(0.1)  # Back off on error

    async def get_imu_data(self) -> Optional[IMUData]:
        """Get latest IMU data."""
        imu_data = self.data_cache.get("imu")
        if isinstance(imu_data, IMUData):
            return imu_data
        return None

    async def get_odometry_data(self) -> Optional[OdometryData]:
        """Get latest odometry data."""
        odometry_data = self.data_cache.get("odometry")
        if isinstance(odometry_data, OdometryData):
            return odometry_data
        return None

    def get_sensor_health(self) -> Dict[str, bool]:
        """Get health status of all sensors."""
        return {name: sensor.is_healthy for name, sensor in self.sensors.items()}

    def get_sensor_data_age(self, sensor_name: str) -> float:
        """
        Get age of sensor data in seconds.

        Args:
            sensor_name: Name of the sensor

        Returns:
            Age of data in seconds, or float('inf') if no data
        """
        data = self.data_cache.get(sensor_name)
        if data:
            return time.time() - data.timestamp
        return float("inf")
