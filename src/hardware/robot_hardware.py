"""
Hardware interface for robot actuators and sensors.
"""
import asyncio
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

import numpy as np

from ..utils.logger import RobotLogger


class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces."""
    
    def __init__(self, config: Dict[str, Any], simulation: bool = False):
        """
        Initialize hardware interface.
        
        Args:
            config: Hardware configuration
            simulation: Whether to run in simulation mode
        """
        self.config = config
        self.simulation = simulation
        self.logger = RobotLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_initialized = False
        self.emergency_stop_active = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize hardware."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown hardware."""
        pass
    
    @abstractmethod
    async def send_velocity_command(self, linear: float, angular: float) -> None:
        """Send velocity command to robot."""
        pass
    
    @abstractmethod
    async def get_system_status(self) -> Dict[str, Any]:
        """Get hardware system status."""
        pass
    
    @abstractmethod
    async def emergency_stop(self) -> None:
        """Activate emergency stop."""
        pass
    
    @abstractmethod
    async def release_emergency_stop(self) -> None:
        """Release emergency stop."""
        pass


class DifferentialDriveHardware(HardwareInterface):
    """Hardware interface for differential drive robots."""
    
    def __init__(self, config: Dict[str, Any], simulation: bool = False):
        super().__init__(config, simulation)
        
        # Robot parameters
        self.wheel_base = config.get("wheel_base", 0.3)  # meters
        self.wheel_radius = config.get("wheel_radius", 0.05)  # meters
        self.max_wheel_velocity = config.get("max_wheel_velocity", 10.0)  # rad/s
        
        # Motor controllers
        self.left_motor_port = config.get("left_motor_port", "/dev/ttyUSB0")
        self.right_motor_port = config.get("right_motor_port", "/dev/ttyUSB1")
        
        # Internal state
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.battery_voltage = 12.0
        self.motor_temperatures = {"left": 25.0, "right": 25.0}
        self.controller_temperature = 25.0
        
        # Simulation state
        self.sim_position = np.zeros(3)
        self.sim_orientation = np.zeros(3)
        self.sim_velocity = np.zeros(3)
        
    async def initialize(self) -> None:
        """Initialize differential drive hardware."""
        self.logger.info("Initializing differential drive hardware")
        
        if self.simulation:
            self.logger.info("Running in simulation mode")
        else:
            # Initialize actual hardware
            await self._initialize_motor_controllers()
            await self._initialize_sensors()
        
        self.is_initialized = True
        self.logger.info("Differential drive hardware initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown differential drive hardware."""
        self.logger.info("Shutting down differential drive hardware")
        
        # Stop motors
        await self.send_velocity_command(0.0, 0.0)
        
        if not self.simulation:
            # Shutdown actual hardware
            await self._shutdown_motor_controllers()
        
        self.is_initialized = False
        self.logger.info("Differential drive hardware shutdown complete")
    
    async def send_velocity_command(self, linear: float, angular: float) -> None:
        """
        Send velocity command to differential drive robot.
        
        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        if not self.is_initialized:
            raise RuntimeError("Hardware not initialized")
        
        if self.emergency_stop_active:
            linear = 0.0
            angular = 0.0
        
        # Convert to wheel velocities
        left_velocity, right_velocity = self._differential_drive_kinematics(linear, angular)
        
        # Apply limits
        left_velocity = np.clip(left_velocity, -self.max_wheel_velocity, self.max_wheel_velocity)
        right_velocity = np.clip(right_velocity, -self.max_wheel_velocity, self.max_wheel_velocity)
        
        if self.simulation:
            # Update simulation state
            await self._update_simulation_state(linear, angular)
        else:
            # Send commands to actual motors
            await self._send_motor_commands(left_velocity, right_velocity)
        
        self.current_linear_velocity = linear
        self.current_angular_velocity = angular
        
        self.logger.debug(f"Velocity command: linear={linear:.3f}, angular={angular:.3f}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get hardware system status."""
        return {
            "battery_voltage": self.battery_voltage,
            "motor_temperature": max(self.motor_temperatures.values()),
            "controller_temperature": self.controller_temperature,
            "emergency_stop_active": self.emergency_stop_active,
            "linear_velocity": self.current_linear_velocity,
            "angular_velocity": self.current_angular_velocity,
            "simulation_mode": self.simulation
        }
    
    async def emergency_stop(self) -> None:
        """Activate emergency stop."""
        self.logger.warning("Emergency stop activated")
        self.emergency_stop_active = True
        await self.send_velocity_command(0.0, 0.0)
    
    async def release_emergency_stop(self) -> None:
        """Release emergency stop."""
        self.logger.info("Emergency stop released")
        self.emergency_stop_active = False
    
    def _differential_drive_kinematics(self, linear: float, angular: float) -> Tuple[float, float]:
        """
        Convert linear and angular velocities to wheel velocities.
        
        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
            
        Returns:
            Left and right wheel velocities (rad/s)
        """
        # Differential drive kinematics
        left_velocity = (linear - angular * self.wheel_base / 2.0) / self.wheel_radius
        right_velocity = (linear + angular * self.wheel_base / 2.0) / self.wheel_radius
        
        return left_velocity, right_velocity
    
    async def _initialize_motor_controllers(self) -> None:
        """Initialize motor controllers."""
        self.logger.info("Initializing motor controllers")
        # In real implementation, initialize serial connections to motor controllers
        await asyncio.sleep(0.1)  # Simulate initialization delay
    
    async def _initialize_sensors(self) -> None:
        """Initialize hardware sensors."""
        self.logger.info("Initializing hardware sensors")
        # In real implementation, initialize voltage/temperature sensors
        await asyncio.sleep(0.1)  # Simulate initialization delay
    
    async def _shutdown_motor_controllers(self) -> None:
        """Shutdown motor controllers."""
        self.logger.info("Shutting down motor controllers")
        # In real implementation, close serial connections
        await asyncio.sleep(0.1)  # Simulate shutdown delay
    
    async def _send_motor_commands(self, left_velocity: float, right_velocity: float) -> None:
        """
        Send velocity commands to motors.
        
        Args:
            left_velocity: Left wheel velocity (rad/s)
            right_velocity: Right wheel velocity (rad/s)
        """
        # In real implementation, send serial commands to motor controllers
        # TODO: Implement actual motor communication
        _ = left_velocity, right_velocity  # Placeholder for unused arguments
        await asyncio.sleep(0.001)  # Simulate communication delay
    
    async def _update_simulation_state(self, linear: float, angular: float) -> None:
        """
        Update simulation state based on velocity commands.
        
        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        dt = 0.01  # Integration time step
        
        # Update position and orientation
        self.sim_position[0] += linear * np.cos(self.sim_orientation[2]) * dt
        self.sim_position[1] += linear * np.sin(self.sim_orientation[2]) * dt
        self.sim_orientation[2] += angular * dt
        
        # Normalize yaw angle
        while self.sim_orientation[2] > np.pi:
            self.sim_orientation[2] -= 2 * np.pi
        while self.sim_orientation[2] < -np.pi:
            self.sim_orientation[2] += 2 * np.pi
        
        # Update velocity
        self.sim_velocity[0] = linear * np.cos(self.sim_orientation[2])
        self.sim_velocity[1] = linear * np.sin(self.sim_orientation[2])
        
        # Simulate battery drain
        power_consumption = abs(linear) * 2.0 + abs(angular) * 1.0
        self.battery_voltage -= power_consumption * dt * 0.001
        self.battery_voltage = max(10.0, self.battery_voltage)
        
        # Simulate temperature changes
        for motor in self.motor_temperatures:
            target_temp = 25.0 + power_consumption * 5.0
            self.motor_temperatures[motor] += (target_temp - self.motor_temperatures[motor]) * dt * 0.1
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            "position": self.sim_position.copy(),
            "orientation": self.sim_orientation.copy(),
            "velocity": self.sim_velocity.copy()
        }


class RobotHardware:
    """Main hardware interface managing different robot types."""
    
    def __init__(self, config: Dict[str, Any], simulation: bool = False):
        """
        Initialize robot hardware.
        
        Args:
            config: Hardware configuration
            simulation: Whether to run in simulation mode
        """
        self.logger = RobotLogger(__name__)
        self.config = config
        self.simulation = simulation
        
        # Create appropriate hardware interface
        hardware_type = config.get("type", "differential_drive")
        
        if hardware_type == "differential_drive":
            self.hardware = DifferentialDriveHardware(config, simulation)
        else:
            raise ValueError(f"Unsupported hardware type: {hardware_type}")
        
        self.logger.info(f"Created {hardware_type} hardware interface")
    
    async def initialize(self) -> None:
        """Initialize robot hardware."""
        await self.hardware.initialize()
    
    async def shutdown(self) -> None:
        """Shutdown robot hardware."""
        await self.hardware.shutdown()
    
    async def send_velocity_command(self, linear: float, angular: float) -> None:
        """Send velocity command to robot."""
        await self.hardware.send_velocity_command(linear, angular)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get hardware system status."""
        return await self.hardware.get_system_status()
    
    async def emergency_stop(self) -> None:
        """Activate emergency stop."""
        await self.hardware.emergency_stop()
    
    async def release_emergency_stop(self) -> None:
        """Release emergency stop."""
        await self.hardware.release_emergency_stop()
