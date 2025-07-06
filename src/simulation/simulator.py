"""
Basic simulation environment for the agile dynamic robot.
"""

import asyncio
import time
from typing import Dict, Any, List

import numpy as np

from ..utils.logger import RobotLogger
from ..utils.state import RobotState


class SimpleSimulator:
    """Simple physics simulator for robot testing."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulator.

        Args:
            config: Simulation configuration
        """
        self.logger = RobotLogger(__name__)
        self.config = config

        # Simulation parameters
        self.timestep = config.get("timestep", 0.01)  # seconds
        self.gravity = config.get("gravity", -9.81)  # m/s^2
        self.friction = config.get("friction_coefficient", 0.8)

        # Environment
        self.workspace_size = config.get("workspace_size", [10.0, 10.0, 2.0])
        self.obstacles: List[Dict[str, np.ndarray]] = []

        # Robot state
        self.robot_state = RobotState()
        self.is_running = False

        # Simulation time
        self.sim_time = 0.0
        self.real_start_time = 0.0

        self.logger.info("Simple simulator initialized")

    async def start(self) -> None:
        """Start the simulation."""
        self.logger.info("Starting simulation")
        self.is_running = True
        self.sim_time = 0.0
        self.real_start_time = time.time()

        # Initialize robot at origin
        self.robot_state.position = np.array([0.0, 0.0, 0.0])
        self.robot_state.orientation = np.array([0.0, 0.0, 0.0])

        # Start simulation loop
        asyncio.create_task(self._simulation_loop())

    async def stop(self) -> None:
        """Stop the simulation."""
        self.logger.info("Stopping simulation")
        self.is_running = False

    async def _simulation_loop(self) -> None:
        """Main simulation loop."""
        while self.is_running:
            start_time = time.time()

            # Update simulation
            self._update_physics(self.timestep)
            self.sim_time += self.timestep

            # Control simulation speed
            elapsed = time.time() - start_time
            sleep_time = max(0, self.timestep - elapsed)
            await asyncio.sleep(sleep_time)

    def _update_physics(self, dt: float) -> None:
        """
        Update physics simulation.

        Args:
            dt: Time step in seconds
        """
        # Simple integration of robot dynamics
        # In practice, this would use sophisticated physics engine

        # Update position from velocity
        self.robot_state.position += self.robot_state.velocity * dt

        # Update orientation from angular velocity
        self.robot_state.orientation += self.robot_state.angular_velocity * dt

        # Apply friction to velocity
        friction_force = -self.friction * self.robot_state.velocity
        self.robot_state.velocity += friction_force * dt

        # Boundary constraints
        for i in range(3):
            if self.robot_state.position[i] < -self.workspace_size[i] / 2:
                self.robot_state.position[i] = -self.workspace_size[i] / 2
                self.robot_state.velocity[i] = 0
            elif self.robot_state.position[i] > self.workspace_size[i] / 2:
                self.robot_state.position[i] = self.workspace_size[i] / 2
                self.robot_state.velocity[i] = 0

    def get_robot_state(self) -> RobotState:
        """Get current robot state."""
        return self.robot_state.copy()

    def set_robot_velocity(self, linear: np.ndarray, angular: np.ndarray) -> None:
        """Set robot velocity."""
        self.robot_state.velocity = linear.copy()
        self.robot_state.angular_velocity = angular.copy()

    def add_obstacle(self, position: np.ndarray, size: np.ndarray) -> None:
        """Add obstacle to simulation."""
        self.obstacles.append({"position": position, "size": size})

    def get_simulation_time(self) -> float:
        """Get current simulation time."""
        return self.sim_time


async def main() -> None:
    """Main entry point for simulation."""
    from ..utils.config_loader import ConfigLoader

    # Load simulation configuration
    config = ConfigLoader.load_config("config/simulation.yaml")

    # Create and start simulator
    simulator = SimpleSimulator(config.get("physics", {}))
    await simulator.start()

    # Run for a while
    await asyncio.sleep(10.0)

    await simulator.stop()


if __name__ == "__main__":
    asyncio.run(main())
