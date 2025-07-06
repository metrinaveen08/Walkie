"""
Test configuration and basic functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.state import RobotState, ControlCommand, TrajectoryPoint
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class TestRobotState:
    """Test robot state functionality."""

    def test_robot_state_creation(self):
        """Test creating robot state."""
        state = RobotState()
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.orientation.shape == (3,)

    def test_robot_state_properties(self):
        """Test robot state properties."""
        state = RobotState()
        state.position = np.array([1.0, 2.0, 3.0])
        state.orientation = np.array([0.0, 0.0, 1.57])

        assert state.x == 1.0
        assert state.y == 2.0
        assert state.z == 3.0
        assert abs(state.yaw - 1.57) < 0.001

    def test_distance_calculation(self):
        """Test distance calculation between states."""
        state1 = RobotState()
        state1.position = np.array([0.0, 0.0, 0.0])

        state2 = RobotState()
        state2.position = np.array([3.0, 4.0, 0.0])

        distance = state1.distance_to(state2)
        assert abs(distance - 5.0) < 0.001


class TestControlCommand:
    """Test control command functionality."""

    def test_control_command_creation(self):
        """Test creating control command."""
        cmd = ControlCommand()
        assert cmd.linear_velocity.shape == (3,)
        assert cmd.angular_velocity.shape == (3,)

    def test_control_command_values(self):
        """Test setting control command values."""
        cmd = ControlCommand()
        cmd.linear_velocity = np.array([1.0, 0.0, 0.0])
        cmd.angular_velocity = np.array([0.0, 0.0, 0.5])

        assert cmd.linear_velocity[0] == 1.0
        assert cmd.angular_velocity[2] == 0.5


class TestConfigLoader:
    """Test configuration loading."""

    def test_config_loader_hardware(self):
        """Test loading hardware configuration."""
        config_path = Path(__file__).parent.parent / "config" / "hardware.yaml"
        if config_path.exists():
            config = ConfigLoader.load_config(config_path)
            assert "control" in config
            assert "hardware" in config
            assert "sensors" in config

    def test_config_loader_simulation(self):
        """Test loading simulation configuration."""
        config_path = Path(__file__).parent.parent / "config" / "simulation.yaml"
        if config_path.exists():
            config = ConfigLoader.load_config(config_path)
            assert "control" in config
            assert "hardware" in config
            assert "physics" in config


class TestLogger:
    """Test logging functionality."""

    def test_logger_setup(self):
        """Test logger setup."""
        logger = setup_logger("test", level="INFO")
        assert logger is not None
        assert logger.name == "test"

    def test_logger_basic_functionality(self):
        """Test basic logger functionality."""
        logger = setup_logger("test", level="DEBUG")

        # These should not raise exceptions
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
