"""
Configuration loader utility for the robot system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union

import yaml


class ConfigLoader:
    """Utility class for loading and validating configuration files."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If config is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")

        if config is None:
            config = {}

        # Validate basic structure
        ConfigLoader._validate_config(config)

        return config  # type: ignore[no-any-return]

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Check for required top-level sections
        required_sections = ["control", "hardware", "sensors"]

        for section in required_sections:
            if section not in config:
                logging.warning("Missing configuration section: %s", section)

        # Validate control section
        if "control" in config:
            control_config = config["control"]
            if "control_frequency" in control_config:
                freq = control_config["control_frequency"]
                if not isinstance(freq, (int, float)) or freq <= 0:
                    raise ValueError("control_frequency must be a positive number")

        # Validate hardware section
        if "hardware" in config:
            hardware_config = config["hardware"]
            if "type" not in hardware_config:
                logging.warning("Hardware type not specified")

        logging.info("Configuration validation completed")


class ConfigManager:
    """Manages configuration state and updates."""

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load()
        self.logger = logging.getLogger(__name__)

    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        return ConfigLoader.load_config(self.config_path)

    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self.load()
        self.logger.info("Configuration reloaded from %s", self.config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.logger.info("Configuration saved to %s", self.config_path)
