"""
Logging configuration and utilities for the robot system.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (None for root logger)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console: Whether to log to console

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging from configuration dictionary.

    Args:
        config: Logging configuration
    """
    # Default logging configuration
    default_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(module)s - %(funcName)s - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": "logs/robot.log",
                "mode": "a",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "src.control": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": False,
            },
            "src.safety": {
                "handlers": ["console", "file"],
                "level": "WARNING",
                "propagate": False,
            },
        },
    }

    # Merge with provided config
    if config:
        default_config.update(config)

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.config.dictConfig(default_config)


class RobotLogger:
    """Enhanced logger for robot operations with structured logging."""

    def __init__(self, name: str):
        """
        Initialize robot logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs: Any) -> None:
        """Set logging context that will be included in all log messages."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear logging context."""
        self.context.clear()

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{context_str}] {message}"
        return message

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message), **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message), **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message), **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message), **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message), **kwargs)

    def log_state(self, state: Any, prefix: str = "State") -> None:
        """Log robot state information."""
        self.debug(
            f"{prefix}: pos=({state.x:.3f}, {state.y:.3f}, {state.z:.3f}), "
            f"vel=({state.velocity[0]:.3f}, {state.velocity[1]:.3f}), "
            f"yaw={state.yaw:.3f}"
        )

    def log_performance(self, operation: str, duration: float, success: bool = True) -> None:
        """Log performance metrics."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Performance: {operation} - {status} - {duration:.3f}s")

    def log_safety_event(self, event: str, severity: str = "WARNING") -> None:
        """Log safety-related events."""
        log_method = getattr(self.logger, severity.lower())
        log_method(f"SAFETY: {event}")

    def log_sensor_data(self, sensor_name: str, data_quality: float) -> None:
        """Log sensor data quality."""
        self.debug(f"Sensor {sensor_name}: quality={data_quality:.3f}")

    def log_trajectory_point(self, point_idx: int, position: Any, velocity: Any) -> None:
        """Log trajectory following information."""
        self.debug(
            f"Trajectory point {point_idx}: "
            f"pos=({position[0]:.3f}, {position[1]:.3f}), "
            f"vel=({velocity[0]:.3f}, {velocity[1]:.3f})"
        )
