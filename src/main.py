"""
Main entry point for the agile dynamic robot control system.
"""

import asyncio
import argparse
from pathlib import Path

from .control.robot_controller import RobotController
from .hardware.robot_hardware import RobotHardware
from .planning.path_planner import PathPlanner
from .sensors.sensor_manager import SensorManager
from .vision.vision_system import VisionSystem
from .utils.config_loader import ConfigLoader
from .utils.logger import setup_logger


async def main() -> None:
    """Main entry point for the robot control system."""
    parser = argparse.ArgumentParser(description="Agile Dynamic Robot Control System")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hardware.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(level=args.log_level)
    logger.info("Starting Agile Dynamic Robot Control System")

    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = ConfigLoader.load_config(config_path)
        logger.info("Loaded configuration from %s", config_path)

        # Initialize components
        sensor_manager = SensorManager(config.get("sensors", {}))
        vision_system = VisionSystem(config.get("vision", {}))
        hardware = RobotHardware(config.get("hardware", {}), simulation=args.simulation)
        path_planner = PathPlanner(config.get("planning", {}))

        # Initialize main controller
        controller = RobotController(
            hardware=hardware,
            sensor_manager=sensor_manager,
            vision_system=vision_system,
            path_planner=path_planner,
            config=config.get("control", {}),
        )

        # Start the control loop
        logger.info("Starting robot control loop...")
        await controller.start()

        # Keep running until interrupted
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        raise
    finally:
        logger.info("Shutting down robot control system")
        if "controller" in locals():
            await controller.stop()


if __name__ == "__main__":
    asyncio.run(main())
