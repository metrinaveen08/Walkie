#!/usr/bin/env python3
"""
Example script demonstrating basic robot control.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger


async def basic_control_example() -> None:
    """Example of basic robot control."""
    logger = setup_logger("example", level="INFO")
    logger.info("Starting basic robot control example")

    try:
        # This would normally run the main robot control system
        # For now, just demonstrate the structure
        logger.info("Robot control system would start here...")
        await asyncio.sleep(2.0)
        logger.info("Example completed successfully")

    except Exception as e:
        logger.error("Example failed: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(basic_control_example())
