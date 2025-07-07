#!/usr/bin/env python3
"""
Safe simulation runner with resource monitoring and graceful shutdown.
"""

import asyncio
import signal
import sys
import time
import psutil
from typing import Optional

from src.simulation.simulator import SimpleSimulator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import RobotLogger


class SafeSimulationRunner:
    """Safely runs simulation with resource monitoring."""
    
    def __init__(self):
        self.logger = RobotLogger(__name__)
        self.simulator: Optional[SimpleSimulator] = None
        self.is_running = False
        self.start_time = time.time()
        
        # Resource limits
        self.max_cpu_percent = 80.0  # Don't use more than 80% CPU
        self.max_memory_mb = 500     # Limit to 500MB RAM
        self.max_run_time = 30       # Auto-stop after 30 seconds
        
    async def setup_simulation(self) -> bool:
        """Setup simulation with safe configuration."""
        try:
            # Load config with conservative settings
            config = ConfigLoader.load_config("config/simulation.yaml")
            
            # Override with safe settings
            safe_physics_config = {
                "timestep": 0.05,  # Slower timestep (20Hz instead of 100Hz)
                "gravity": -9.81,
                "friction_coefficient": 0.8,
                "workspace_size": [5.0, 5.0, 1.0]  # Smaller workspace
            }
            
            self.simulator = SimpleSimulator(safe_physics_config)
            self.logger.info("Simulation setup complete with safe parameters")
            return True
            
        except Exception as e:
            self.logger.error("Failed to setup simulation: %s", e)
            return False
    
    def check_system_resources(self) -> bool:
        """Check if system resources are within safe limits."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.max_cpu_percent:
                self.logger.warning("High CPU usage: %.1f%%", cpu_percent)
                return False
            
            # Check memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.max_memory_mb:
                self.logger.warning("High memory usage: %.1f MB", memory_mb)
                return False
            
            # Check runtime
            runtime = time.time() - self.start_time
            if runtime > self.max_run_time:
                self.logger.info("Max runtime reached: %.1f seconds", runtime)
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Resource check failed: %s", e)
            return False
    
    async def run_simulation(self) -> None:
        """Run simulation with safety monitoring."""
        if not await self.setup_simulation():
            return
        
        self.logger.info("Starting safe simulation...")
        self.logger.info("Resource limits: CPU < %.1f%%, Memory < %d MB, Time < %d sec", 
                        self.max_cpu_percent, self.max_memory_mb, self.max_run_time)
        
        self.is_running = True
        iteration = 0
        
        try:
            # Start simulator
            await self.simulator.start()
            
            # Main simulation loop with safety checks
            while self.is_running:
                # Check resources every 10 iterations
                if iteration % 10 == 0:
                    if not self.check_system_resources():
                        self.logger.warning("Resource limit exceeded, stopping simulation")
                        break
                
                # Run one simulation step
                await asyncio.sleep(0.05)  # 20Hz update rate
                
                iteration += 1
                
                # Log progress every 100 iterations
                if iteration % 100 == 0:
                    runtime = time.time() - self.start_time
                    self.logger.info("Simulation running: %.1f sec, iteration %d", 
                                   runtime, iteration)
        
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        except Exception as e:
            self.logger.error("Simulation error: %s", e)
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean shutdown of simulation."""
        self.is_running = False
        if self.simulator:
            await self.simulator.stop()
        
        runtime = time.time() - self.start_time
        self.logger.info("Simulation stopped after %.1f seconds", runtime)
    
    def setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on signals."""
        def signal_handler(signum, frame):
            self.logger.info("Received signal %d, shutting down gracefully...", signum)
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for safe simulation."""
    runner = SafeSimulationRunner()
    runner.setup_signal_handlers()
    
    print("🤖 Starting SAFE Walkie Simulation")
    print("📊 Resource monitoring enabled")
    print("⏱️  Auto-stop after 30 seconds")
    print("🛑 Press Ctrl+C to stop early")
    print("-" * 50)
    
    await runner.run_simulation()
    
    print("-" * 50)
    print("✅ Simulation completed safely!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        sys.exit(1)