#!/usr/bin/env python3
"""
Visual simulation runner with real-time plotting and robot visualization.

This module provides a comprehensive real-time visualization system for the
agile dynamic robot simulation, featuring matplotlib-based plotting with
performance monitoring and safety constraints.
"""

import asyncio
import signal
import sys
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
import psutil

# Fix import path to match project structure
from src.simulation.simulator import SimpleSimulator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import RobotLogger
from src.utils.state import RobotState


@dataclass
class VisualizationConfig:
    """Configuration parameters for simulation visualization.
    
    This dataclass encapsulates all visualization settings following the
    composition over inheritance principle and provides type safety.
    
    Attributes:
        window_size: Tuple of (width, height) for matplotlib figure
        update_rate_hz: Visualization update frequency in Hz
        trail_length: Maximum number of trail points to display
        show_velocity_vectors: Whether to display velocity arrows
        show_workspace_bounds: Whether to display workspace boundaries
        robot_radius: Physical radius of the robot in meters
        velocity_scale: Scaling factor for velocity vector display
    """
    window_size: Tuple[int, int] = (14, 8)
    update_rate_hz: float = 30.0
    trail_length: int = 300
    show_velocity_vectors: bool = True
    show_workspace_bounds: bool = True
    robot_radius: float = 0.1
    velocity_scale: float = 2.0


class RealTimeVisualizer:
    """Real-time matplotlib visualization for robot simulation.
    
    This class implements the observer pattern for real-time robot state
    visualization with performance monitoring and safety constraints.
    Follows composition over inheritance with proper encapsulation.
    """
    
    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize the real-time visualizer.
        
        Args:
            config: Visualization configuration parameters
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate input parameters (safety guideline)
        if config.update_rate_hz <= 0:
            raise ValueError("Update rate must be positive")
        if config.trail_length <= 0:
            raise ValueError("Trail length must be positive")
        if config.robot_radius <= 0:
            raise ValueError("Robot radius must be positive")
            
        self.config = config
        self.logger = RobotLogger(__name__)
        
        # Initialize matplotlib with proper backend
        plt.ion()  # Interactive mode for real-time updates
        
        # Create figure with specified dimensions
        self.fig, (self.ax_main, self.ax_metrics) = plt.subplots(
            1, 2, figsize=self.config.window_size
        )
        
        # Initialize data containers with proper typing
        self.robot_trail_x: List[float] = []
        self.robot_trail_y: List[float] = []
        self.timestamps: List[float] = []
        self.velocities: List[float] = []
        
        # Setup visualization components
        self._setup_main_plot()
        self._setup_metrics_plot()
        
        # Initialize animation elements with proper typing
        self.robot_circle: Optional[Circle] = None
        self.velocity_arrow: Optional[Annotation] = None
        self.trail_line: Optional[Line2D] = None
        self.velocity_line: Optional[Line2D] = None
        
        self.logger.info("Real-time visualizer initialized successfully")
    
    def _setup_main_plot(self) -> None:
        """Setup the main robot workspace visualization.
        
        Configures the primary plot area with proper scaling, grid, and
        robot visualization elements following the project's safety guidelines.
        """
        # Set workspace bounds with safety margins
        workspace_limit = 3.0
        self.ax_main.set_xlim(-workspace_limit, workspace_limit)
        self.ax_main.set_ylim(-workspace_limit, workspace_limit)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        
        # Configure plot appearance
        self.ax_main.set_title(
            '🤖 Walkie Robot Simulation',
            fontsize=14,
            fontweight='bold'
        )
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Y Position (m)')
        
        # Add workspace boundaries if enabled
        if self.config.show_workspace_bounds:
            boundary_size = workspace_limit - 0.5  # Safety margin
            workspace_rect = Rectangle(
                (-boundary_size, -boundary_size),
                2 * boundary_size,
                2 * boundary_size,
                fill=False,
                edgecolor='red',
                linewidth=2,
                linestyle='--'
            )
            self.ax_main.add_patch(workspace_rect)
            self.ax_main.text(
                -boundary_size + 0.1,
                boundary_size + 0.2,
                'Workspace Boundary',
                color='red',
                fontsize=10
            )
        
        # Initialize robot visualization
        self.robot_circle = Circle(
            (0, 0),
            self.config.robot_radius,
            color='blue',
            alpha=0.8
        )
        self.ax_main.add_patch(self.robot_circle)
        
        # Initialize trail line
        self.trail_line, = self.ax_main.plot(
            [], [],
            'b-',
            alpha=0.6,
            linewidth=2,
            label='Robot Trail'
        )
        
        # Initialize velocity vector if enabled
        if self.config.show_velocity_vectors:
            self.velocity_arrow = self.ax_main.annotate(
                '',
                xy=(0, 0),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10,
                color='green'
            )
        
        self.ax_main.legend(loc='upper right')
    
    def _setup_metrics_plot(self) -> None:
        """Setup the performance metrics visualization.
        
        Configures the metrics plot for real-time performance monitoring
        with proper scaling and labeling.
        """
        self.ax_metrics.set_title(
            '📊 Performance Metrics',
            fontsize=14,
            fontweight='bold'
        )
        self.ax_metrics.set_xlabel('Time (s)')
        self.ax_metrics.set_ylabel('Velocity (m/s)')
        self.ax_metrics.grid(True, alpha=0.3)
        
        # Initialize velocity plot
        self.velocity_line, = self.ax_metrics.plot(
            [], [],
            'g-',
            linewidth=2,
            label='Speed'
        )
        self.ax_metrics.legend()
        self.ax_metrics.set_ylim(0, 1.0)  # Initial reasonable range
    
    def update_robot_state(self, state: RobotState, timestamp: float) -> None:
        """Update visualization with new robot state.
        
        Args:
            state: Current robot state with position and velocity
            timestamp: Current simulation timestamp
            
        Raises:
            ValueError: If state parameters are invalid
        """
        # Validate input parameters (safety guideline)
        if state.position is None or len(state.position) < 2:
            raise ValueError("Invalid robot state position")
        if state.velocity is None or len(state.velocity) < 2:
            raise ValueError("Invalid robot state velocity")
        
        # Add to trail data with proper type conversion
        self.robot_trail_x.append(float(state.position[0]))
        self.robot_trail_y.append(float(state.position[1]))
        self.timestamps.append(timestamp)
        
        # Calculate 2D speed using NumPy vectorization
        velocity_2d = np.array(state.velocity[:2])
        speed = float(np.linalg.norm(velocity_2d))
        self.velocities.append(speed)
        
        # Maintain trail length for memory efficiency
        if len(self.robot_trail_x) > self.config.trail_length:
            self.robot_trail_x.pop(0)
            self.robot_trail_y.pop(0)
            self.timestamps.pop(0)
            self.velocities.pop(0)
    
    def render_frame(self) -> None:
        """Render one frame of the visualization.
        
        Updates all visualization elements with current robot state data.
        Implements proper error handling for robust operation.
        """
        if not self.robot_trail_x:
            return
        
        try:
            # Update robot position
            current_x = self.robot_trail_x[-1]
            current_y = self.robot_trail_y[-1]
            
            if self.robot_circle is not None:
                self.robot_circle.center = (current_x, current_y)
            
            # Update trail
            if self.trail_line is not None:
                self.trail_line.set_data(self.robot_trail_x, self.robot_trail_y)
            
            # Update velocity vector
            if (self.config.show_velocity_vectors and 
                self.velocity_arrow is not None and
                len(self.robot_trail_x) >= 2):
                
                # Calculate velocity direction from position derivatives
                dx = self.robot_trail_x[-1] - self.robot_trail_x[-2]
                dy = self.robot_trail_y[-1] - self.robot_trail_y[-2]
                
                # Scale velocity vector for visibility
                end_x = current_x + dx * self.config.velocity_scale
                end_y = current_y + dy * self.config.velocity_scale
                
                self.velocity_arrow.xy = (end_x, end_y)
                self.velocity_arrow.xytext = (current_x, current_y)
            
            # Update metrics plot
            if len(self.timestamps) > 1 and self.velocity_line is not None:
                # Normalize timestamps to start from zero
                rel_times = [t - self.timestamps[0] for t in self.timestamps]
                
                self.velocity_line.set_data(rel_times, self.velocities)
                
                # Auto-scale metrics plot
                if rel_times and self.velocities:
                    max_time = max(rel_times)
                    max_vel = max(self.velocities)
                    
                    self.ax_metrics.set_xlim(0, max_time)
                    self.ax_metrics.set_ylim(0, max(max_vel * 1.1, 0.1))
            
            # Update display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.logger.error(f"Error rendering visualization frame: {e}")
    
    def add_status_text(self, iteration: int, runtime: float,
                       cpu_percent: float, memory_mb: float) -> None:
        """Add real-time status information to the visualization.
        
        Args:
            iteration: Current simulation iteration count
            runtime: Total simulation runtime in seconds
            cpu_percent: Current CPU usage percentage
            memory_mb: Current memory usage in megabytes
        """
        status_text = (
            f"Iteration: {iteration:,}\n"
            f"Runtime: {runtime:.1f}s\n"
            f"CPU: {cpu_percent:.1f}%\n"
            f"Memory: {memory_mb:.1f}MB"
        )
        
        # Remove previous status text to prevent overlap
        texts_to_remove = [
            txt for txt in self.ax_main.texts
            if txt.get_position()[0] < -2
        ]
        for txt in texts_to_remove:
            txt.remove()
        
        # Add new status text
        self.ax_main.text(
            -2.9, -2.9,
            status_text,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="lightgray",
                alpha=0.8
            ),
            fontsize=9,
            verticalalignment='bottom',
            fontfamily='monospace'
        )


class VisualSimulationRunner:
    """Simulation runner with real-time visualization.
    
    This class implements the main simulation loop with proper async/await
    patterns, resource monitoring, and safety constraints following the
    project's architecture guidelines.
    """
    
    def __init__(self) -> None:
        """Initialize the visual simulation runner."""
        self.logger = RobotLogger(__name__)
        self.simulator: Optional[SimpleSimulator] = None
        self.visualizer: Optional[RealTimeVisualizer] = None
        self.is_running = False
        self.start_time = time.time()
        
        # Resource limits for safety and reliability
        self.max_cpu_percent = 75.0
        self.max_memory_mb = 400
        self.max_run_time = 60.0  # Extended time for visual demo
    
    async def setup_simulation(self) -> bool:
        """Setup simulation and visualization components.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Load configuration with validation
            config = ConfigLoader.load_config("config/simulation.yaml")
            
            # Setup simulator with physics parameters
            physics_config: Dict[str, Any] = {
                "timestep": 1.0 / 30.0,  # 30Hz for smooth visuals
                "gravity": -9.81,
                "friction_coefficient": 0.2,
                "workspace_size": [5.0, 5.0, 1.0]
            }
            
            self.simulator = SimpleSimulator(physics_config)
            
            # Setup visualizer with optimal configuration
            viz_config = VisualizationConfig(
                window_size=(16, 9),
                update_rate_hz=30.0,
                trail_length=500,
                show_velocity_vectors=True,
                show_workspace_bounds=True,
                robot_radius=0.1,
                velocity_scale=3.0
            )
            self.visualizer = RealTimeVisualizer(viz_config)
            
            self.logger.info("Visual simulation setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup visual simulation: {e}")
            return False
    
    def check_system_resources(self) -> bool:
        """Check system resources with proper error handling.
        
        Returns:
            True if resources are within limits, False otherwise
        """
        try:
            # Monitor CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.01)
            
            # Monitor memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Check runtime
            runtime = time.time() - self.start_time
            
            # Apply safety limits
            if cpu_percent > self.max_cpu_percent:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                return False
            
            if memory_mb > self.max_memory_mb:
                self.logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                return False
            
            if runtime > self.max_run_time:
                self.logger.info(f"Max runtime reached: {runtime:.1f} seconds")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
            return True  # Continue on resource check errors
    
    def generate_interesting_motion(self, t: float) -> np.ndarray:
        """Generate robot motion using Lissajous curves.
        
        Mathematical formulation:
        x(t) = A * cos(ωt + φ) + perturbation
        y(t) = B * sin(ωt * ratio + ψ) + perturbation
        
        Args:
            t: Current simulation time in seconds
            
        Returns:
            3D velocity vector [vx, vy, vz] in m/s
        """
        # Base motion parameters
        base_speed = 0.8  # m/s
        frequency = 0.5   # Hz
        ratio = 2.0       # Frequency ratio for Lissajous curves
        
        # Generate Lissajous curve motion
        x_vel = base_speed * np.cos(frequency * t)
        y_vel = base_speed * np.sin(frequency * ratio * t)
        
        # Add harmonic perturbations for natural motion
        perturbation_amplitude = 0.15
        x_vel += perturbation_amplitude * np.sin(t * 2.5 + np.pi/3)
        y_vel += perturbation_amplitude * np.cos(t * 1.8 + np.pi/6)
        
        return np.array([x_vel, y_vel, 0.0], dtype=np.float64)
    
    async def run_visual_simulation(self) -> None:
        """Run simulation with real-time visualization.
        
        Implements the main simulation loop with proper async/await patterns,
        resource monitoring, and error handling.
        """
        if not await self.setup_simulation():
            self.logger.error("Simulation setup failed")
            return
        
        self.logger.info("Starting visual simulation")
        print("🎬 Visual simulation starting - matplotlib window opening...")
        print("🤖 Robot performing Lissajous curve motion patterns")
        print("📊 Real-time performance metrics displayed")
        print("🛑 Close the plot window or press Ctrl+C to stop")
        
        self.is_running = True
        iteration = 0
        
        try:
            # Start simulator
            if self.simulator is not None:
                await self.simulator.start()
            
            # Main simulation loop
            while self.is_running:
                current_time = time.time()
                sim_time = current_time - self.start_time
                
                # Generate motion using mathematical formulation
                target_velocity = self.generate_interesting_motion(sim_time)
                
                # Update robot state with physics integration
                if self.simulator is not None:
                    self.simulator.robot_state.velocity = target_velocity
                    
                    # Numerical integration using Euler method
                    dt = 1.0 / 30.0
                    position_delta = target_velocity * dt
                    self.simulator.robot_state.position += position_delta
                    
                    # Apply workspace boundary constraints
                    workspace_limit = 2.3
                    pos = self.simulator.robot_state.position
                    pos[0] = np.clip(pos[0], -workspace_limit, workspace_limit)
                    pos[1] = np.clip(pos[1], -workspace_limit, workspace_limit)
                    
                    # Update visualization
                    if self.visualizer is not None:
                        self.visualizer.update_robot_state(
                            self.simulator.robot_state,
                            sim_time
                        )
                
                # Render visualization frame
                if iteration % 1 == 0 and self.visualizer is not None:
                    self.visualizer.render_frame()
                    
                    # Add status information
                    cpu_percent = psutil.cpu_percent(interval=0.01)
                    memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    self.visualizer.add_status_text(
                        iteration,
                        sim_time,
                        cpu_percent,
                        memory_mb
                    )
                
                # Periodic resource monitoring
                if iteration % 30 == 0:
                    if not self.check_system_resources():
                        self.logger.warning("Resource limit exceeded, stopping")
                        break
                
                # Check if visualization window was closed
                if not plt.get_fignums():
                    self.logger.info("Visualization window closed, stopping")
                    break
                
                # Maintain consistent timing
                await asyncio.sleep(dt)
                iteration += 1
                
                # Periodic progress logging
                if iteration % 300 == 0:
                    self.logger.info(
                        f"Visual simulation running: {sim_time:.1f} sec, {iteration} iterations"
                    )
        
        except KeyboardInterrupt:
            self.logger.info("Visual simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"Visual simulation error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean shutdown with proper resource management."""
        self.is_running = False
        
        if self.simulator is not None:
            await self.simulator.stop()
        
        runtime = time.time() - self.start_time
        self.logger.info(f"Visual simulation stopped after {runtime:.1f} seconds")
        
        # Keep plot open for a moment
        if plt.get_fignums():
            print("\n🎉 Simulation complete! Close the plot window to exit.")
            plt.savefig('simulation_plot.png', dpi=150, bbox_inches='tight')
            print("Plot saved as simulation_plot.png")


async def main() -> None:
    """Main entry point for visual simulation."""
    runner = VisualSimulationRunner()
    
    # Setup signal handlers
    def signal_handler(signum: int, frame: Any) -> None:
        print(f"\n⚠️  Received signal {signum}, shutting down gracefully...")
        runner.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🤖 Starting VISUAL Walkie Simulation")
    print("🎬 Real-time robot visualization with matplotlib")
    print("📊 Live performance metrics and trail tracking")
    print("⏱️  Auto-stop after 60 seconds")
    print("🛑 Close plot window or press Ctrl+C to stop early")
    print("=" * 60)
    
    await runner.run_visual_simulation()
    
    print("=" * 60)
    print("✅ Visual simulation completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Visual simulation failed: {e}")
        sys.exit(1)