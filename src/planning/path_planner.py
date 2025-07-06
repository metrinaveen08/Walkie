"""
Path planning system for dynamic navigation.
"""

from typing import Dict, Any, Optional, List
import time

import numpy as np
from scipy.spatial.distance import euclidean

from ..utils.state import RobotState, TrajectoryPoint
from ..utils.logger import RobotLogger


class PathPlanner:
    """Dynamic path planning system with obstacle avoidance."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize path planner.

        Args:
            config: Planning configuration
        """
        self.logger = RobotLogger(__name__)
        self.config = config

        # Planning parameters
        self.planning_horizon = config.get("planning_horizon", 5.0)  # seconds
        self.replanning_threshold = config.get("replanning_threshold", 0.5)  # meters
        self.goal_tolerance = config.get("goal_tolerance", 0.1)  # meters
        self.max_velocity = config.get("max_velocity", 1.0)  # m/s
        self.max_acceleration = config.get("max_acceleration", 2.0)  # m/s^2

        # RRT* parameters
        self.rrt_max_iterations = config.get("rrt_max_iterations", 1000)
        self.rrt_step_size = config.get("rrt_step_size", 0.2)  # meters
        self.rrt_goal_bias = config.get("rrt_goal_bias", 0.1)  # probability
        self.rrt_neighbor_radius = config.get("rrt_neighbor_radius", 0.5)  # meters

        # Current path and planning state
        self.current_path: List[np.ndarray] = []
        self.last_goal: Optional[np.ndarray] = None
        self.last_planning_time = 0.0
        self.planning_frequency = config.get("planning_frequency", 10.0)  # Hz

    async def plan_path(
        self,
        start_state: RobotState,
        goal_state: RobotState,
        obstacle_map: Optional[np.ndarray] = None,
    ) -> List[TrajectoryPoint]:
        """
        Plan path from start to goal avoiding obstacles.

        Args:
            start_state: Starting robot state
            goal_state: Goal robot state
            obstacle_map: Occupancy grid with obstacles

        Returns:
            List of trajectory points
        """
        start_time = time.time()

        start_pos = start_state.position[:2]  # x, y only
        goal_pos = goal_state.position[:2]

        self.logger.debug(f"Planning path from {start_pos} to {goal_pos}")

        # Use RRT* for path planning
        path_points = self._rrt_star_planning(start_pos, goal_pos, obstacle_map)

        if not path_points:
            self.logger.warning("No path found!")
            return []

        # Convert to trajectory points with timing
        trajectory = self._path_to_trajectory(path_points, start_state)

        planning_time = time.time() - start_time
        self.logger.log_performance("path_planning", planning_time)

        return trajectory

    def needs_replanning(
        self,
        current_state: RobotState,
        goal_state: RobotState,
        obstacle_map: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Check if replanning is needed.

        Args:
            current_state: Current robot state
            goal_state: Goal robot state
            obstacle_map: Current obstacle map

        Returns:
            True if replanning is needed
        """
        current_time = time.time()

        # Check planning frequency
        if current_time - self.last_planning_time < 1.0 / self.planning_frequency:
            return False

        # Check if goal changed significantly
        if self.last_goal is not None:
            goal_distance = euclidean(goal_state.position[:2], self.last_goal[:2])
            if goal_distance > self.replanning_threshold:
                self.logger.debug("Goal changed, replanning needed")
                return True

        # Check if robot deviated from path
        if self.current_path:
            deviation = self._compute_path_deviation(current_state)
            if deviation > self.replanning_threshold:
                self.logger.debug(f"Path deviation {deviation:.3f}m, replanning needed")
                return True

        # Check for new obstacles (simplified)
        if obstacle_map is not None and self._path_blocked(obstacle_map):
            self.logger.debug("Path blocked by obstacles, replanning needed")
            return True

        return False

    def _rrt_star_planning(
        self, start: np.ndarray, goal: np.ndarray, obstacle_map: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        RRT* path planning algorithm.

        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacle_map: Obstacle map

        Returns:
            List of path points
        """
        # Simple implementation of RRT* for demonstration
        # In practice, would use more sophisticated implementation

        # Tree nodes: each node is [x, y, parent_index, cost]
        tree = [np.array([start[0], start[1], -1, 0.0])]

        for iteration in range(self.rrt_max_iterations):
            # Sample random point
            if np.random.random() < self.rrt_goal_bias:
                sample = goal.copy()
            else:
                sample = self._sample_random_point()

            # Find nearest node
            nearest_idx = self._find_nearest_node(tree, sample)
            nearest_node = tree[nearest_idx]

            # Steer towards sample
            new_point = self._steer(nearest_node[:2], sample, self.rrt_step_size)

            # Check if collision-free
            if obstacle_map is not None and self._is_collision(
                nearest_node[:2], new_point, obstacle_map
            ):
                continue

            # Find nearby nodes for rewiring
            nearby_indices = self._find_nearby_nodes(tree, new_point, self.rrt_neighbor_radius)

            # Choose parent with minimum cost
            min_cost = nearest_node[3] + euclidean(nearest_node[:2], new_point)
            best_parent = nearest_idx

            for idx in nearby_indices:
                node = tree[idx]
                cost = node[3] + euclidean(node[:2], new_point)
                if cost < min_cost and not self._is_collision(node[:2], new_point, obstacle_map):
                    min_cost = cost
                    best_parent = idx

            # Add new node
            new_node = np.array([new_point[0], new_point[1], best_parent, min_cost])
            tree.append(new_node)
            new_idx = len(tree) - 1

            # Rewire nearby nodes
            for idx in nearby_indices:
                node = tree[idx]
                cost_through_new = min_cost + euclidean(new_point, node[:2])
                if cost_through_new < node[3] and not self._is_collision(
                    new_point, node[:2], obstacle_map
                ):
                    tree[idx][2] = new_idx  # Update parent
                    tree[idx][3] = cost_through_new  # Update cost

            # Check if goal reached
            if euclidean(new_point, goal) < self.goal_tolerance:
                # Extract path
                path = self._extract_path(tree, new_idx)
                self.logger.debug(f"Path found in {iteration} iterations with {len(path)} points")
                return path

        self.logger.warning("RRT* failed to find path")
        return []

    def _sample_random_point(self) -> np.ndarray:
        """Sample random point in workspace."""
        # Simple rectangular workspace
        x = np.random.uniform(-5.0, 5.0)
        y = np.random.uniform(-5.0, 5.0)
        return np.array([x, y])  # type: ignore[no-any-return]

    def _find_nearest_node(self, tree: List[np.ndarray], point: np.ndarray) -> int:
        """Find nearest node in tree to given point."""
        min_distance = float("inf")
        nearest_idx = 0

        for i, node in enumerate(tree):
            distance = euclidean(node[:2], point)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i

        return nearest_idx

    def _steer(self, from_point: np.ndarray, to_point: np.ndarray, step_size: float) -> np.ndarray:
        """Steer from one point towards another with given step size."""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)

        if distance <= step_size:
            return to_point.copy()  # type: ignore[no-any-return]

        unit_direction = direction / distance
        return from_point + unit_direction * step_size  # type: ignore[no-any-return]

    def _find_nearby_nodes(
        self, tree: List[np.ndarray], point: np.ndarray, radius: float
    ) -> List[int]:
        """Find all nodes within radius of given point."""
        nearby = []
        for i, node in enumerate(tree):
            if euclidean(node[:2], point) <= radius:
                nearby.append(i)
        return nearby

    def _is_collision(
        self, start: np.ndarray, end: np.ndarray, obstacle_map: Optional[np.ndarray]
    ) -> bool:
        """Check if line segment collides with obstacles."""
        if obstacle_map is None:
            return False

        # Simple line collision check
        # In practice, would use more sophisticated collision detection
        num_checks = int(euclidean(start, end) / 0.05) + 1

        for i in range(num_checks):
            t = i / max(1, num_checks - 1)
            point = start + t * (end - start)

            # Convert to grid coordinates
            grid_x = int((point[0] + 5.0) * 10)  # Assuming 10m x 10m grid
            grid_y = int((point[1] + 5.0) * 10)

            if (
                0 <= grid_x < obstacle_map.shape[1]
                and 0 <= grid_y < obstacle_map.shape[0]
                and obstacle_map[grid_y, grid_x] > 0.5
            ):
                return True

        return False

    def _extract_path(self, tree: List[np.ndarray], goal_idx: int) -> List[np.ndarray]:
        """Extract path from tree starting from goal node."""
        path = []
        current_idx = goal_idx

        while current_idx != -1:
            node = tree[current_idx]
            path.append(node[:2].copy())
            current_idx = int(node[2])  # Parent index

        path.reverse()
        return path

    def _path_to_trajectory(
        self, path_points: List[np.ndarray], start_state: RobotState
    ) -> List[TrajectoryPoint]:
        """Convert path points to trajectory with timing."""
        if not path_points:
            return []

        trajectory = []
        cumulative_time = 0.0

        for i, point in enumerate(path_points):
            # Create robot state for this point
            state = RobotState()
            state.position = np.array([point[0], point[1], 0.0])

            if i < len(path_points) - 1:
                # Compute direction to next point
                next_point = path_points[i + 1]
                direction = next_point - point
                distance = np.linalg.norm(direction)

                if distance > 0:
                    # Compute travel time at max velocity
                    travel_time = distance / self.max_velocity
                    cumulative_time += travel_time

                    # Set orientation towards next point
                    yaw = np.arctan2(direction[1], direction[0])
                    state.orientation = np.array([0.0, 0.0, yaw])

                    # Set velocity
                    velocity_magnitude = min(self.max_velocity, distance / travel_time)
                    state.velocity = np.array(
                        [velocity_magnitude * np.cos(yaw), velocity_magnitude * np.sin(yaw), 0.0]
                    )

            trajectory_point = TrajectoryPoint(state=state, time=cumulative_time)
            trajectory.append(trajectory_point)

        self.current_path = path_points
        self.last_goal = path_points[-1] if path_points else None
        self.last_planning_time = time.time()

        return trajectory

    def _compute_path_deviation(self, current_state: RobotState) -> float:
        """Compute deviation from current path."""
        if not self.current_path:
            return 0.0

        current_pos = current_state.position[:2]
        min_distance = float("inf")

        # Find minimum distance to path
        for i in range(len(self.current_path) - 1):
            point1 = self.current_path[i]
            point2 = self.current_path[i + 1]

            # Distance to line segment
            distance = self._point_to_line_distance(current_pos, point1, point2)
            min_distance = min(min_distance, distance)

        return min_distance

    def _point_to_line_distance(
        self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """Compute distance from point to line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start

        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            return float(np.linalg.norm(point_vec))  # type: ignore[return-value]

        line_unit = line_vec / line_len
        proj_length = np.dot(point_vec, line_unit)

        if proj_length < 0:
            return float(np.linalg.norm(point_vec))  # type: ignore[return-value]
        elif proj_length > line_len:
            return float(np.linalg.norm(point - line_end))  # type: ignore[return-value]
        else:
            proj_point = line_start + proj_length * line_unit
            return float(np.linalg.norm(point - proj_point))  # type: ignore[return-value]

    def _path_blocked(self, obstacle_map: np.ndarray) -> bool:
        """Check if current path is blocked by obstacles."""
        if not self.current_path:
            return False

        # Check path segments for collisions
        for i in range(len(self.current_path) - 1):
            if self._is_collision(self.current_path[i], self.current_path[i + 1], obstacle_map):
                return True

        return False
