"""
Computer vision system for perception and object detection.
"""

import asyncio
from typing import Dict, Any, Optional, List
import time

import numpy as np
import cv2

from ..utils.state import VisionData
from ..utils.logger import RobotLogger


class VisionSystem:
    """Computer vision system for robot perception."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vision system.

        Args:
            config: Vision system configuration
        """
        self.logger = RobotLogger(__name__)
        self.config = config

        # Camera configuration
        self.camera_id = config.get("camera_id", 0)
        self.frame_width = config.get("frame_width", 640)
        self.frame_height = config.get("frame_height", 480)
        self.fps = config.get("fps", 30)

        # Vision processing parameters
        self.enable_object_detection = config.get("enable_object_detection", True)
        self.enable_visual_odometry = config.get("enable_visual_odometry", True)
        self.enable_obstacle_detection = config.get("enable_obstacle_detection", True)

        # Internal state
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.last_pose_estimate: Optional[VisionData] = None
        self.obstacle_map: Optional[np.ndarray] = None

        # Vision processing tasks
        self._capture_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None

        # Object detection setup
        self._setup_object_detection()

    def _setup_object_detection(self) -> None:
        """Setup object detection models."""
        # In a real implementation, this would load actual ML models
        # For now, we'll use simple color-based detection
        self.object_classes = ["person", "car", "bicycle", "obstacle"]
        self.logger.info("Object detection setup completed")

    async def start(self) -> None:
        """Start the vision system."""
        self.logger.info("Starting vision system")

        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            # If camera not available, use simulation mode
            self.logger.warning("Camera not available, using simulation mode")
            self.camera = None
        else:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_running = True

        # Start processing tasks
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._processing_task = asyncio.create_task(self._processing_loop())

        self.logger.info("Vision system started successfully")

    async def stop(self) -> None:
        """Stop the vision system."""
        self.logger.info("Stopping vision system")

        self.is_running = False

        # Cancel tasks
        tasks_to_cancel = []
        if self._capture_task and not self._capture_task.done():
            self._capture_task.cancel()
            tasks_to_cancel.append(self._capture_task)
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            tasks_to_cancel.append(self._processing_task)

        # Wait for tasks to complete
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Release camera
        if self.camera:
            self.camera.release()

        self.logger.info("Vision system stopped")

    async def _capture_loop(self) -> None:
        """Camera capture loop."""
        while self.is_running:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret:
                        self.current_frame = frame
                else:
                    # Simulation mode - generate synthetic frame
                    self.current_frame = self._generate_synthetic_frame()

                # Control frame rate
                await asyncio.sleep(1.0 / self.fps)

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                await asyncio.sleep(0.1)

    async def _processing_loop(self) -> None:
        """Vision processing loop."""
        while self.is_running:
            try:
                if self.current_frame is not None:
                    # Process current frame
                    await self._process_frame(self.current_frame)

                # Process at lower frequency than capture
                await asyncio.sleep(0.1)  # 10 Hz

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_frame(self, frame: np.ndarray) -> None:
        """
        Process a camera frame.

        Args:
            frame: Camera frame to process
        """
        # Object detection
        if self.enable_object_detection:
            detected_objects = self._detect_objects(frame)
        else:
            detected_objects = []

        # Visual odometry
        if self.enable_visual_odometry:
            pose_estimate = self._estimate_pose(frame)
            self.last_pose_estimate = pose_estimate

        # Obstacle detection
        if self.enable_obstacle_detection:
            self.obstacle_map = self._detect_obstacles(frame)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic camera frame for simulation."""
        # Create a simple synthetic frame
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Add some synthetic objects
        cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)  # Green rectangle
        cv2.circle(frame, (400, 300), 50, (0, 0, 255), -1)  # Red circle

        # Add some noise
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        return frame  # type: ignore[no-any-return]

    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the frame.

        Args:
            frame: Camera frame

        Returns:
            List of detected objects
        """
        objects = []

        # Simple color-based detection for simulation
        # In practice, this would use sophisticated ML models

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect red objects (potential obstacles)
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                # Estimate distance (simplified)
                distance = max(1.0, 1000.0 / max(w, h))  # Rough distance estimate

                objects.append(
                    {
                        "class": "obstacle",
                        "confidence": 0.8,
                        "bbox": [x, y, w, h],
                        "center": [center_x, center_y],
                        "distance": distance,
                    }
                )

        return objects

    def _estimate_pose(self, frame: np.ndarray) -> Optional[VisionData]:
        """
        Estimate robot pose from visual odometry.

        Args:
            frame: Camera frame

        Returns:
            Pose estimate or None
        """
        # Simplified visual odometry simulation
        # In practice, this would use sophisticated SLAM algorithms

        current_time = time.time()

        # Simulate some drift and noise
        position = np.array(
            [np.random.normal(0, 0.1), np.random.normal(0, 0.1), 0.0]  # x  # y  # z
        )

        orientation = np.array([0.0, 0.0, np.random.normal(0, 0.05)])  # roll  # pitch  # yaw

        return VisionData(
            timestamp=current_time,
            position=position,
            orientation=orientation,
            confidence=0.7,
            objects=[],
        )

    def _detect_obstacles(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect obstacles and create obstacle map.

        Args:
            frame: Camera frame

        Returns:
            Obstacle map (occupancy grid)
        """
        # Create a simple obstacle map
        # In practice, this would use depth estimation and mapping

        # Create a 100x100 grid representing 10m x 10m area
        obstacle_map = np.zeros((100, 100), dtype=np.float32)

        # Add some random obstacles for simulation
        for _ in range(5):
            x = np.random.randint(10, 90)
            y = np.random.randint(10, 90)
            size = np.random.randint(3, 8)

            # Mark obstacle region
            obstacle_map[y - size : y + size, x - size : x + size] = 1.0

        return obstacle_map  # type: ignore[no-any-return]

    async def get_pose_estimate(self) -> Optional[VisionData]:
        """Get latest pose estimate from visual odometry."""
        return self.last_pose_estimate

    async def get_obstacle_map(self) -> Optional[np.ndarray]:
        """Get current obstacle map."""
        return self.obstacle_map

    async def get_detected_objects(self) -> List[Dict[str, Any]]:
        """Get list of detected objects."""
        if self.current_frame is not None:
            return self._detect_objects(self.current_frame)
        return []

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        return {
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.fps,
            "is_active": self.camera is not None and self.camera.isOpened(),
        }

    async def capture_image(self, filename: str) -> bool:
        """
        Capture and save current frame.

        Args:
            filename: Output filename

        Returns:
            True if successful
        """
        if self.current_frame is not None:
            return cv2.imwrite(filename, self.current_frame)
        return False
