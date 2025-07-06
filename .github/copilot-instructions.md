<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for Agile Dynamic Robot Project

## Project Overview
This is a comprehensive robotics project focused on creating an agile dynamic robot with advanced motion control, sensor integration, and real-time planning capabilities.

## Code Style and Conventions
- Follow PEP 8 for Python code
- Use type hints for all function parameters and return values
- Write docstrings in Google style format
- Prefer composition over inheritance
- Use async/await for I/O operations and real-time control loops

## Architecture Guidelines
- **Motion Control**: Implement kinematic and dynamic models using NumPy and SciPy
- **Sensor Fusion**: Use Kalman filters and particle filters for state estimation
- **Computer Vision**: Leverage OpenCV and MediaPipe for perception tasks
- **Path Planning**: Implement sampling-based and optimization-based planners
- **Real-time Control**: Ensure deterministic timing with proper thread management
- **Hardware Interface**: Use abstract base classes for device drivers

## Specific Patterns
- Use dataclasses for configuration and state representations
- Implement observer pattern for sensor data publishing
- Use factory pattern for creating different robot configurations
- Apply strategy pattern for interchangeable algorithms (planners, controllers)

## Performance Considerations
- Profile critical paths and optimize bottlenecks
- Use NumPy vectorization for mathematical operations
- Consider Cython or C++ extensions for performance-critical components
- Implement proper buffering for real-time data streams

## Testing Guidelines
- Write unit tests for all mathematical functions
- Use property-based testing for control algorithms
- Mock hardware interfaces in tests
- Include integration tests with simulation environment

## Safety and Reliability
- Always validate input parameters and sensor data
- Implement proper error handling and recovery mechanisms
- Add safety checks for motion limits and collision avoidance
- Log all critical events and state changes

## Documentation
- Document all mathematical formulations and algorithms
- Include usage examples in docstrings
- Maintain up-to-date configuration schemas
- Document hardware setup and calibration procedures
