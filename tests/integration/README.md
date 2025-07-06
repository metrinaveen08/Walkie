# Integration Tests

This directory contains integration tests for the agile dynamic robot system.

## Test Structure

- `test_robot_integration.py` - End-to-end robot control tests
- `test_simulation_integration.py` - Simulation environment tests
- `test_sensor_fusion_integration.py` - Multi-sensor integration tests
- `test_planning_integration.py` - Path planning with obstacles tests

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/integration/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/integration/test_robot_integration.py -v
```
