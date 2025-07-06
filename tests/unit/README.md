# Unit Tests

This directory contains unit tests for individual components of the agile dynamic robot system.

## Test Structure

- `test_basic.py` - Basic functionality tests
- `test_state.py` - Robot state and data structures
- `test_control.py` - Control system tests
- `test_sensors.py` - Sensor management tests
- `test_planning.py` - Path planning tests
- `test_vision.py` - Computer vision tests
- `test_hardware.py` - Hardware interface tests

## Running Tests

```bash
# Run all tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_basic.py -v
```
