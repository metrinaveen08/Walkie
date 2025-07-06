# Agile Dynamic Robot Documentation

Welcome to the Agile Dynamic Robot documentation!

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- CMake (for C++ components)
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agile-dynamic-robot
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package:
```bash
pip install -e ".[dev,vision,simulation]"
```

### Quick Start

#### Simulation Mode
```bash
python -m src.main --config config/simulation.yaml --simulation
```

#### Hardware Mode
```bash
python -m src.main --config config/hardware.yaml
```

#### Running Tests
```bash
pytest tests/ -v --cov=src
```

## Architecture Overview

The system is organized into several key modules:

- **Control**: High-level robot control and coordination
- **Hardware**: Hardware abstraction layer
- **Sensors**: Sensor data acquisition and fusion
- **Vision**: Computer vision and perception
- **Planning**: Path and motion planning
- **Motion Control**: Low-level trajectory following
- **Utils**: Utility functions and data structures
- **Simulation**: Physics simulation environment

## Configuration

The system uses YAML configuration files:

- `config/hardware.yaml`: Hardware-specific settings
- `config/simulation.yaml`: Simulation parameters

## Development

### Code Style
- Follow PEP 8 for Python code
- Use type hints for all functions
- Write docstrings in Google style
- Use Black for code formatting

### Testing
- Write unit tests for all modules
- Include integration tests
- Maintain high test coverage

### Performance
- Profile critical paths
- Use NumPy for mathematical operations
- Consider Cython for performance-critical code

## API Reference

See the individual module documentation for detailed API information.
