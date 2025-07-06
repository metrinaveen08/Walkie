# Walkie - Agile Dynamic Robot 🤖

A high-performance robotics framework for building agile dynamic robots with real-time control, advanced perception, and intelligent path planning.

## 🚀 What is Walkie?

Walkie is a comprehensive robotics platform designed for creating agile, dynamic robots that can navigate complex environments with precision and safety. Whether you're building a research robot, autonomous vehicle, or industrial automation system, Walkie provides the foundation you need.

### Key Features

- **Real-time Control** - Async-based control loops for responsive robot behavior
- **Advanced Perception** - Computer vision and sensor fusion for environmental awareness  
- **Intelligent Planning** - RRT* path planning with dynamic obstacle avoidance
- **Safety First** - Built-in safety monitoring and emergency stop capabilities
- **Hardware Agnostic** - Works with simulation or real hardware through abstract interfaces
- **Developer Friendly** - Full type safety, comprehensive testing, and VS Code integration

## 🏗️ Architecture

Walkie follows a modular architecture with clear separation of concerns:

```
├── Control System     # High-level robot control and coordination
├── Motion Control     # Trajectory generation and following
├── Path Planning      # RRT* algorithm for navigation
├── Perception        # Computer vision and sensor fusion
├── Hardware Interface # Abstract hardware drivers
├── Safety Monitor    # Real-time safety constraint checking
└── Utilities         # Logging, configuration, and state management
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- OpenCV (for computer vision)
- NumPy, SciPy (for mathematical operations)
- ROS2 (optional, for robotics middleware)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/walkie.git
   cd walkie
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .[dev,vision,simulation]
   ```

4. **Run tests to verify installation**
   ```bash
   pytest tests/
   ```

### Your First Robot

```python
# Quick example - see scripts/basic_control_example.py for full code
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.control.robot_controller import RobotController

# Initialize logging
logger = setup_logger()

# Load configuration
config = ConfigLoader.load_config('config/hardware.yaml')

# Your robot is ready to go!
logger.info("Robot initialized successfully!")
```

## 🎮 Usage

### Running in Simulation

```bash
# Launch in simulation mode
python launch.py simulation --config config/simulation.yaml

# Or run the basic example
python scripts/basic_control_example.py
```

### Running with Real Hardware

```bash
# Configure your hardware in config/hardware.yaml
# Then launch with hardware
python launch.py hardware --config config/hardware.yaml
```

### Development Mode

```bash
# Run tests
python -m pytest tests/

# Type checking
python -m mypy src

# Code formatting
python -m black src/
```

## 📁 Project Structure

```
walkie/
├── src/                    # Main source code
│   ├── control/           # Robot control system
│   ├── hardware/          # Hardware interfaces
│   ├── motion_control/    # Trajectory control
│   ├── planning/          # Path planning algorithms
│   ├── sensors/           # Sensor management
│   ├── vision/            # Computer vision system
│   └── utils/             # Utilities and helpers
├── config/                # Configuration files
├── tests/                 # Test suites
├── scripts/               # Example scripts
├── docs/                  # Documentation
└── launch.py             # Main launcher
```

## 🔧 Configuration

Walkie uses YAML configuration files for easy customization:

```yaml
# config/hardware.yaml
hardware:
  wheel_base: 0.3          # Robot wheelbase in meters
  wheel_radius: 0.05       # Wheel radius in meters
  max_velocity: 1.0        # Maximum velocity in m/s

control:
  update_rate: 50          # Control loop frequency in Hz
  
safety:
  max_linear_velocity: 2.0  # Safety limits
  max_angular_velocity: 1.0
```

## 🛡️ Safety Features

Safety is paramount in robotics. Walkie includes:

- **Real-time Safety Monitoring** - Continuous checking of robot state
- **Velocity Limiting** - Automatic speed reduction when approaching limits
- **Emergency Stop** - Immediate robot shutdown capability
- **Collision Avoidance** - Path planning with obstacle detection
- **Workspace Boundaries** - Configurable operational limits

## 🧪 Testing

Walkie includes comprehensive testing:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## 📊 Performance

- **Control Loop Frequency**: Up to 1000Hz for real-time control
- **Path Planning**: RRT* with adaptive sampling
- **Sensor Fusion**: Kalman filtering for state estimation
- **Type Safety**: 100% type-checked with MyPy
- **Test Coverage**: 28% overall, 90%+ for critical components

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write tests for new features
- Update documentation as needed
- Use meaningful commit messages

## 📚 Documentation

- **API Reference**: See `docs/` directory
- **Examples**: Check `scripts/` for usage examples
- **Configuration**: Detailed config options in `config/`
- **Architecture**: System design in `docs/architecture.md`

## 🐛 Troubleshooting

### Common Issues

**ImportError with OpenCV**
```bash
pip install opencv-python
```

**Type checking errors**
```bash
# Install type stubs
pip install types-PyYAML
```

**Permission errors on hardware**
```bash
# Add user to dialout group (Linux)
sudo usermod -a -G dialout $USER
```

## 🔮 Roadmap

- [ ] **ROS2 Integration** - Native ROS2 node support
- [ ] **Web Interface** - Browser-based robot monitoring
- [ ] **Machine Learning** - Neural network path planning
- [ ] **Multi-Robot** - Swarm robotics capabilities
- [ ] **Advanced Sensors** - LiDAR, stereo cameras, IMU fusion
- [ ] **Cloud Integration** - Remote monitoring and control

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Thanks to the open-source robotics community
- Inspired by modern robotics research
- Built with Python's amazing ecosystem

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/walkie/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/walkie/discussions)
- **Email**: your.email@example.com

---

**Made with ❤️ for the robotics community**

*Walkie - Where robots come to life* 🤖✨
