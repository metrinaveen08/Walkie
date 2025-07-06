#!/usr/bin/env python3
"""
Launch script for the agile dynamic robot system.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Agile Dynamic Robot Launcher")
    parser.add_argument(
        "mode",
        choices=["hardware", "simulation", "test", "example"],
        help="Launch mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Custom configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Get project root and python executable
    project_root = Path(__file__).parent
    python_exe = project_root / ".venv" / "bin" / "python"
    
    if not python_exe.exists():
        print("Virtual environment not found. Please run 'pip install -e .' first.")
        sys.exit(1)
    
    # Build command based on mode
    if args.mode == "hardware":
        config = args.config or "config/hardware.yaml"
        cmd = [str(python_exe), "-m", "src.main", "--config", config]
        if args.debug:
            cmd.extend(["--log-level", "DEBUG"])
        
    elif args.mode == "simulation":
        config = args.config or "config/simulation.yaml"
        cmd = [str(python_exe), "-m", "src.main", "--config", config, "--simulation"]
        if args.debug:
            cmd.extend(["--log-level", "DEBUG"])
        
    elif args.mode == "test":
        cmd = [str(python_exe), "-m", "pytest", "tests/", "-v"]
        if args.debug:
            cmd.extend(["--cov=src", "--cov-report=html"])
        
    elif args.mode == "example":
        cmd = [str(python_exe), "scripts/basic_control_example.py"]
    
    # Run the command
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
