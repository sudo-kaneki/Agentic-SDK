#!/usr/bin/env python3
"""
Development Setup Script for EnergyAI SDK
Installs dependencies and sets up development environment.
"""

import os
import subprocess  # nosec B404
import sys
from pathlib import Path


def run_command(command: str, description: str = None):
    """Run a command and handle errors."""
    if description:
        print(f"=> {description}...")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )  # nosec B602
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Set up development environment."""
    print("=> EnergyAI SDK Development Setup")
    print("=" * 40)

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print(f"=> Working directory: {project_root}")

    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("L Failed to install requirements")
        sys.exit(1)

    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("L Failed to install package")
        sys.exit(1)

    # Test imports
    print(">� Testing imports...")
    try:
        from energyai_sdk import agent, bootstrap_agents, tool

        print(" Core imports work")
    except ImportError as e:
        print(f"L Import error: {e}")
        sys.exit(1)

    # Create test agent
    print("> Testing decorator functionality...")
    try:

        @tool(name="test_tool")
        def test_tool(x: float) -> dict:
            return {"result": x * 2}

        @agent(name="TestAgent", tools=["test_tool"])
        class TestAgent:
            system_prompt = "I am a test agent"

        print(" Decorators work correctly")
    except Exception as e:
        print(f"L Decorator error: {e}")
        sys.exit(1)

    print("\n<� Development setup complete!")
    print("\nNext steps:")
    print("1. Install Semantic Kernel: pip install semantic-kernel")
    print("2. Run tests: python scripts/run_tests.py")
    print("3. Try examples: python examples/decorator_concept_demo.py")


if __name__ == "__main__":
    main()
