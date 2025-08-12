#!/usr/bin/env python3
"""
Test Runner Script for EnergyAI SDK
Runs all tests with proper configuration.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str = None):
    """Run a command and handle errors."""
    if description:
        print(f"🔧 {description}...")

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Run all tests for the SDK."""
    print("🧪 EnergyAI SDK Test Runner")
    print("=" * 40)

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print(f"📁 Working directory: {project_root}")

    # Check if tests directory exists
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        print("❌ Tests directory not found")
        sys.exit(1)

    # Run pytest with coverage if available
    test_commands = [
        "python -m pytest tests/ -v",
        "python -m pytest tests/ -v --cov=energyai_sdk --cov-report=term-missing",
    ]

    # Try running with coverage first, fallback to basic pytest
    success = False
    for cmd in test_commands:
        print(f"\n🔧 Running: {cmd}")
        if run_command(cmd, "Running tests"):
            success = True
            break
        print("⚠️ Command failed, trying next option...")

    if not success:
        print("❌ All test commands failed")
        sys.exit(1)

    print("\n✅ Tests completed successfully!")

    # Run type checking if mypy is available
    print("\n🔧 Checking types with mypy...")
    if run_command("python -m mypy energyai_sdk --ignore-missing-imports", "Type checking"):
        print("✅ Type checking passed")
    else:
        print("⚠️ Type checking not available or failed")


if __name__ == "__main__":
    main()
