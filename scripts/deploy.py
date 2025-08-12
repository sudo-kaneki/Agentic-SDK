#!/usr/bin/env python3
"""
Deployment Script for EnergyAI SDK
Handles packaging and deployment tasks.
"""

import os
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path


def run_command(command: str, description: str = None):
    """Run a command and handle errors."""
    if description:
        print(f"🔧 {description}...")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )  # nosec B602
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def clean_build_artifacts():
    """Clean build artifacts and temporary files."""
    print("🧹 Cleaning build artifacts...")

    artifacts_to_clean = [
        "build",
        "dist",
        "*.egg-info",
        "__pycache__",
        ".pytest_cache",
        ".coverage",
        ".mypy_cache",
    ]

    for artifact in artifacts_to_clean:
        if "*" in artifact:
            # Use find command for wildcard patterns
            run_command(f"find . -name '{artifact}' -exec rm -rf {{}} +", f"Removing {artifact}")
        else:
            # Direct removal
            if os.path.exists(artifact):
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                    print(f"✅ Removed directory: {artifact}")
                else:
                    os.remove(artifact)
                    print(f"✅ Removed file: {artifact}")


def build_package():
    """Build the package for distribution."""
    print("📦 Building package...")

    # Install build dependencies
    if not run_command("pip install build twine", "Installing build dependencies"):
        return False

    # Build the package
    if not run_command("python -m build", "Building distribution packages"):
        return False

    print("✅ Package built successfully")
    return True


def run_quality_checks():
    """Run quality checks before deployment."""
    print("🔍 Running quality checks...")

    # Run tests
    if not run_command("python scripts/run_tests.py", "Running tests"):
        print("❌ Tests failed, deployment aborted")
        return False

    # Check package structure
    if not run_command("python -m build --sdist --no-isolation", "Checking package structure"):
        return False

    # Validate distribution
    if not run_command("python -m twine check dist/*", "Validating distribution"):
        return False

    print("✅ Quality checks passed")
    return True


def main():
    """Main deployment script."""
    print("🚀 EnergyAI SDK Deployment")
    print("=" * 40)

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print(f"📁 Working directory: {project_root}")

    # Parse command line arguments
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
    else:
        action = "build"

    if action == "clean":
        clean_build_artifacts()
        print("✅ Cleanup completed")

    elif action == "build":
        # Clean, check, and build
        clean_build_artifacts()

        if not run_quality_checks():
            print("❌ Quality checks failed")
            sys.exit(1)

        if not build_package():
            print("❌ Package build failed")
            sys.exit(1)

        print("\n✅ Deployment build completed successfully!")
        print("\nNext steps:")
        print("1. Test the built package: pip install dist/*.whl")
        print("2. Upload to test PyPI: python -m twine upload --repository testpypi dist/*")
        print("3. Upload to PyPI: python -m twine upload dist/*")

    elif action == "test-upload":
        # Upload to test PyPI
        if not run_command(
            "python -m twine upload --repository testpypi dist/*", "Uploading to test PyPI"
        ):
            print("❌ Test upload failed")
            sys.exit(1)
        print("✅ Test upload completed")

    elif action == "upload":
        # Upload to PyPI
        if not run_command("python -m twine upload dist/*", "Uploading to PyPI"):
            print("❌ Upload failed")
            sys.exit(1)
        print("✅ Upload to PyPI completed")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: clean, build, test-upload, upload")
        sys.exit(1)


if __name__ == "__main__":
    main()
