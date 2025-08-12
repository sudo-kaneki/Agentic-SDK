#!/usr/bin/env python3
"""
Example usage of EnergyAI SDK configuration system.

This script demonstrates how to use different configuration methods.
"""

import os

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk.config import config_manager, get_settings


def example_env_variables():
    """Example 1: Using environment variables."""
    print("=" * 60)
    print("🌍 Example 1: Environment Variables Configuration")
    print("=" * 60)

    # Set some example environment variables
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-dev"
    os.environ["ENERGYAI_LOG_LEVEL"] = "DEBUG"
    os.environ["ENERGYAI_APP_PORT"] = "8001"

    # Get settings from environment
    settings = get_settings()

    print("✅ Configuration loaded from environment variables:")
    if hasattr(settings, "dict"):
        config_dict = settings.dict()
    else:
        config_dict = settings

    for key, value in config_dict.items():
        if value and "key" not in key.lower():  # Don't print sensitive keys
            print(f"  {key}: {value}")

    print("\n💡 To use this method:")
    print("1. Set environment variables (AZURE_OPENAI_*, ENERGYAI_*)")
    print("2. Call get_settings() or config_manager.get_settings()")


def example_yaml_config():
    """Example 2: Using YAML configuration files."""
    print("\n" + "=" * 60)
    print("📄 Example 2: YAML Configuration Files")
    print("=" * 60)

    config_dir = Path(__file__).parent

    # Try loading development config
    try:
        dev_config = config_manager.load_from_file(config_dir / "development.yaml")
        print("✅ Development configuration loaded:")
        print(f"  Application Title: {dev_config.get('application', {}).get('title')}")
        print(f"  Debug Mode: {dev_config.get('application', {}).get('debug')}")
        print(f"  Log Level: {dev_config.get('logging', {}).get('level')}")
        print(f"  Port: {dev_config.get('application', {}).get('port')}")

    except Exception as e:
        print(f"⚠️  Could not load development.yaml: {e}")

    print("\n💡 To use this method:")
    print("1. Create YAML config files (development.yaml, production.yaml)")
    print("2. Call config_manager.load_from_file('path/to/config.yaml')")


def example_platform_creation():
    """Example 3: Creating energy platform with configuration."""
    print("\n" + "=" * 60)
    print("🚀 Example 3: Energy Platform Creation")
    print("=" * 60)

    try:
        # This would create a platform using available configuration
        print("🔧 Creating energy platform...")
        print("   - Looking for configuration files...")
        print("   - Loading agent definitions...")
        print("   - Setting up telemetry...")

        # Note: We're not actually calling create_energy_platform() here
        # because it requires valid API keys
        print("✅ Platform creation process outlined")
        print("   (Actual creation requires valid API keys)")

        print("\n💡 To use this method:")
        print("1. Configure your API keys in .env or YAML files")
        print(
            "2. Call create_energy_platform() or create_energy_platform('config/production.yaml')"
        )

    except Exception as e:
        print(f"⚠️  Platform creation example: {e}")


def example_configuration_priorities():
    """Example 4: Configuration priority and overrides."""
    print("\n" + "=" * 60)
    print("⚡ Example 4: Configuration Priority")
    print("=" * 60)

    print("EnergyAI SDK configuration priority (highest to lowest):")
    print("1. 🔴 Environment Variables (highest priority)")
    print("   - AZURE_OPENAI_*, ENERGYAI_*, LANGFUSE_*")
    print("2. 🟡 YAML Configuration Files")
    print("   - development.yaml, production.yaml, testing.yaml")
    print("3. 🟢 Default Values (lowest priority)")
    print("   - Built-in defaults in the SDK")

    print("\n🎯 Best Practices:")
    print("• Use .env files for local development")
    print("• Use environment variables in production")
    print("• Use YAML files for complex multi-environment setups")
    print("• Never commit sensitive data (API keys, secrets)")


def show_available_configs():
    """Show what config files are available."""
    print("\n" + "=" * 60)
    print("📁 Available Configuration Files")
    print("=" * 60)

    config_dir = Path(__file__).parent

    for config_file in config_dir.glob("*.yaml"):
        print(f"✅ {config_file.name}")

    if (config_dir / ".env.example").exists():
        print("✅ .env.example (template for environment variables)")

    if (config_dir / "README.md").exists():
        print("✅ README.md (configuration documentation)")

    print(f"\n📍 Config directory: {config_dir}")


def main():
    """Run all configuration examples."""
    print("🚀 EnergyAI SDK Configuration Examples")

    show_available_configs()
    example_env_variables()
    example_yaml_config()
    example_platform_creation()
    example_configuration_priorities()

    print("\n" + "=" * 60)
    print("✨ Configuration Examples Complete!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("1. Copy config/.env.example to .env and fill in your API keys")
    print("2. Choose the configuration method that works best for your use case")
    print("3. Run: python -m energyai_sdk.config to test your configuration")
    print("4. See config/README.md for detailed documentation")


if __name__ == "__main__":
    main()
