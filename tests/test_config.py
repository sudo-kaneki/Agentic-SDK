# tests/test_config.py
"""
Test configuration management system and settings.
"""

import json
import os
from unittest.mock import patch

import pytest
import yaml

from energyai_sdk.config import (
    ApplicationConfig,
    ConfigurationManager,
    ModelConfig,
    SecurityConfig,
    TelemetryConfig,
    get_settings,
)

# Only test pydantic features if available
try:
    from energyai_sdk.config import EnergyAISettings

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating ModelConfig instance."""

        config = ModelConfig(
            deployment_name="gpt-4o",
            model_type="azure_openai",
            api_key="test-key-123",
            endpoint="https://test.openai.azure.com/",
            api_version="2024-02-01",
            temperature=0.7,
            max_tokens=1000,
            is_default=True,
        )

        assert config.deployment_name == "gpt-4o"
        assert config.model_type == "azure_openai"
        assert config.api_key == "test-key-123"
        assert config.endpoint == "https://test.openai.azure.com/"
        assert config.temperature == 0.7
        assert config.is_default is True

    def test_model_config_to_dict(self):
        """Test ModelConfig to_dict method."""

        config = ModelConfig(
            deployment_name="gpt-4o",
            model_type="azure_openai",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            is_default=True,
        )

        config_dict = config.to_dict()

        assert config_dict["deployment_name"] == "gpt-4o"
        assert config_dict["model_type"] == "azure_openai"
        assert config_dict["api_key"] == "test-key"
        assert config_dict["is_default"] is True
        assert config_dict["service_id"] == "gpt-4o"  # Should default to deployment_name

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""

        config = ModelConfig(deployment_name="test-model", model_type="openai", api_key="test-key")

        assert config.endpoint is None
        assert config.api_version == "2024-02-01"
        assert config.service_id is None
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.is_default is False


class TestTelemetryConfig:
    """Test TelemetryConfig dataclass."""

    def test_telemetry_config_creation(self):
        """Test creating TelemetryConfig instance."""

        config = TelemetryConfig(
            enable_azure_monitor=True,
            azure_monitor_connection_string="InstrumentationKey=test-key",
            enable_langfuse=True,
            langfuse_public_key="pk_test",
            langfuse_secret_key="sk_test",
            langfuse_host="https://test.langfuse.com",
            sample_rate=0.5,
        )

        assert config.enable_azure_monitor is True
        assert config.azure_monitor_connection_string == "InstrumentationKey=test-key"
        assert config.enable_langfuse is True
        assert config.langfuse_public_key == "pk_test"
        assert config.sample_rate == 0.5

    def test_telemetry_config_defaults(self):
        """Test TelemetryConfig default values."""

        config = TelemetryConfig()

        assert config.enable_azure_monitor is False
        assert config.azure_monitor_connection_string is None
        assert config.enable_langfuse is False
        assert config.langfuse_host == "https://cloud.langfuse.com"
        assert config.langfuse_environment == "production"
        assert config.sample_rate == 1.0


class TestSecurityConfig:
    """Test SecurityConfig dataclass."""

    def test_security_config_creation(self):
        """Test creating SecurityConfig instance."""

        config = SecurityConfig(
            api_keys=["key1", "key2", "key3"],
            bearer_tokens=["token1", "token2"],
            enable_auth=True,
            enable_rate_limiting=True,
            max_requests_per_minute=120,
            max_requests_per_hour=5000,
            enable_cors=False,
            allowed_origins=["https://example.com"],
        )

        assert config.api_keys == ["key1", "key2", "key3"]
        assert config.bearer_tokens == ["token1", "token2"]
        assert config.enable_auth is True
        assert config.max_requests_per_minute == 120
        assert config.enable_cors is False
        assert config.allowed_origins == ["https://example.com"]

    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""

        config = SecurityConfig()

        assert config.api_keys == []
        assert config.bearer_tokens == []
        assert config.enable_auth is True
        assert config.enable_rate_limiting is True
        assert config.max_requests_per_minute == 60
        assert config.max_requests_per_hour == 1000
        assert config.max_requests_per_day == 10000
        assert config.enable_cors is True
        assert config.allowed_origins == ["*"]


class TestApplicationConfig:
    """Test ApplicationConfig dataclass."""

    def test_application_config_creation(self):
        """Test creating ApplicationConfig instance."""

        config = ApplicationConfig(
            title="Custom Energy Platform",
            version="2.0.0",
            description="Custom description",
            host="0.0.0.0",
            port=8080,
            debug=True,
            max_message_length=5000,
            enable_caching=False,
            cache_ttl_seconds=600,
        )

        assert config.title == "Custom Energy Platform"
        assert config.version == "2.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.debug is True
        assert config.max_message_length == 5000
        assert config.enable_caching is False

    def test_application_config_defaults(self):
        """Test ApplicationConfig default values."""

        config = ApplicationConfig()

        assert config.title == "EnergyAI Agentic SDK"
        assert config.version == "1.0.0"
        assert config.description == "AI Agent Platform for Energy Analytics"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.enable_gzip is True
        assert config.max_message_length == 10000
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 300


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestEnergyAISettings:
    """Test EnergyAISettings pydantic model."""

    def test_settings_initialization_defaults(self):
        """Test settings with default values."""

        with patch.dict(os.environ, {}, clear=True):
            settings = EnergyAISettings()

            assert settings.app_title == "EnergyAI Agentic SDK"
            assert settings.app_version == "1.0.0"
            assert settings.app_host == "0.0.0.0"
            assert settings.app_port == 8000
            assert settings.app_debug is False
            assert settings.enable_auth is True
            assert settings.log_level == "INFO"

    def test_settings_from_environment_variables(self):
        """Test settings loaded from environment variables."""

        env_vars = {
            "ENERGYAI_APP_TITLE": "Test Platform",
            "ENERGYAI_APP_VERSION": "2.0.0",
            "ENERGYAI_APP_DEBUG": "true",
            "ENERGYAI_APP_PORT": "8080",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-api-key",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o-test",
            "LANGFUSE_PUBLIC_KEY": "pk_test_key",
            "LANGFUSE_SECRET_KEY": "sk_test_key",
            "ENERGYAI_API_KEYS": "key1,key2,key3",
            "ENERGYAI_MAX_REQUESTS_PER_MINUTE": "100",
            "ENERGYAI_ENABLE_CACHING": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()

            assert settings.app_title == "Test Platform"
            assert settings.app_version == "2.0.0"
            assert settings.app_debug is True
            assert settings.app_port == 8080
            assert settings.azure_openai_endpoint == "https://test.openai.azure.com/"
            assert settings.azure_openai_api_key == "test-api-key"
            assert settings.langfuse_public_key == "pk_test_key"
            assert settings.api_keys == ["key1", "key2", "key3"]
            assert settings.max_requests_per_minute == 100
            assert settings.enable_caching is False

    def test_settings_api_keys_parsing(self):
        """Test API keys parsing from comma-separated string."""

        env_vars = {"ENERGYAI_API_KEYS": "key1,key2,key3,key4"}

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()

            assert settings.api_keys == ["key1", "key2", "key3", "key4"]

    def test_settings_bearer_tokens_parsing(self):
        """Test bearer tokens parsing from comma-separated string."""

        env_vars = {"ENERGYAI_BEARER_TOKENS": "token1,token2,token3"}

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()

            assert settings.bearer_tokens == ["token1", "token2", "token3"]

    def test_get_model_configs_azure_openai(self):
        """Test getting model configs for Azure OpenAI."""

        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o-test",
            "AZURE_OPENAI_API_VERSION": "2024-03-01",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()
            model_configs = settings.get_model_configs()

            assert len(model_configs) == 1
            config = model_configs[0]

            assert config.deployment_name == "gpt-4o-test"
            assert config.model_type.value == "azure_openai"
            assert config.api_key == "test-key"
            assert config.endpoint == "https://test.openai.azure.com/"
            assert config.api_version == "2024-03-01"
            assert config.is_default is True

    def test_get_model_configs_openai(self):
        """Test getting model configs for OpenAI."""

        env_vars = {
            "OPENAI_API_KEY": "openai-test-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()
            model_configs = settings.get_model_configs()

            assert len(model_configs) == 1
            config = model_configs[0]

            assert config.deployment_name == "gpt-4o"
            assert config.model_type.value == "openai"
            assert config.api_key == "openai-test-key"
            assert config.endpoint == "https://api.openai.com/v1"
            assert config.is_default is True

    def test_get_model_configs_both_providers(self):
        """Test getting model configs when both providers are configured."""

        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o-azure",
            "OPENAI_API_KEY": "openai-key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()
            model_configs = settings.get_model_configs()

            assert len(model_configs) == 2

            # Azure should be default (added first)
            azure_config = model_configs[0]
            assert azure_config.model_type.value == "azure_openai"
            assert azure_config.is_default is True

            openai_config = model_configs[1]
            assert openai_config.model_type.value == "openai"
            assert openai_config.is_default is False

    def test_get_telemetry_config(self):
        """Test getting telemetry configuration."""

        env_vars = {
            "AZURE_MONITOR_CONNECTION_STRING": "InstrumentationKey=test-key",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
            "LANGFUSE_HOST": "https://custom.langfuse.com",
            "LANGFUSE_ENVIRONMENT": "staging",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()
            telemetry_config = settings.get_telemetry_config()

            assert telemetry_config.enable_azure_monitor is True
            assert telemetry_config.azure_monitor_connection_string == "InstrumentationKey=test-key"
            assert telemetry_config.enable_langfuse is True
            assert telemetry_config.langfuse_public_key == "pk_test"
            assert telemetry_config.langfuse_secret_key == "sk_test"
            assert telemetry_config.langfuse_host == "https://custom.langfuse.com"
            assert telemetry_config.langfuse_environment == "staging"

    def test_get_security_config(self):
        """Test getting security configuration."""

        env_vars = {
            "ENERGYAI_API_KEYS": "key1,key2",
            "ENERGYAI_BEARER_TOKENS": "token1,token2",
            "ENERGYAI_ENABLE_AUTH": "true",
            "ENERGYAI_ENABLE_RATE_LIMITING": "true",
            "ENERGYAI_MAX_REQUESTS_PER_MINUTE": "120",
            "ENERGYAI_MAX_REQUESTS_PER_HOUR": "6000",
            "ENERGYAI_ENABLE_CORS": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()
            security_config = settings.get_security_config()

            assert security_config.api_keys == ["key1", "key2"]
            assert security_config.bearer_tokens == ["token1", "token2"]
            assert security_config.enable_auth is True
            assert security_config.enable_rate_limiting is True
            assert security_config.max_requests_per_minute == 120
            assert security_config.max_requests_per_hour == 6000
            assert security_config.enable_cors is False

    def test_get_application_config(self):
        """Test getting application configuration."""

        env_vars = {
            "ENERGYAI_APP_TITLE": "Custom Platform",
            "ENERGYAI_APP_VERSION": "3.0.0",
            "ENERGYAI_APP_HOST": "127.0.0.1",
            "ENERGYAI_APP_PORT": "9000",
            "ENERGYAI_APP_DEBUG": "true",
            "ENERGYAI_MAX_MESSAGE_LENGTH": "5000",
            "ENERGYAI_ENABLE_CACHING": "false",
            "ENERGYAI_CACHE_TTL_SECONDS": "600",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()
            app_config = settings.get_application_config()

            assert app_config.title == "Custom Platform"
            assert app_config.version == "3.0.0"
            assert app_config.host == "127.0.0.1"
            assert app_config.port == 9000
            assert app_config.debug is True
            assert app_config.max_message_length == 5000
            assert app_config.enable_caching is False
            assert app_config.cache_ttl_seconds == 600


class TestConfigurationManager:
    """Test ConfigurationManager functionality."""

    def test_configuration_manager_initialization(self):
        """Test ConfigurationManager initialization."""

        manager = ConfigurationManager()

        assert manager.config_path is None
        assert isinstance(manager._config_cache, dict)

    def test_configuration_manager_with_path(self, tmp_path):
        """Test ConfigurationManager with config path."""

        config_file = tmp_path / "test_config.yaml"
        manager = ConfigurationManager(config_file)

        assert manager.config_path == config_file

    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""

        config_data = {
            "application": {"title": "Test App", "port": 8080},
            "models": [{"deployment_name": "gpt-4o", "api_key": "test-key"}],
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        manager = ConfigurationManager()
        loaded_config = manager.load_from_file(config_file)

        assert loaded_config == config_data
        assert loaded_config["application"]["title"] == "Test App"
        assert loaded_config["application"]["port"] == 8080

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""

        config_data = {
            "application": {"title": "Test YAML App", "debug": True},
            "security": {"enable_auth": False, "api_keys": ["key1", "key2"]},
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigurationManager()

        try:
            loaded_config = manager.load_from_file(config_file)

            assert loaded_config == config_data
            assert loaded_config["application"]["title"] == "Test YAML App"
            assert loaded_config["security"]["enable_auth"] is False
        except ImportError:
            pytest.skip("PyYAML not available")

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file raises error."""

        manager = ConfigurationManager()

        with pytest.raises(FileNotFoundError):
            manager.load_from_file("/nonexistent/config.json")

    def test_load_from_unsupported_format(self, tmp_path):
        """Test loading from unsupported file format."""

        config_file = tmp_path / "config.txt"
        config_file.write_text("This is not a valid config format")

        manager = ConfigurationManager()

        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            manager.load_from_file(config_file)

    def test_load_from_env(self):
        """Test loading configuration from environment variables."""

        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "env-api-key",
            "ENERGYAI_APP_DEBUG": "true",
            "ENERGYAI_APP_PORT": "8080",
            "ENERGYAI_LOG_LEVEL": "DEBUG",
        }

        manager = ConfigurationManager()

        with patch.dict(os.environ, env_vars, clear=True):
            config = manager.load_from_env()

            if PYDANTIC_AVAILABLE:
                # Should return settings object
                assert hasattr(config, "azure_openai_endpoint")
                assert config.azure_openai_endpoint == "https://env.openai.azure.com/"
            else:
                # Should return dictionary
                assert isinstance(config, dict)
                assert config["azure_openai_endpoint"] == "https://env.openai.azure.com/"

    def test_get_settings(self):
        """Test get_settings method."""

        manager = ConfigurationManager()
        settings = manager.get_settings()

        if PYDANTIC_AVAILABLE:
            from energyai_sdk.config import EnergyAISettings

            assert isinstance(settings, EnergyAISettings)
        else:
            assert isinstance(settings, dict)


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_complete_configuration_flow(self, tmp_path):
        """Test complete configuration loading and usage flow."""

        # Create comprehensive config file
        config_data = {
            "application": {
                "title": "Integration Test Platform",
                "version": "1.0.0",
                "debug": True,
                "port": 8080,
            },
            "models": [
                {
                    "deployment_name": "gpt-4o-integration",
                    "model_type": "azure_openai",
                    "endpoint": "https://integration.openai.azure.com/",
                    "api_key": "integration-key",
                    "is_default": True,
                }
            ],
            "telemetry": {
                "enable_azure_monitor": True,
                "azure_monitor_connection_string": "InstrumentationKey=integration-key",
                "enable_langfuse": True,
                "langfuse_public_key": "pk_integration",
                "langfuse_secret_key": "sk_integration",
            },
            "security": {
                "enable_auth": True,
                "api_keys": ["integration-key-1", "integration-key-2"],
                "max_requests_per_minute": 200,
            },
        }

        config_file = tmp_path / "integration_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Test loading with ConfigurationManager
        manager = ConfigurationManager(config_file)
        loaded_config = manager.load_from_file(config_file)

        # Verify all sections loaded correctly
        assert loaded_config["application"]["title"] == "Integration Test Platform"
        assert loaded_config["application"]["port"] == 8080
        assert loaded_config["models"][0]["deployment_name"] == "gpt-4o-integration"
        assert loaded_config["telemetry"]["enable_azure_monitor"] is True
        assert loaded_config["security"]["max_requests_per_minute"] == 200

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_environment_override_priority(self):
        """Test that environment variables override default values."""

        # Set environment variables that should override defaults
        env_vars = {
            "ENERGYAI_APP_TITLE": "Environment Override App",
            "ENERGYAI_APP_PORT": "9999",
            "ENERGYAI_MAX_REQUESTS_PER_MINUTE": "500",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = EnergyAISettings()

            # Environment variables should override defaults
            assert settings.app_title == "Environment Override App"
            assert settings.app_port == 9999
            assert settings.max_requests_per_minute == 500

            # Non-overridden values should use defaults
            assert settings.app_version == "1.0.0"  # Default
            assert settings.enable_caching is True  # Default

    def test_get_settings_global_function(self):
        """Test global get_settings function."""

        settings = get_settings()

        if PYDANTIC_AVAILABLE:
            from energyai_sdk.config import EnergyAISettings

            assert isinstance(settings, EnergyAISettings)
        else:
            assert isinstance(settings, dict)

        # Should be able to access basic settings
        if PYDANTIC_AVAILABLE:
            assert hasattr(settings, "app_title")
            assert hasattr(settings, "app_port")
        else:
            assert "app_title" in settings or settings.get("app_title") is not None


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_invalid_json_file(self, tmp_path):
        """Test handling of invalid JSON file."""

        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json content")

        manager = ConfigurationManager()

        with pytest.raises(Exception):  # JSON decode error
            manager.load_from_file(config_file)

    def test_invalid_yaml_file(self, tmp_path):
        """Test handling of invalid YAML file."""

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: {")

        manager = ConfigurationManager()

        try:
            with pytest.raises(Exception):  # YAML parse error
                manager.load_from_file(config_file)
        except ImportError:
            pytest.skip("PyYAML not available")

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_invalid_environment_values(self):
        """Test handling of invalid environment variable values."""

        env_vars = {"ENERGYAI_APP_PORT": "not_a_number", "ENERGYAI_APP_DEBUG": "not_a_boolean"}

        with patch.dict(os.environ, env_vars, clear=True):
            # Should raise validation error for invalid port
            with pytest.raises(Exception):  # Pydantic validation error
                EnergyAISettings()

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_missing_required_config(self):
        """Test behavior when required configuration is missing."""

        # Clear all environment variables
        with patch.dict(os.environ, {}, clear=True):
            settings = EnergyAISettings()

            # Should use defaults when optional config is missing
            assert settings.azure_openai_endpoint is None
            assert settings.azure_openai_api_key is None

            # Model configs should be empty
            model_configs = settings.get_model_configs()
            assert len(model_configs) == 0


@pytest.mark.integration
class TestConfigurationIntegrationWithApplication:
    """Integration tests between configuration and application."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_configuration_applied_to_application(self):
        """Test that configuration is properly applied to application."""

        env_vars = {
            "ENERGYAI_APP_TITLE": "Config Integration App",
            "ENERGYAI_APP_VERSION": "2.5.0",
            "ENERGYAI_APP_DEBUG": "true",
            "ENERGYAI_ENABLE_AUTH": "false",
            "ENERGYAI_MAX_REQUESTS_PER_MINUTE": "150",
        }

        with patch.dict(os.environ, env_vars, clear=True):

            # This would create an application with the configured settings
            # We can't test the full flow without all dependencies, but we can
            # test that the configuration is read correctly
            settings = EnergyAISettings()
            app_config = settings.get_application_config()
            security_config = settings.get_security_config()

            assert app_config.title == "Config Integration App"
            assert app_config.version == "2.5.0"
            assert app_config.debug is True
            assert security_config.enable_auth is False
            assert security_config.max_requests_per_minute == 150


if __name__ == "__main__":
    pytest.main([__file__])
