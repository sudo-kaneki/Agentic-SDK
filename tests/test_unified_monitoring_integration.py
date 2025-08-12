"""
Tests for Unified Monitoring Integration.

Tests the integration between EnergyAIApplication and the new unified MonitoringClient
for comprehensive observability (Langfuse + OpenTelemetry).
"""

from unittest.mock import MagicMock, patch

import pytest

from energyai_sdk.application import ChatRequest, EnergyAIApplication, create_application
from energyai_sdk.core import AgentResponse


class TestUnifiedMonitoringIntegration:
    """Test cases for unified monitoring integration (Langfuse + OpenTelemetry)."""

    @pytest.fixture
    def mock_monitoring_client(self):
        """Mock MonitoringClient for testing."""
        client = MagicMock()

        # Mock health check
        client.health_check.return_value = {
            "langfuse": True,
            "opentelemetry": True,
            "overall": True,
        }

        # Mock Langfuse sub-client
        client.langfuse_client = MagicMock()
        client.langfuse_client.is_enabled.return_value = True

        # Mock trace creation
        mock_trace = MagicMock()
        mock_trace.id = "test-trace-id"
        client.create_trace.return_value = mock_trace

        # Mock generation creation
        mock_generation = MagicMock()
        mock_generation.id = "test-generation-id"
        client.create_generation.return_value = mock_generation

        # Mock span creation
        mock_span = MagicMock()
        mock_span.id = "test-span-id"
        client.create_span.return_value = mock_span

        # Mock other methods
        client.end_generation = MagicMock()
        client.end_span = MagicMock()
        client.update_trace = MagicMock()
        client.flush = MagicMock()

        return client

    @pytest.fixture
    def app_with_monitoring(self, mock_monitoring_client):
        """Create application with unified monitoring enabled."""
        with patch("energyai_sdk.application.get_monitoring_client") as mock_get_client:
            mock_get_client.return_value = mock_monitoring_client
            app = EnergyAIApplication(
                title="Test App with Unified Monitoring",
                langfuse_monitoring_client=None,  # Now using unified client
            )
            return app

    @pytest.fixture
    def app_without_monitoring(self):
        """Create application without monitoring."""
        with patch("energyai_sdk.application.get_monitoring_client") as mock_get_client:
            mock_get_client.return_value = None
            app = EnergyAIApplication(
                title="Test App without Monitoring",
                langfuse_monitoring_client=None,
            )
            return app

    @pytest.mark.asyncio
    async def test_chat_request_with_unified_monitoring(
        self, app_with_monitoring, mock_monitoring_client
    ):
        """Test chat request processing with unified monitoring enabled."""
        # Arrange
        request = ChatRequest(
            message="Test message",
            agent_id="test_agent",
            session_id="test_session",
            user_id="test_user",
        )

        # Mock agent processing
        mock_response = AgentResponse(
            content="Test response",
            agent_id="test_agent",
            session_id="test_session",
            request_id=request.request_id,
            execution_time_ms=100,
        )

        with patch.object(
            app_with_monitoring, "_process_agent_request", return_value=mock_response
        ):
            # Act
            response = await app_with_monitoring._process_chat_request(request, None)

            # Assert
            assert response is not None
            assert response.content == "Test response"

            # Verify monitoring calls were made
            mock_monitoring_client.create_trace.assert_called_once()
            trace_call = mock_monitoring_client.create_trace.call_args
            assert "agent-run:test_agent" in str(trace_call)

            # Verify generation was created
            mock_monitoring_client.create_generation.assert_called_once()
            generation_call = mock_monitoring_client.create_generation.call_args
            assert generation_call is not None

            # Verify generation was ended
            mock_monitoring_client.end_generation.assert_called_once()
            end_gen_call = mock_monitoring_client.end_generation.call_args
            assert end_gen_call is not None

            # Verify trace was updated
            mock_monitoring_client.update_trace.assert_called()

            # Verify flush was called
            mock_monitoring_client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_request_without_monitoring(self, app_without_monitoring):
        """Test chat request processing without monitoring."""
        # Arrange
        request = ChatRequest(
            message="Test message",
            agent_id="test_agent",
            session_id="test_session",
            user_id="test_user",
        )

        mock_response = AgentResponse(
            content="Test response",
            agent_id="test_agent",
            session_id="test_session",
            request_id=request.request_id,
            execution_time_ms=100,
        )

        with patch.object(
            app_without_monitoring, "_process_agent_request", return_value=mock_response
        ):
            # Act
            response = await app_without_monitoring._process_chat_request(request, None)

            # Assert
            assert response is not None
            assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_context_loading_with_monitoring(
        self, app_with_monitoring, mock_monitoring_client
    ):
        """Test context loading with monitoring spans."""
        request = ChatRequest(
            message="Test with context",
            agent_id="test_agent",
            session_id="test_session_with_context",
            user_id="test_user",
        )

        mock_response = AgentResponse(
            content="Response with context",
            agent_id="test_agent",
            session_id="test_session_with_context",
            request_id=request.request_id,
            execution_time_ms=150,
        )

        # Mock context store to return existing session
        mock_session = {
            "id": "test_session_with_context",
            "thread": [
                {
                    "sender": "user",
                    "content": "Previous message",
                    "timestamp": "2023-01-01T00:00:00Z",
                },
                {
                    "sender": "agent",
                    "content": "Previous response",
                    "timestamp": "2023-01-01T00:01:00Z",
                },
            ],
        }

        with patch.object(
            app_with_monitoring, "_process_agent_request", return_value=mock_response
        ):
            with patch.object(
                app_with_monitoring.context_store_client,
                "load_or_create_new_session",
                return_value=mock_session,
            ):
                # Act
                response = await app_with_monitoring._process_chat_request(request, None)

                # Assert
                assert response is not None

                # Verify span creation for context loading
                mock_monitoring_client.create_span.assert_called()
                span_call = mock_monitoring_client.create_span.call_args
                assert "context-loading" in str(span_call)

                # Verify span was ended
                mock_monitoring_client.end_span.assert_called()

                # Verify generation was created with conversation history
                generation_call = mock_monitoring_client.create_generation.call_args
                assert generation_call is not None

    @pytest.mark.asyncio
    async def test_error_handling_with_monitoring(
        self, app_with_monitoring, mock_monitoring_client
    ):
        """Test error handling with monitoring."""
        request = ChatRequest(
            message="Test error",
            agent_id="test_agent",
            session_id="test_session",
            user_id="test_user",
        )

        # Mock agent to raise an error
        with patch.object(
            app_with_monitoring, "_process_agent_request", side_effect=Exception("Test error")
        ):
            # Act & Assert
            with pytest.raises(Exception, match="Test error"):
                await app_with_monitoring._process_chat_request(request, None)

            # Verify error was recorded in monitoring
            end_gen_calls = mock_monitoring_client.end_generation.call_args_list
            assert len(end_gen_calls) > 0

            # Verify trace was updated with error
            update_trace_calls = mock_monitoring_client.update_trace.call_args_list
            assert len(update_trace_calls) > 0

            # Verify flush was still called
            mock_monitoring_client.flush.assert_called()

    def test_langfuse_status_check(self, mock_monitoring_client):
        """Test Langfuse status check method."""
        with patch("energyai_sdk.application.get_monitoring_client") as mock_get_client:
            mock_get_client.return_value = mock_monitoring_client

            app = EnergyAIApplication(title="Test App")

            # Test with enabled Langfuse
            mock_monitoring_client.langfuse_client.is_enabled.return_value = True
            status = app._get_langfuse_status()
            assert status == "healthy"

            # Test with disabled Langfuse
            mock_monitoring_client.langfuse_client.is_enabled.return_value = False
            status = app._get_langfuse_status()
            assert status == "not_configured"

            # Test with no monitoring client
            mock_get_client.return_value = None
            status = app._get_langfuse_status()
            assert status == "not_configured"

    def test_application_factory_with_monitoring(self, mock_monitoring_client):
        """Test application factory function with monitoring enabled."""
        with patch("energyai_sdk.application.get_monitoring_client") as mock_get_client:
            with patch("energyai_sdk.clients.monitoring.initialize_monitoring") as mock_init:
                mock_get_client.return_value = None  # Initially not configured

                app = create_application(
                    enable_observability=True,
                    langfuse_public_key="test_key",
                    langfuse_secret_key="test_secret",
                    langfuse_host="https://test.langfuse.com",
                    langfuse_environment="test",
                )

                # Verify monitoring was initialized
                mock_init.assert_called_once()
                init_call = mock_init.call_args
                config = init_call[0][0]  # First positional argument

                assert config.enable_langfuse is True
                assert config.langfuse_public_key == "test_key"
                assert config.langfuse_secret_key == "test_secret"
                assert config.langfuse_host == "https://test.langfuse.com"
                assert config.environment == "test"

    def test_application_factory_without_monitoring(self):
        """Test application factory function with monitoring disabled."""
        app = create_application(enable_observability=False)

        # App should still be created successfully
        assert app is not None
        assert app.title == "EnergyAI Agentic SDK"

    @pytest.mark.asyncio
    async def test_health_check_with_monitoring(self, app_with_monitoring, mock_monitoring_client):
        """Test application health check with monitoring."""
        # Setup health check response
        mock_monitoring_client.health_check.return_value = {
            "langfuse": True,
            "opentelemetry": True,
            "overall": True,
        }

        health = await app_with_monitoring.health_check()

        assert health["status"] == "healthy"
        assert "observability" in health["components"]
        assert health["components"]["observability"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_without_monitoring(self, app_without_monitoring):
        """Test application health check without monitoring."""
        health = await app_without_monitoring.health_check()

        assert health["status"] == "healthy"
        assert "observability" in health["components"]
        assert health["components"]["observability"] == "not_configured"


class TestMonitoringClientIntegration:
    """Test direct monitoring client integration."""

    def test_monitoring_client_availability(self):
        """Test monitoring client availability check."""
        from energyai_sdk.clients.monitoring import get_monitoring_client

        # Should not raise an error
        client = get_monitoring_client()
        # May be None if not initialized, but should not error

    def test_monitoring_client_initialization(self):
        """Test monitoring client initialization."""
        from energyai_sdk.clients.monitoring import MonitoringConfig, initialize_monitoring

        config = MonitoringConfig(
            enable_langfuse=False,
            enable_opentelemetry=False,
        )

        # Should not raise an error
        client = initialize_monitoring(config)
        assert client is not None

    def test_mock_monitoring_client(self):
        """Test mock monitoring client functionality."""
        from energyai_sdk.clients.monitoring import MockMonitoringClient

        mock_client = MockMonitoringClient()

        # Test basic functionality
        trace = mock_client.create_trace("test_trace")
        assert trace is not None

        generation = mock_client.create_generation(trace, "test_generation")
        assert generation is not None

        mock_client.end_generation(generation, output="test output")
        mock_client.update_trace(trace, output="trace output")
        mock_client.flush()

        # Test health check
        health = mock_client.health_check()
        assert health["overall"] is True

    def test_monitoring_config_validation(self):
        """Test monitoring configuration validation."""
        from energyai_sdk.clients.monitoring import MonitoringConfig

        # Test with Langfuse enabled
        config = MonitoringConfig(
            enable_langfuse=True,
            langfuse_public_key="test_key",
            langfuse_secret_key="test_secret",
        )
        assert config.enable_langfuse is True

        # Test with OpenTelemetry enabled
        config = MonitoringConfig(
            enable_opentelemetry=True,
            otlp_trace_endpoint="http://localhost:4317",
        )
        assert config.enable_opentelemetry is True

    def test_monitoring_decorator(self):
        """Test monitoring decorator functionality."""
        from energyai_sdk.clients.monitoring import monitor

        @monitor("test_operation")
        def test_function():
            return "test_result"

        # Should not raise an error
        result = test_function()
        assert result == "test_result"
