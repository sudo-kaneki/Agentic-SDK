"""
Tests for the enhanced application with Azure integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from energyai_sdk.application import (
    ChatRequest,
    ChatResponse,
    EnergyAIApplication,
    create_application,
    create_production_application,
)
from energyai_sdk.clients import (
    MockContextStoreClient,
    MockMonitoringClient,
    MockRegistryClient,
)
from energyai_sdk.core import AgentRequest, AgentResponse, CoreAgent


class MockAgent(CoreAgent):
    """Mock agent for testing."""

    def __init__(self):
        super().__init__(
            agent_name="test_agent", agent_description="Test agent", system_prompt="Test prompt"
        )

    async def process_request(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            content=f"Response to: {request.message}",
            agent_id=self.agent_name,
            session_id=request.session_id,
        )

    def get_capabilities(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "description": self.agent_description,
            "type": "mock_agent",
            "capabilities": ["testing"],
            "models": ["mock-model"],
            "tools": ["mock_tool"],
            "skills": ["mock_skill"],
        }


class TestEnergyAIApplicationAzure:
    """Test EnergyAI application with Azure integration."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock Azure service clients."""
        registry_client = MockRegistryClient()
        context_store_client = MockContextStoreClient()
        monitoring_client = MockMonitoringClient()

        return {
            "registry": registry_client,
            "context_store": context_store_client,
            "monitoring": monitoring_client,
        }

    @pytest.fixture
    def app_with_azure(self, mock_clients):
        """Create application with Azure clients."""
        return EnergyAIApplication(
            title="Test App with Azure",
            registry_client=mock_clients["registry"],
            context_store_client=mock_clients["context_store"],
            monitoring_client=mock_clients["monitoring"],
        )

    def test_application_initialization_with_clients(self, app_with_azure, mock_clients):
        """Test application initializes with Azure clients."""
        assert app_with_azure.registry_client == mock_clients["registry"]
        assert app_with_azure.context_store_client == mock_clients["context_store"]
        assert app_with_azure.monitoring_client == mock_clients["monitoring"]

    @pytest.mark.asyncio
    async def test_startup_checks_external_services(self, app_with_azure):
        """Test startup performs health checks on external services."""
        await app_with_azure._startup()

        # Verify health checks were performed
        assert "external_registry" in app_with_azure.components_status
        assert "context_store" in app_with_azure.components_status
        assert "monitoring" in app_with_azure.components_status

    @pytest.mark.asyncio
    async def test_shutdown_closes_clients(self, app_with_azure, mock_clients):
        """Test shutdown closes Azure clients."""
        # Mock the close methods
        mock_clients["registry"].close = AsyncMock()
        mock_clients["context_store"].close = AsyncMock()
        mock_clients["monitoring"].shutdown = MagicMock()

        await app_with_azure._shutdown()

        # Verify cleanup was called
        mock_clients["registry"].close.assert_called_once()
        mock_clients["context_store"].close.assert_called_once()
        mock_clients["monitoring"].shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_session_persistence(self, app_with_azure):
        """Test chat request with session persistence."""
        # Add a mock agent
        mock_agent = MockAgent()
        app_with_azure.add_agent(mock_agent)

        # Create chat request with session
        chat_request = ChatRequest(
            message="Test message",
            session_id="test_session_123",
            subject_id="test_user",
            agent_id="test_agent",
        )

        # Process chat request
        response = await app_with_azure._process_chat_request(chat_request, None)

        # Verify response
        assert isinstance(response, ChatResponse)
        assert response.content == "Response to: Test message"
        assert response.session_id == "test_session_123"
        assert response.agent_id == "test_agent"

        # Verify session was created in context store
        session_doc = await app_with_azure.context_store_client.get_session("test_session_123")
        assert session_doc is not None
        assert session_doc.subject_id == "test_user"

    @pytest.mark.asyncio
    async def test_chat_without_session(self, app_with_azure):
        """Test chat request without session persistence."""
        mock_agent = MockAgent()
        app_with_azure.add_agent(mock_agent)

        chat_request = ChatRequest(message="Test without session", agent_id="test_agent")

        response = await app_with_azure._process_chat_request(chat_request, None)

        assert isinstance(response, ChatResponse)
        assert response.content == "Response to: Test without session"
        assert response.session_id is None

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, app_with_azure):
        """Test monitoring integration during chat processing."""
        mock_agent = MockAgent()
        app_with_azure.add_agent(mock_agent)

        chat_request = ChatRequest(message="Monitor this request", agent_id="test_agent")

        # Process request
        await app_with_azure._process_chat_request(chat_request, None)

        # Verify metrics were recorded
        metrics = app_with_azure.monitoring_client.get_recorded_metrics()
        assert len(metrics) >= 2  # Should have duration and count metrics

        # Verify traces were recorded
        traces = app_with_azure.monitoring_client.get_recorded_traces()
        assert len(traces) >= 1  # Should have at least one span


@pytest.mark.asyncio
class TestApplicationFactory:
    """Test application factory functions with Azure integration."""

    def test_create_application_with_clients(self):
        """Test creating application with custom Azure clients."""
        registry_client = MockRegistryClient()
        context_store_client = MockContextStoreClient()
        monitoring_client = MockMonitoringClient()

        app = create_application(
            title="Custom App",
            registry_client=registry_client,
            context_store_client=context_store_client,
            monitoring_client=monitoring_client,
        )

        assert app.title == "Custom App"
        assert app.registry_client == registry_client
        assert app.context_store_client == context_store_client
        assert app.monitoring_client == monitoring_client

    @patch("energyai_sdk.application.RegistryClient")
    @patch("energyai_sdk.application.ContextStoreClient")
    @patch("energyai_sdk.application.MonitoringClient")
    def test_create_production_application(
        self, mock_monitoring_cls, mock_context_cls, mock_registry_cls
    ):
        """Test creating production application with real Azure clients."""
        # Mock the class constructors
        mock_registry_cls.return_value = MagicMock()
        mock_context_cls.return_value = MagicMock()
        mock_monitoring_instance = MagicMock()
        mock_monitoring_cls.return_value = mock_monitoring_instance

        app = create_production_application(
            api_keys=["test_key"],
            cosmos_endpoint="https://test.documents.azure.com:443/",
            cosmos_key="test_key",
            azure_monitor_connection_string="InstrumentationKey=test",
        )

        # Verify clients were created
        mock_registry_cls.assert_called_once()
        mock_context_cls.assert_called_once()
        mock_monitoring_cls.assert_called_once()
        mock_monitoring_instance.initialize.assert_called_once()

        assert app.title == "EnergyAI Production Platform"
        assert app.enable_cors is False  # Production should have restrictive CORS


@pytest.mark.asyncio
class TestSessionManagementEndpoints:
    """Test the new session management API endpoints."""

    @pytest.fixture
    def app_with_fastapi(self):
        """Create application with FastAPI enabled."""
        # This would require FastAPI to be available
        try:
            app = create_application(
                registry_client=MockRegistryClient(),
                context_store_client=MockContextStoreClient(),
                monitoring_client=MockMonitoringClient(),
            )
            return app
        except Exception:
            pytest.skip("FastAPI not available for endpoint testing")

    def test_session_api_endpoints_exist(self, app_with_fastapi):
        """Test that session management endpoints are registered."""
        if not app_with_fastapi.app:
            pytest.skip("FastAPI app not available")

        # Check that routes exist
        routes = [route.path for route in app_with_fastapi.app.routes]

        expected_routes = ["/sessions/{session_id}", "/registry/reload"]

        # Note: FastAPI adds parameter routes differently, so we check for basic patterns
        session_routes = [r for r in routes if "/sessions/" in r]
        registry_routes = [r for r in routes if "/registry/" in r]

        assert len(session_routes) > 0
        assert len(registry_routes) > 0


class TestChatRequestEnhancements:
    """Test enhanced ChatRequest with subject_id support."""

    def test_chat_request_with_subject_id(self):
        """Test ChatRequest with new subject_id field."""
        request = ChatRequest(
            message="Test message",
            session_id="session_123",
            subject_id="user_456",
            agent_id="test_agent",
        )

        assert request.message == "Test message"
        assert request.session_id == "session_123"
        assert request.subject_id == "user_456"
        assert request.agent_id == "test_agent"

    def test_chat_request_backward_compatibility(self):
        """Test ChatRequest maintains backward compatibility with user_id."""
        request = ChatRequest(message="Test message", user_id="legacy_user")

        assert request.message == "Test message"
        assert request.user_id == "legacy_user"
        assert request.subject_id is None  # New field defaults to None

    def test_chat_request_validation(self):
        """Test ChatRequest validation."""
        # Valid request
        request = ChatRequest(message="Valid message")
        assert request.message == "Valid message"

        # Test field constraints would go here if using Pydantic validation


@pytest.mark.asyncio
class TestKernelFactoryIntegration:
    """Test KernelFactory integration with registry client."""

    @patch("energyai_sdk.core.SEMANTIC_KERNEL_AVAILABLE", True)
    @patch("energyai_sdk.core.KernelFactory.create_kernel")
    async def test_load_tools_from_registry(self, mock_create_kernel):
        """Test loading tools from registry into kernel."""
        from energyai_sdk.core import KernelFactory

        # Mock kernel
        mock_kernel = MagicMock()
        mock_create_kernel.return_value = mock_kernel

        # Create registry client with tools
        registry_client = MockRegistryClient()

        # Test tool loading
        loaded_count = await KernelFactory.load_tools_from_registry(mock_kernel, registry_client)

        # Should have loaded at least one tool from mock registry
        assert loaded_count >= 1

    async def test_load_tools_no_kernel(self):
        """Test tool loading with no kernel returns 0."""
        from energyai_sdk.core import KernelFactory

        registry_client = MockRegistryClient()
        loaded_count = await KernelFactory.load_tools_from_registry(None, registry_client)

        assert loaded_count == 0

    async def test_load_tools_no_registry(self):
        """Test tool loading with no registry returns 0."""
        from energyai_sdk.core import KernelFactory

        mock_kernel = MagicMock()
        loaded_count = await KernelFactory.load_tools_from_registry(mock_kernel, None)

        assert loaded_count == 0


# Performance and stress tests
@pytest.mark.asyncio
class TestPerformanceWithAzure:
    """Test performance aspects of Azure integration."""

    async def test_concurrent_session_operations(self):
        """Test concurrent session operations."""
        import asyncio

        context_store = MockContextStoreClient()

        # Create multiple sessions concurrently
        tasks = []
        for i in range(10):
            task = context_store.create_session(f"session_{i}", f"user_{i}")
            tasks.append(task)

        sessions = await asyncio.gather(*tasks)
        assert len(sessions) == 10

        # Retrieve all sessions concurrently
        retrieve_tasks = []
        for i in range(10):
            task = context_store.get_session(f"session_{i}")
            retrieve_tasks.append(task)

        retrieved_sessions = await asyncio.gather(*retrieve_tasks)
        assert len([s for s in retrieved_sessions if s is not None]) == 10

    def test_monitoring_metric_performance(self):
        """Test monitoring performance with many metrics."""
        monitoring_client = MockMonitoringClient()

        # Record many metrics
        for i in range(100):
            monitoring_client.record_metric(f"test_metric_{i}", float(i), {"batch": "test"})

        metrics = monitoring_client.get_recorded_metrics()
        assert len(metrics) == 100

    def test_monitoring_span_nesting(self):
        """Test nested span performance."""
        monitoring_client = MockMonitoringClient()

        with monitoring_client.start_span("outer_span"):
            with monitoring_client.start_span("inner_span_1"):
                with monitoring_client.start_span("inner_span_2"):
                    pass

        traces = monitoring_client.get_recorded_traces()
        assert len(traces) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
