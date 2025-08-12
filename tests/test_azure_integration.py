"""
Tests for Azure integration features including Registry Client, Context Store, and Monitoring.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from energyai_sdk.clients import (
    ContextStoreClient,
    MockContextStoreClient,
    MockMonitoringClient,
    MockRegistryClient,
    MonitoringClient,
    MonitoringConfig,
    RegistryClient,
)
from energyai_sdk.clients.context_store_client import SessionDocument
from energyai_sdk.clients.registry_client import AgentDefinition, ToolDefinition
from energyai_sdk.exceptions import SDKError


class TestMockRegistryClient:
    """Test the mock registry client."""

    @pytest.fixture
    def mock_registry(self):
        return MockRegistryClient()

    @pytest.mark.asyncio
    async def test_get_tool_definition(self, mock_registry):
        """Test fetching tool definition."""
        tool_def = await mock_registry.get_tool_definition("energy_calculator")

        assert tool_def is not None
        assert tool_def.name == "Energy Calculator"
        assert tool_def.category == "energy"
        assert "energy" in tool_def.tags
        assert tool_def.version == "1.2.0"

    @pytest.mark.asyncio
    async def test_get_nonexistent_tool(self, mock_registry):
        """Test fetching non-existent tool returns None."""
        tool_def = await mock_registry.get_tool_definition("nonexistent_tool")
        assert tool_def is None

    @pytest.mark.asyncio
    async def test_get_agent_definition(self, mock_registry):
        """Test fetching agent definition."""
        agent_def = await mock_registry.get_agent_definition("energy_analyst")

        assert agent_def is not None
        assert agent_def.name == "Energy Analyst"
        assert agent_def.temperature == 0.3
        assert "energy_calculator" in agent_def.tools
        assert "financial_analysis" in agent_def.capabilities

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_registry):
        """Test listing tools."""
        tools = await mock_registry.list_tools()
        assert len(tools) > 0
        assert any(tool.name == "Energy Calculator" for tool in tools)

    @pytest.mark.asyncio
    async def test_list_agents(self, mock_registry):
        """Test listing agents."""
        agents = await mock_registry.list_agents()
        assert len(agents) > 0
        assert any(agent.name == "Energy Analyst" for agent in agents)

    @pytest.mark.asyncio
    async def test_health_check(self, mock_registry):
        """Test health check."""
        is_healthy = await mock_registry.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_registry):
        """Test async context manager."""
        async with mock_registry:
            tool_def = await mock_registry.get_tool_definition("energy_calculator")
            assert tool_def is not None


class TestMockContextStoreClient:
    """Test the mock context store client."""

    @pytest.fixture
    def mock_context_store(self):
        return MockContextStoreClient()

    @pytest.mark.asyncio
    async def test_create_session(self, mock_context_store):
        """Test creating a session."""
        session_id = "test_session_123"
        subject_id = "user_456"

        session_doc = await mock_context_store.create_session(
            session_id=session_id, subject_id=subject_id, initial_context={"test": "data"}
        )

        assert session_doc.session_id == session_id
        assert session_doc.subject_id == subject_id
        assert session_doc.context["test"] == "data"

    @pytest.mark.asyncio
    async def test_create_duplicate_session(self, mock_context_store):
        """Test creating duplicate session raises error."""
        session_id = "duplicate_session"
        subject_id = "user_123"

        await mock_context_store.create_session(session_id, subject_id)

        with pytest.raises(SDKError, match="already exists"):
            await mock_context_store.create_session(session_id, subject_id)

    @pytest.mark.asyncio
    async def test_get_session(self, mock_context_store):
        """Test retrieving session."""
        session_id = "retrieve_test"
        subject_id = "user_789"

        # Create session first
        await mock_context_store.create_session(session_id, subject_id)

        # Retrieve it
        session_doc = await mock_context_store.get_session(session_id)
        assert session_doc is not None
        assert session_doc.session_id == session_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, mock_context_store):
        """Test getting non-existent session returns None."""
        session_doc = await mock_context_store.get_session("nonexistent")
        assert session_doc is None

    @pytest.mark.asyncio
    async def test_update_session(self, mock_context_store):
        """Test updating session context."""
        session_id = "update_test"
        subject_id = "user_update"

        # Create session
        await mock_context_store.create_session(session_id, subject_id)

        # Update it
        new_context = {"updated": True, "data": "new_value"}
        updated_session = await mock_context_store.update_session(session_id, new_context)

        assert updated_session.context == new_context
        assert updated_session.updated_at > updated_session.created_at

    @pytest.mark.asyncio
    async def test_append_to_context(self, mock_context_store):
        """Test appending to session context."""
        session_id = "append_test"
        subject_id = "user_append"

        # Create session with initial context
        await mock_context_store.create_session(
            session_id, subject_id, initial_context={"messages": []}
        )

        # Append to messages
        await mock_context_store.append_to_context(session_id, "messages", {"user": "Hello"})

        session_doc = await mock_context_store.get_session(session_id)
        assert len(session_doc.context["messages"]) == 1
        assert session_doc.context["messages"][0]["user"] == "Hello"

    @pytest.mark.asyncio
    async def test_delete_session(self, mock_context_store):
        """Test deleting session."""
        session_id = "delete_test"
        subject_id = "user_delete"

        # Create and then delete
        await mock_context_store.create_session(session_id, subject_id)
        success = await mock_context_store.delete_session(session_id)

        assert success is True

        # Verify it's gone
        session_doc = await mock_context_store.get_session(session_id)
        assert session_doc is None

    @pytest.mark.asyncio
    async def test_list_sessions_by_subject(self, mock_context_store):
        """Test listing sessions by subject."""
        subject_id = "list_user"

        # Create multiple sessions for same subject
        await mock_context_store.create_session("session_1", subject_id)
        await mock_context_store.create_session("session_2", subject_id)
        await mock_context_store.create_session("session_3", "other_user")

        sessions = await mock_context_store.list_sessions_by_subject(subject_id)

        assert len(sessions) == 2
        assert all(s.subject_id == subject_id for s in sessions)

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, mock_context_store):
        """Test cleanup of expired sessions."""
        # Mock client doesn't implement real expiration, but test the interface
        cleaned_count = await mock_context_store.cleanup_expired_sessions()
        assert isinstance(cleaned_count, int)
        assert cleaned_count >= 0


class TestMockMonitoringClient:
    """Test the mock monitoring client."""

    @pytest.fixture
    def mock_monitoring(self):
        return MockMonitoringClient()

    def test_initialization(self, mock_monitoring):
        """Test monitoring client initialization."""
        assert mock_monitoring._initialized is True
        assert len(mock_monitoring._traces) == 0
        assert len(mock_monitoring._metrics) == 0

    def test_record_metric(self, mock_monitoring):
        """Test recording metrics."""
        mock_monitoring.record_metric("test_counter", 5.0, {"service": "test"})

        metrics = mock_monitoring.get_recorded_metrics()
        assert len(metrics) == 1
        assert metrics[0]["name"] == "test_counter"
        assert metrics[0]["value"] == 5.0
        assert metrics[0]["tags"]["service"] == "test"

    def test_start_span(self, mock_monitoring):
        """Test span creation."""
        with mock_monitoring.start_span("test_operation", service="test") as span:
            assert span is not None
            assert span["name"] == "test_operation"

        traces = mock_monitoring.get_recorded_traces()
        assert len(traces) == 1
        assert traces[0]["name"] == "test_operation"

    def test_health_check(self, mock_monitoring):
        """Test monitoring health check."""
        assert mock_monitoring.health_check() is True

    def test_clear_data(self, mock_monitoring):
        """Test clearing recorded data."""
        # Record some data
        mock_monitoring.record_metric("test", 1.0)
        with mock_monitoring.start_span("test"):
            pass

        # Verify data exists
        assert len(mock_monitoring.get_recorded_traces()) > 0
        assert len(mock_monitoring.get_recorded_metrics()) > 0

        # Clear and verify
        mock_monitoring.clear_data()
        assert len(mock_monitoring.get_recorded_traces()) == 0
        assert len(mock_monitoring.get_recorded_metrics()) == 0


class TestRealClientInitialization:
    """Test real client initialization (without actual connections)."""

    @patch("energyai_sdk.clients.registry_client.COSMOS_AVAILABLE", True)
    def test_registry_client_init_with_cosmos_available(self):
        """Test RegistryClient initialization when Cosmos SDK is available."""
        client = RegistryClient(
            cosmos_endpoint="https://test.documents.azure.com:443/", cosmos_key="test_key"
        )

        assert client.cosmos_endpoint == "https://test.documents.azure.com:443/"
        assert client.database_name == "energyai_platform"
        assert client.agents_container == "agents"
        assert client.tools_container == "tools"

    @patch("energyai_sdk.clients.registry_client.COSMOS_AVAILABLE", False)
    def test_registry_client_init_without_cosmos(self):
        """Test RegistryClient initialization fails when Cosmos SDK not available."""
        with pytest.raises(SDKError, match="Azure Cosmos DB SDK not available"):
            RegistryClient("https://test.com", "key")

    @patch("energyai_sdk.clients.context_store_client.COSMOS_AVAILABLE", True)
    def test_context_store_client_init_with_cosmos_available(self):
        """Test ContextStoreClient initialization when Cosmos SDK is available."""
        client = ContextStoreClient(
            cosmos_endpoint="https://test.documents.azure.com:443/",
            cosmos_key="test_key",
            default_ttl=7200,
        )

        assert client.cosmos_endpoint == "https://test.documents.azure.com:443/"
        assert client.default_ttl == 7200

    @patch("energyai_sdk.clients.monitoring.OTEL_AVAILABLE", True)
    def test_monitoring_client_init_with_otel_available(self):
        """Test MonitoringClient initialization when OpenTelemetry is available."""
        config = MonitoringConfig(service_name="test-service", environment="test")

        client = MonitoringClient(config)
        assert client.config.service_name == "test-service"
        assert client.config.environment == "test"

    @patch("energyai_sdk.clients.monitoring.OTEL_AVAILABLE", False)
    def test_monitoring_client_init_without_otel(self):
        """Test MonitoringClient initialization fails when OpenTelemetry not available."""
        config = MonitoringConfig()

        with pytest.raises(SDKError, match="OpenTelemetry not available"):
            MonitoringClient(config)


class TestDataModels:
    """Test data model classes."""

    def test_tool_definition_creation(self):
        """Test ToolDefinition creation."""
        now = datetime.now(timezone.utc)

        tool_def = ToolDefinition(
            id="test_tool",
            name="Test Tool",
            description="A test tool",
            category="testing",
            schema={"type": "function"},
            created_at=now,
            tags=["test", "example"],
        )

        assert tool_def.id == "test_tool"
        assert tool_def.name == "Test Tool"
        assert tool_def.category == "testing"
        assert tool_def.created_at == now
        assert "test" in tool_def.tags

    def test_agent_definition_creation(self):
        """Test AgentDefinition creation."""
        agent_def = AgentDefinition(
            id="test_agent",
            name="Test Agent",
            description="A test agent",
            system_prompt="You are a test agent",
            model_config={"deployment": "gpt-4"},
            tools=["tool1", "tool2"],
            capabilities=["cap1", "cap2"],
        )

        assert agent_def.id == "test_agent"
        assert agent_def.system_prompt == "You are a test agent"
        assert len(agent_def.tools) == 2
        assert len(agent_def.capabilities) == 2

    def test_session_document_creation(self):
        """Test SessionDocument creation."""
        now = datetime.now(timezone.utc)

        session_doc = SessionDocument(
            session_id="test_session",
            subject_id="test_user",
            created_at=now,
            updated_at=now,
            context={"messages": []},
            ttl=3600,
        )

        assert session_doc.session_id == "test_session"
        assert session_doc.subject_id == "test_user"
        assert session_doc.ttl == 3600

    def test_session_document_to_dict(self):
        """Test SessionDocument serialization."""
        now = datetime.now(timezone.utc)

        session_doc = SessionDocument(
            session_id="test",
            subject_id="user",
            created_at=now,
            updated_at=now,
            context={"data": "test"},
        )

        data = session_doc.to_dict()

        assert data["id"] == "test"
        assert data["session_id"] == "test"
        assert data["subject_id"] == "user"
        assert data["context"]["data"] == "test"
        assert "created_at" in data

    def test_session_document_from_dict(self):
        """Test SessionDocument deserialization."""
        now = datetime.now(timezone.utc)

        data = {
            "session_id": "test",
            "subject_id": "user",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "context": {"data": "test"},
            "metadata": {"key": "value"},
            "ttl": 1800,
        }

        session_doc = SessionDocument.from_dict(data)

        assert session_doc.session_id == "test"
        assert session_doc.subject_id == "user"
        assert session_doc.context["data"] == "test"
        assert session_doc.ttl == 1800


class TestMonitoringConfig:
    """Test MonitoringConfig class."""

    def test_default_config(self):
        """Test default monitoring configuration."""
        config = MonitoringConfig()

        assert config.service_name == "energyai-sdk"
        assert config.environment == "development"
        assert config.enable_traces is True
        assert config.enable_metrics is True
        assert config.trace_sample_rate == 1.0

    def test_custom_config(self):
        """Test custom monitoring configuration."""
        config = MonitoringConfig(
            service_name="custom-service",
            environment="production",
            azure_monitor_connection_string="InstrumentationKey=test",
            otlp_trace_endpoint="https://otel.example.com:4317",
            trace_sample_rate=0.1,
            enable_traces=False,
        )

        assert config.service_name == "custom-service"
        assert config.environment == "production"
        assert config.azure_monitor_connection_string == "InstrumentationKey=test"
        assert config.trace_sample_rate == 0.1
        assert config.enable_traces is False


# Integration test fixtures
@pytest.fixture
def sample_tool_definition():
    """Sample tool definition for testing."""
    return ToolDefinition(
        id="sample_tool",
        name="Sample Tool",
        description="A sample tool for testing",
        category="testing",
        schema={
            "type": "function",
            "function": {
                "name": "sample_function",
                "parameters": {"type": "object", "properties": {"input": {"type": "string"}}},
            },
        },
        endpoint_url="https://api.example.com/tools/sample",
        version="1.0.0",
        tags=["sample", "test"],
    )


@pytest.fixture
def sample_agent_definition():
    """Sample agent definition for testing."""
    return AgentDefinition(
        id="sample_agent",
        name="Sample Agent",
        description="A sample agent for testing",
        system_prompt="You are a helpful sample agent",
        model_config={"deployment": "gpt-4", "temperature": 0.5},
        tools=["sample_tool"],
        capabilities=["testing", "examples"],
        temperature=0.5,
        max_tokens=1000,
        tags=["sample", "test"],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
