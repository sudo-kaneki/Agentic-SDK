# tests/test_application.py
"""
Test FastAPI application framework and web service integration.
"""

from unittest.mock import Mock, patch

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from energyai_sdk import AgentResponse, agent_registry
from energyai_sdk.application import (
    AgentInfo,
    ChatRequest,
    ChatResponse,
    DevelopmentServer,
    EnergyAIApplication,
    HealthCheck,
    create_application,
    create_production_application,
)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestEnergyAIApplication:
    """Test EnergyAIApplication core functionality."""

    def test_application_initialization(self):
        """Test application initialization."""

        app = EnergyAIApplication(
            title="Test Application", version="1.0.0", description="Test description", debug=True
        )

        assert app.title == "Test Application"
        assert app.version == "1.0.0"
        assert app.description == "Test description"
        assert app.debug is True
        assert app.start_time is not None
        assert app.is_ready is False  # Not ready until startup

    def test_application_with_fastapi(self):
        """Test that FastAPI app is created when available."""

        app = EnergyAIApplication()

        assert app.app is not None
        assert isinstance(app.app, FastAPI)

    def test_application_add_agent(self, sample_agent):
        """Test adding agents to application."""

        app = EnergyAIApplication()

        # Agent should not be in registry initially
        initial_count = len(agent_registry.agents)

        app.add_agent(sample_agent)

        # Agent should be added to registry
        assert len(agent_registry.agents) == initial_count + 1
        assert sample_agent.agent_name in agent_registry.agents

    def test_application_set_middleware_pipeline(self, middleware_pipeline):
        """Test setting middleware pipeline."""

        app = EnergyAIApplication()
        original_pipeline = app.middleware_pipeline

        app.set_middleware_pipeline(middleware_pipeline)

        assert app.middleware_pipeline == middleware_pipeline
        assert app.middleware_pipeline != original_pipeline


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestFastAPIEndpoints:
    """Test FastAPI endpoints functionality."""

    def test_health_endpoint(self, fastapi_test_client):
        """Test health check endpoint."""

        response = fastapi_test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "agents_count" in data
        assert "uptime_seconds" in data
        assert "telemetry_configured" in data
        assert "components" in data

    def test_list_agents_endpoint_empty(self, fastapi_test_client):
        """Test list agents endpoint with no agents."""

        response = fastapi_test_client.get("/agents")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        # May be empty or have test agents depending on test order

    def test_list_agents_endpoint_with_agents(self, fastapi_test_client, sample_agent):
        """Test list agents endpoint with agents."""

        response = fastapi_test_client.get("/agents")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

        # Find our test agent
        test_agent_info = None
        for agent_info in data:
            if agent_info["agent_id"] == sample_agent.agent_name:
                test_agent_info = agent_info
                break

        assert test_agent_info is not None
        assert test_agent_info["name"] == sample_agent.agent_name
        assert test_agent_info["is_available"] is True

    def test_get_agent_info_endpoint(self, fastapi_test_client, sample_agent):
        """Test get specific agent info endpoint."""

        response = fastapi_test_client.get(f"/agents/{sample_agent.agent_name}")

        assert response.status_code == 200
        data = response.json()

        assert data["agent_id"] == sample_agent.agent_name
        assert data["name"] == sample_agent.agent_name
        assert data["is_available"] is True
        assert "capabilities" in data
        assert "models" in data

    def test_get_agent_info_not_found(self, fastapi_test_client):
        """Test get agent info for non-existent agent."""

        response = fastapi_test_client.get("/agents/NonExistentAgent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch("energyai_sdk.application.EnergyAIApplication._process_chat_request")
    def test_chat_endpoint(self, mock_process, fastapi_test_client, sample_agent_response):
        """Test chat endpoint."""

        mock_process.return_value = ChatResponse(
            content=sample_agent_response.content,
            agent_id=sample_agent_response.agent_id,
            session_id=sample_agent_response.session_id,
            execution_time_ms=sample_agent_response.execution_time_ms,
        )

        chat_data = {
            "message": "Hello, test message",
            "agent_id": "TestAgent",
            "session_id": "test_session",
        }

        response = fastapi_test_client.post("/chat", json=chat_data)

        assert response.status_code == 200
        data = response.json()

        assert data["content"] == sample_agent_response.content
        assert data["agent_id"] == sample_agent_response.agent_id
        assert data["session_id"] == sample_agent_response.session_id

        mock_process.assert_called_once()

    @patch("energyai_sdk.application.EnergyAIApplication._process_chat_request")
    def test_agent_specific_chat_endpoint(
        self, mock_process, fastapi_test_client, sample_agent_response
    ):
        """Test agent-specific chat endpoint."""

        mock_process.return_value = ChatResponse(
            content=sample_agent_response.content,
            agent_id="SpecificAgent",
            session_id=sample_agent_response.session_id,
            execution_time_ms=sample_agent_response.execution_time_ms,
        )

        chat_data = {"message": "Hello, specific agent", "session_id": "test_session"}

        response = fastapi_test_client.post("/agents/SpecificAgent/chat", json=chat_data)

        assert response.status_code == 200
        data = response.json()

        assert data["agent_id"] == "SpecificAgent"
        mock_process.assert_called_once()

    def test_reset_agent_context_endpoint(self, fastapi_test_client, sample_agent):
        """Test reset agent context endpoint."""

        response = fastapi_test_client.post(f"/agents/{sample_agent.agent_name}/reset")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert sample_agent.agent_name in data["message"]

    def test_reset_agent_context_not_found(self, fastapi_test_client):
        """Test reset context for non-existent agent."""

        response = fastapi_test_client.post("/agents/NonExistentAgent/reset")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_registry_capabilities_endpoint(self, fastapi_test_client):
        """Test registry capabilities endpoint."""

        response = fastapi_test_client.get("/registry/capabilities")

        assert response.status_code == 200
        data = response.json()

        assert "agents" in data
        assert "tools" in data
        assert "prompts" in data
        assert "skills" in data
        assert "planners" in data

    def test_tools_openapi_endpoint(self, fastapi_test_client, sample_tool):
        """Test tools OpenAPI schema endpoint."""

        response = fastapi_test_client.get("/tools/openapi")

        assert response.status_code == 200
        data = response.json()

        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        assert "components" in data

        # Should include our test tool
        if agent_registry.tools:
            assert len(data["paths"]) > 0


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestChatProcessing:
    """Test chat request processing."""

    @pytest.mark.asyncio
    async def test_process_chat_request_success(
        self, test_application, sample_agent, sample_agent_response
    ):
        """Test successful chat request processing."""

        # Add agent to application
        test_application.add_agent(sample_agent)

        # Mock agent processing
        with patch.object(sample_agent, "process_request", return_value=sample_agent_response):
            chat_request = ChatRequest(
                message="Test message", agent_id=sample_agent.agent_name, session_id="test_session"
            )

            response = await test_application._process_chat_request(chat_request, None)

            assert isinstance(response, ChatResponse)
            assert response.content == sample_agent_response.content
            assert response.agent_id == sample_agent.agent_name
            assert response.execution_time_ms is not None

    @pytest.mark.asyncio
    async def test_process_chat_request_no_agent_specified(self, test_application, sample_agent):
        """Test chat request with no agent specified."""

        test_application.add_agent(sample_agent)

        with patch.object(sample_agent, "process_request") as mock_process:
            mock_process.return_value = AgentResponse(
                content="Response", agent_id=sample_agent.agent_name
            )

            chat_request = ChatRequest(
                message="Test message"
                # No agent_id specified
            )

            response = await test_application._process_chat_request(chat_request, None)

            # Should use the first available agent
            assert response.agent_id == sample_agent.agent_name
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_chat_request_no_agents_available(self, test_application):
        """Test chat request when no agents are available."""

        chat_request = ChatRequest(message="Test message", agent_id="NonExistentAgent")

        with pytest.raises(Exception):  # Should raise HTTPException
            await test_application._process_chat_request(chat_request, None)

    @pytest.mark.asyncio
    async def test_process_chat_request_agent_not_found(self, test_application):
        """Test chat request for non-existent agent."""

        chat_request = ChatRequest(message="Test message", agent_id="NonExistentAgent")

        with pytest.raises(Exception):  # Should raise HTTPException
            await test_application._process_chat_request(chat_request, None)

    @pytest.mark.asyncio
    async def test_process_chat_request_with_middleware(
        self, test_application, sample_agent, middleware_pipeline
    ):
        """Test chat request processing with middleware."""

        test_application.add_agent(sample_agent)
        test_application.set_middleware_pipeline(middleware_pipeline)

        # Mock successful processing
        mock_response = AgentResponse(
            content="Middleware processed response", agent_id=sample_agent.agent_name
        )

        with patch.object(middleware_pipeline, "execute") as mock_execute:

            # Mock middleware execution
            async def mock_pipeline_execute(context):
                context.response = mock_response
                return context

            mock_execute.side_effect = mock_pipeline_execute

            chat_request = ChatRequest(message="Test message", agent_id=sample_agent.agent_name)

            response = await test_application._process_chat_request(chat_request, None)

            assert response.content == mock_response.content
            mock_execute.assert_called_once()


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestDataModels:
    """Test Pydantic data models."""

    def test_chat_request_model(self):
        """Test ChatRequest model validation."""

        # Valid request
        request = ChatRequest(
            message="Test message",
            agent_id="TestAgent",
            session_id="session_123",
            temperature=0.7,
            max_tokens=1000,
            metadata={"key": "value"},
        )

        assert request.message == "Test message"
        assert request.agent_id == "TestAgent"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000

    def test_chat_request_validation_errors(self):
        """Test ChatRequest validation errors."""

        with pytest.raises(Exception):  # Pydantic validation error
            ChatRequest(message="", agent_id="TestAgent")  # Empty message should fail validation

        with pytest.raises(Exception):  # Temperature out of range
            ChatRequest(message="Test", temperature=3.0)  # Too high

    def test_chat_response_model(self):
        """Test ChatResponse model."""

        response = ChatResponse(
            content="Test response",
            agent_id="TestAgent",
            session_id="session_123",
            execution_time_ms=150,
            model_used="gpt-4o",
            metadata={"tokens": 10},
        )

        assert response.content == "Test response"
        assert response.agent_id == "TestAgent"
        assert response.execution_time_ms == 150
        assert response.model_used == "gpt-4o"
        assert response.timestamp is not None

    def test_agent_info_model(self):
        """Test AgentInfo model."""

        info = AgentInfo(
            agent_id="TestAgent",
            name="Test Agent",
            description="Test description",
            type="SemanticKernelAgent",
            capabilities=["chat", "completion"],
            models=["gpt-4o"],
            tools=["calculator"],
            skills=["math"],
            is_available=True,
            metadata={"version": "1.0"},
        )

        assert info.agent_id == "TestAgent"
        assert info.name == "Test Agent"
        assert "chat" in info.capabilities
        assert "gpt-4o" in info.models
        assert info.is_available is True

    def test_health_check_model(self):
        """Test HealthCheck model."""

        from datetime import datetime, timezone

        health = HealthCheck(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            agents_count=5,
            uptime_seconds=3600,
            telemetry_configured=True,
            components={"registry": "healthy", "pipeline": "healthy"},
        )

        assert health.status == "healthy"
        assert health.agents_count == 5
        assert health.uptime_seconds == 3600
        assert health.telemetry_configured is True


class TestApplicationFactories:
    """Test application factory functions."""

    def test_create_application_basic(self):
        """Test basic application creation."""

        app = create_application(
            title="Test App", version="2.0.0", description="Test description", debug=True
        )

        assert isinstance(app, EnergyAIApplication)
        assert app.title == "Test App"
        assert app.version == "2.0.0"
        assert app.debug is True

    def test_create_application_with_middleware_config(self):
        """Test application creation with middleware configuration."""

        app = create_application(
            enable_default_middleware=True,
            enable_cors=True,
            api_keys={"test_key"},
            max_requests_per_minute=100,
        )

        assert app.middleware_pipeline is not None
        # Should have middleware configured
        assert len(app.middleware_pipeline.preprocessing_middleware) > 0

    def test_create_application_without_middleware(self):
        """Test application creation without default middleware."""

        app = create_application(enable_default_middleware=False)

        # Should have default or empty pipeline
        assert app.middleware_pipeline is not None

    @patch("energyai_sdk.application.initialize_sdk")
    @patch("energyai_sdk.application.create_production_pipeline")
    def test_create_production_application(self, mock_create_pipeline, mock_init_sdk):
        """Test production application creation."""

        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        app = create_production_application(
            api_keys=["prod_key_1", "prod_key_2"],
            azure_monitor_connection_string="InstrumentationKey=test",
            langfuse_public_key="pk_test",
            langfuse_secret_key="sk_test",
            max_requests_per_minute=100,
        )

        assert isinstance(app, EnergyAIApplication)
        mock_init_sdk.assert_called_once()
        mock_create_pipeline.assert_called_once()


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestDevelopmentServer:
    """Test DevelopmentServer functionality."""

    def test_development_server_initialization(self, test_application):
        """Test development server initialization."""

        server = DevelopmentServer(application=test_application, host="127.0.0.1", port=8001)

        assert server.application == test_application
        assert server.host == "127.0.0.1"
        assert server.port == 8001

    @patch("energyai_sdk.application.uvicorn")
    def test_development_server_run(self, mock_uvicorn, test_application):
        """Test development server run method."""

        server = DevelopmentServer(test_application)

        # Mock uvicorn.run to avoid actually starting server
        server.run(reload=True)

        mock_uvicorn.run.assert_called_once()
        call_args = mock_uvicorn.run.call_args

        assert call_args[1]["host"] == server.host
        assert call_args[1]["port"] == server.port
        assert call_args[1]["reload"] is True


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestStreamingEndpoints:
    """Test streaming endpoints functionality."""

    @pytest.mark.asyncio
    async def test_stream_chat_response(self, test_application, sample_agent):
        """Test streaming chat response generation."""

        test_application.add_agent(sample_agent)

        chat_request = ChatRequest(
            message="Test streaming message", agent_id=sample_agent.agent_name, stream=True
        )

        # Mock the response processing
        with patch.object(test_application, "_process_chat_request") as mock_process:
            mock_process.return_value = ChatResponse(
                content="This is a streaming response for testing purposes.",
                agent_id=sample_agent.agent_name,
            )

            # Test the streaming generator
            response_chunks = []
            async for chunk in test_application._stream_chat_response(chat_request):
                response_chunks.append(chunk)

            assert len(response_chunks) > 1  # Should have multiple chunks

            # Last chunk should indicate completion
            last_chunk = response_chunks[-1]
            assert "is_final" in last_chunk

    @patch("energyai_sdk.application.EnergyAIApplication._stream_chat_response")
    def test_streaming_endpoint_integration(self, mock_stream, fastapi_test_client):
        """Test streaming endpoint integration."""

        # Mock streaming response
        async def mock_streaming():
            yield "data: {'chunk': 'Hello', 'is_final': false}\n\n"
            yield "data: {'chunk': ' world', 'is_final': false}\n\n"
            yield "data: {'chunk': '', 'is_final': true}\n\n"

        mock_stream.return_value = mock_streaming()

        chat_data = {"message": "Test streaming", "agent_id": "TestAgent", "stream": True}

        response = fastapi_test_client.post("/chat/stream", json=chat_data)

        assert response.status_code == 200
        # Should be streaming response
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


@pytest.mark.integration
class TestApplicationIntegration:
    """Integration tests for complete application functionality."""

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_complete_application_flow(self, sample_agent, sample_tool):
        """Test complete application flow from request to response."""

        # Create application with agent
        app = create_application(
            title="Integration Test App",
            enable_default_middleware=True,
            enable_auth=False,  # Disable auth for testing
        )

        app.add_agent(sample_agent)

        # Create test client
        client = TestClient(app.get_fastapi_app())

        # Test health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # Test list agents
        agents_response = client.get("/agents")
        assert agents_response.status_code == 200
        agents_data = agents_response.json()

        # Should find our agent
        agent_found = any(agent["agent_id"] == sample_agent.agent_name for agent in agents_data)
        assert agent_found

        # Test capabilities endpoint
        capabilities_response = client.get("/registry/capabilities")
        assert capabilities_response.status_code == 200
        capabilities_data = capabilities_response.json()
        assert sample_agent.agent_name in capabilities_data["agents"]

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    @patch("energyai_sdk.agents.SemanticKernelAgent.process_request")
    def test_end_to_end_chat_flow(self, mock_process, sample_agent):
        """Test end-to-end chat flow."""

        # Mock agent response
        mock_response = AgentResponse(
            content="This is a test response from the agent.",
            agent_id=sample_agent.agent_name,
            session_id="test_session",
            execution_time_ms=100,
        )
        mock_process.return_value = mock_response

        # Create application
        app = create_application(enable_auth=False)
        app.add_agent(sample_agent)

        client = TestClient(app.get_fastapi_app())

        # Send chat request
        chat_data = {
            "message": "Hello, test agent!",
            "agent_id": sample_agent.agent_name,
            "session_id": "test_session",
            "temperature": 0.7,
        }

        response = client.post("/chat", json=chat_data)

        assert response.status_code == 200
        data = response.json()

        assert data["content"] == mock_response.content
        assert data["agent_id"] == sample_agent.agent_name
        assert data["session_id"] == "test_session"
        assert data["execution_time_ms"] is not None

        # Verify agent was called with correct request
        mock_process.assert_called_once()
        call_args = mock_process.call_args[0][0]
        assert call_args.message == "Hello, test agent!"
        assert call_args.agent_id == sample_agent.agent_name


if __name__ == "__main__":
    pytest.main([__file__])
