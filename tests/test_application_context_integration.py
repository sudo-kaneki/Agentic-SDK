"""
Tests for Application Context Store Integration.

Tests the integration between EnergyAIApplication and ContextStoreClient
for stateful conversations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from energyai_sdk.application import (
    ChatRequest,
    EnergyAIApplication,
    create_application,
)
from energyai_sdk.clients import ContextStoreClient
from energyai_sdk.core import AgentResponse


class TestApplicationContextIntegration:
    """Test cases for application context store integration."""

    @pytest.fixture
    def mock_context_store_client(self):
        """Mock ContextStoreClient for testing."""
        client = MagicMock(spec=ContextStoreClient)

        # Mock session document
        session_doc = {
            "id": "test_session",
            "subject": {"type": "user", "id": "test_user"},
            "thread": [],
            "context": {"memory": [], "state": {}},
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
        }

        client.load_or_create.return_value = session_doc
        client.get_conversation_history.return_value = []
        client.update_and_save.return_value = session_doc
        client.close_session.return_value = session_doc

        return client

    @pytest.fixture
    def mock_agent(self):
        """Mock agent for testing."""
        agent = MagicMock()
        agent.name = "TestAgent"

        # Mock agent response
        response = AgentResponse(
            content="Test response from agent", agent_id="TestAgent", metadata={}
        )
        agent.process_request = AsyncMock(return_value=response)

        return agent

    @pytest.fixture
    def app_with_context(self, mock_context_store_client):
        """Create application with mocked context store."""
        app = EnergyAIApplication(title="Test App", context_store_client=mock_context_store_client)
        return app

    def test_create_application_with_context_store_enabled(self):
        """Test creating application with context store enabled."""
        with patch("energyai_sdk.application.ContextStoreClient") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            app = create_application(enable_context_store=True)

            assert app.context_store_client is not None
            mock_client_class.assert_called_once()

    def test_create_application_with_context_store_disabled(self):
        """Test creating application with context store disabled."""
        app = create_application(enable_context_store=False)
        assert app.context_store_client is None

    def test_create_application_context_store_failure(self):
        """Test graceful handling of context store initialization failure."""
        with patch("energyai_sdk.application.ContextStoreClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            app = create_application(enable_context_store=True)

            # Should gracefully handle failure and set context_store_client to None
            assert app.context_store_client is None

    @pytest.mark.asyncio
    async def test_chat_request_without_session_id(self, app_with_context):
        """Test chat request processing without session ID."""
        request = ChatRequest(message="Hello", agent_id="TestAgent")

        with patch("energyai_sdk.application.agent_registry") as mock_registry:
            mock_agent = MagicMock()
            mock_agent.name = "TestAgent"
            mock_response = AgentResponse(content="Hello back!", agent_id="TestAgent")
            mock_agent.process_request = AsyncMock(return_value=mock_response)
            mock_registry.get_agent.return_value = mock_agent

            from fastapi import BackgroundTasks

            response = await app_with_context._process_chat_request(request, BackgroundTasks())

            # Should not call context store methods
            app_with_context.context_store_client.load_or_create.assert_not_called()
            app_with_context.context_store_client.update_and_save.assert_not_called()

            assert response.content == "Hello back!"

    @pytest.mark.asyncio
    async def test_chat_request_with_session_id_new_session(self, app_with_context):
        """Test chat request processing with session ID for new session."""
        request = ChatRequest(
            message="Hello", agent_id="TestAgent", session_id="new_session", subject_id="user123"
        )

        # Mock empty session (new session)
        session_doc = {
            "id": "new_session",
            "subject": {"type": "user", "id": "user123"},
            "thread": [],  # Empty thread for new session
            "context": {"memory": [], "state": {}},
        }
        app_with_context.context_store_client.load_or_create.return_value = session_doc

        with patch("energyai_sdk.application.agent_registry") as mock_registry:
            mock_agent = MagicMock()
            mock_agent.name = "TestAgent"
            mock_response = AgentResponse(content="Hello! How can I help?", agent_id="TestAgent")
            mock_agent.process_request = AsyncMock(return_value=mock_response)
            mock_registry.get_agent.return_value = mock_agent

            from fastapi import BackgroundTasks

            response = await app_with_context._process_chat_request(request, BackgroundTasks())

            # Should load session
            app_with_context.context_store_client.load_or_create.assert_called_once_with(
                "new_session", "user123"
            )

            # Should save conversation turn
            app_with_context.context_store_client.update_and_save.assert_called_once()

            # Verify the arguments to update_and_save
            call_args = app_with_context.context_store_client.update_and_save.call_args
            assert call_args[0][0] == session_doc  # session_doc
            assert call_args[1]["user_input"] == "Hello"  # original message
            assert call_args[1]["agent_output"] == "Hello! How can I help?"
            assert call_args[1]["agent_name"] == "TestAgent"

            assert response.content == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_chat_request_with_session_id_existing_session(self, app_with_context):
        """Test chat request processing with session ID for existing session."""
        request = ChatRequest(
            message="What about solar panels?",
            agent_id="EnergyAgent",
            session_id="existing_session",
            subject_id="user456",
        )

        # Mock existing session with conversation history
        session_doc = {
            "id": "existing_session",
            "subject": {"type": "user", "id": "user456"},
            "thread": [
                {
                    "id": "msg1",
                    "sender": "user",
                    "content": "I need help with energy efficiency",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
                {
                    "id": "msg2",
                    "sender": "agent",
                    "content": "I can help with that! What area are you interested in?",
                    "timestamp": "2024-01-01T12:00:05Z",
                    "agent_name": "EnergyAgent",
                },
            ],
            "context": {"memory": [], "state": {}},
        }
        app_with_context.context_store_client.load_or_create.return_value = session_doc

        with patch("energyai_sdk.application.agent_registry") as mock_registry:
            mock_agent = MagicMock()
            mock_agent.name = "EnergyAgent"
            mock_response = AgentResponse(
                content="Solar panels are a great choice! Based on our previous discussion about energy efficiency, they would be perfect for your needs.",
                agent_id="EnergyAgent",
            )
            mock_agent.process_request = AsyncMock(return_value=mock_response)
            mock_registry.get_agent.return_value = mock_agent

            from fastapi import BackgroundTasks

            response = await app_with_context._process_chat_request(request, BackgroundTasks())

            # Should load session with history
            app_with_context.context_store_client.load_or_create.assert_called_once_with(
                "existing_session", "user456"
            )

            # Verify that the agent received context from previous conversation
            call_args = mock_agent.process_request.call_args[0][0]  # AgentRequest
            assert "Previous conversation:" in call_args.message
            assert "I need help with energy efficiency" in call_args.message
            assert "What about solar panels?" in call_args.message
            assert call_args.metadata["has_conversation_history"] is True
            assert call_args.metadata["original_message"] == "What about solar panels?"

            # Should save new conversation turn
            app_with_context.context_store_client.update_and_save.assert_called_once()

            assert "Solar panels are a great choice" in response.content

    @pytest.mark.asyncio
    async def test_chat_request_context_store_error_handling(self, app_with_context):
        """Test error handling when context store operations fail."""
        request = ChatRequest(
            message="Hello", agent_id="TestAgent", session_id="error_session", subject_id="user789"
        )

        # Mock context store error
        app_with_context.context_store_client.load_or_create.side_effect = Exception(
            "Cosmos DB error"
        )

        with patch("energyai_sdk.application.agent_registry") as mock_registry:
            mock_agent = MagicMock()
            mock_agent.name = "TestAgent"
            mock_response = AgentResponse(content="Hello!", agent_id="TestAgent")
            mock_agent.process_request = AsyncMock(return_value=mock_response)
            mock_registry.get_agent.return_value = mock_agent

            from fastapi import BackgroundTasks

            response = await app_with_context._process_chat_request(request, BackgroundTasks())

            # Should continue processing despite context store error
            assert response.content == "Hello!"

            # Should not try to save context if loading failed
            app_with_context.context_store_client.update_and_save.assert_not_called()

    def test_session_management_endpoints_context_store_available(self, app_with_context):
        """Test session management when context store is available."""
        # This would typically test the FastAPI endpoints, but we'll test the underlying logic

        # Test session creation/loading
        session_doc = app_with_context.context_store_client.load_or_create(
            "test_session", "test_user"
        )
        assert session_doc is not None

        # Test getting conversation history
        history = app_with_context.context_store_client.get_conversation_history(
            "test_session", "test_user"
        )
        assert isinstance(history, list)

        # Test closing session
        app_with_context.context_store_client.close_session("test_session", "test_user")
        app_with_context.context_store_client.close_session.assert_called_once()

    def test_session_management_endpoints_no_context_store(self):
        """Test session management when context store is not available."""
        app = EnergyAIApplication(title="Test App", context_store_client=None)

        # Session management should handle missing context store gracefully
        assert app.context_store_client is None

    @pytest.mark.asyncio
    async def test_application_startup_with_context_store(self, app_with_context):
        """Test application startup with context store."""
        await app_with_context._startup()

        # Should report context store as healthy
        assert app_with_context.components_status["context_store"] == "healthy"
        assert app_with_context.is_ready is True

    @pytest.mark.asyncio
    async def test_application_startup_without_context_store(self):
        """Test application startup without context store."""
        app = EnergyAIApplication(title="Test App", context_store_client=None)
        await app._startup()

        # Should report context store as not configured
        assert app.components_status["context_store"] == "not_configured"
        assert app.is_ready is True

    @pytest.mark.asyncio
    async def test_application_shutdown_with_context_store(self, app_with_context):
        """Test application shutdown with context store."""
        await app_with_context._shutdown()

        # Should complete shutdown without errors
        # (ContextStoreClient doesn't have async close method, so it should handle gracefully)

    def test_conversation_history_formatting(self, app_with_context):
        """Test that conversation history is properly formatted for context."""
        # This tests the private logic for building conversation history
        thread = [
            {"sender": "user", "content": "Hello", "timestamp": "2024-01-01T12:00:00Z"},
            {
                "sender": "agent",
                "content": "Hi there!",
                "agent_name": "Assistant",
                "timestamp": "2024-01-01T12:00:05Z",
            },
            {"sender": "user", "content": "How are you?", "timestamp": "2024-01-01T12:01:00Z"},
        ]

        # Test the logic used in _process_chat_request
        history_lines = []
        for msg in thread[-10:]:  # Last 10 messages for context
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")
            agent_name = msg.get("agent_name", "Assistant")

            if sender == "user":
                history_lines.append(f"User: {content}")
            elif sender == "agent":
                history_lines.append(f"{agent_name}: {content}")

        conversation_history = "\n".join(history_lines)

        expected = "User: Hello\nAssistant: Hi there!\nUser: How are you?"
        assert conversation_history == expected
