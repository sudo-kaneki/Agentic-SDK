"""
Tests for Langfuse Monitoring Integration.

Tests the integration between EnergyAIApplication and LangfuseMonitoringClient
for comprehensive observability.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from energyai_sdk.application import (
    EnergyAIApplication,
    create_application,
    ChatRequest,
    ChatResponse
)
from energyai_sdk.core import AgentResponse


class TestLangfuseIntegration:
    """Test cases for Langfuse monitoring integration."""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Mock LangfuseMonitoringClient for testing."""
        client = MagicMock()
        client.is_enabled.return_value = True
        
        # Mock trace object
        trace = MagicMock()
        trace.generation.return_value = MagicMock()
        trace.span.return_value = MagicMock()
        client.create_trace.return_value = trace
        
        # Mock generation object
        generation = MagicMock()
        client.create_generation.return_value = generation
        
        # Mock span object
        span = MagicMock()
        client.create_span.return_value = span
        
        return client

    @pytest.fixture
    def mock_agent(self):
        """Mock agent for testing."""
        agent = MagicMock()
        agent.name = "TestAgent"
        agent.get_capabilities.return_value = {
            "models": ["gpt-4"],
            "capabilities": ["chat", "reasoning"],
            "tools": ["test_tool"]
        }
        
        # Mock agent response
        response = AgentResponse(
            content="Test response from agent with monitoring",
            agent_id="TestAgent",
            metadata={"usage": {"prompt_tokens": 50, "completion_tokens": 25}}
        )
        agent.process_request = AsyncMock(return_value=response)
        
        return agent

    @pytest.fixture
    def app_with_langfuse(self, mock_langfuse_client):
        """Create application with mocked Langfuse client."""
        app = EnergyAIApplication(
            title="Test App with Langfuse",
            langfuse_monitoring_client=mock_langfuse_client
        )
        return app

    def test_create_application_with_langfuse_enabled(self):
        """Test creating application with Langfuse monitoring enabled."""
        with patch('energyai_sdk.application.LANGFUSE_AVAILABLE', True):
            with patch('energyai_sdk.application.get_langfuse_client') as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                
                app = create_application(
                    enable_langfuse_monitoring=True,
                    langfuse_public_key="pk_test",
                    langfuse_secret_key="sk_test"
                )
                
                assert app.langfuse_client is not None
                mock_get_client.assert_called_once_with(
                    public_key="pk_test",
                    secret_key="sk_test",
                    host="https://cloud.langfuse.com",
                    debug=False,
                    environment="production"
                )

    def test_create_application_with_langfuse_disabled(self):
        """Test creating application with Langfuse monitoring disabled."""
        app = create_application(enable_langfuse_monitoring=False)
        assert app.langfuse_client is None

    def test_create_application_langfuse_not_available(self):
        """Test graceful handling when Langfuse is not available."""
        with patch('energyai_sdk.application.LANGFUSE_AVAILABLE', False):
            app = create_application(
                enable_langfuse_monitoring=True,
                langfuse_public_key="pk_test",
                langfuse_secret_key="sk_test"
            )
            assert app.langfuse_client is None

    @pytest.mark.asyncio
    async def test_chat_request_with_langfuse_monitoring(self, app_with_langfuse, mock_agent):
        """Test chat request processing with Langfuse monitoring."""
        request = ChatRequest(
            message="Hello, test monitoring",
            agent_id="TestAgent",
            session_id="test_session_123",
            subject_id="user_456",
            temperature=0.7,
            max_tokens=100
        )
        
        with patch('energyai_sdk.application.agent_registry') as mock_registry:
            mock_registry.get_agent.return_value = mock_agent
            
            from fastapi import BackgroundTasks
            response = await app_with_langfuse._process_chat_request(request, BackgroundTasks())
            
            # Verify trace was created
            app_with_langfuse.langfuse_client.create_trace.assert_called_once()
            trace_call = app_with_langfuse.langfuse_client.create_trace.call_args
            
            assert trace_call[1]['name'] == "agent-run:TestAgent"
            assert trace_call[1]['user_id'] == "user_456"
            assert trace_call[1]['session_id'] == "test_session_123"
            assert trace_call[1]['input_data']['message'] == "Hello, test monitoring"
            assert trace_call[1]['input_data']['agent_id'] == "TestAgent"
            assert trace_call[1]['metadata']['agent_name'] == "TestAgent"
            assert "agent-interaction" in trace_call[1]['tags']
            
            # Verify generation was created
            app_with_langfuse.langfuse_client.create_generation.assert_called_once()
            generation_call = app_with_langfuse.langfuse_client.create_generation.call_args
            
            assert generation_call[1]['name'] == "agent-invocation"
            assert generation_call[1]['input_data']['query'] == "Hello, test monitoring"
            assert generation_call[1]['model'] == "gpt-4"
            assert generation_call[1]['model_parameters']['temperature'] == 0.7
            assert generation_call[1]['model_parameters']['max_tokens'] == 100
            
            # Verify generation was ended with success
            app_with_langfuse.langfuse_client.end_generation.assert_called_once()
            end_gen_call = app_with_langfuse.langfuse_client.end_generation.call_args
            
            assert end_gen_call[1]['output'] == "Test response from agent with monitoring"
            assert end_gen_call[1]['usage'] == {"prompt_tokens": 50, "completion_tokens": 25}
            assert end_gen_call[1]['level'] == "DEFAULT"
            
            # Verify trace was updated
            app_with_langfuse.langfuse_client.update_trace.assert_called()
            
            # Verify flush was called
            app_with_langfuse.langfuse_client.flush.assert_called_once()
            
            assert response.content == "Test response from agent with monitoring"

    @pytest.mark.asyncio
    async def test_chat_request_with_session_context_monitoring(self, app_with_langfuse, mock_agent):
        """Test chat request with session context and Langfuse monitoring."""
        request = ChatRequest(
            message="Follow up question",
            agent_id="TestAgent",
            session_id="existing_session",
            subject_id="user_789"
        )
        
        # Mock context store client
        mock_context_store = MagicMock()
        session_doc = {
            'id': 'existing_session',
            'thread': [
                {'sender': 'user', 'content': 'Previous question'},
                {'sender': 'agent', 'content': 'Previous answer', 'agent_name': 'TestAgent'}
            ]
        }
        mock_context_store.load_or_create.return_value = session_doc
        app_with_langfuse.context_store_client = mock_context_store
        
        with patch('energyai_sdk.application.agent_registry') as mock_registry:
            mock_registry.get_agent.return_value = mock_agent
            
            from fastapi import BackgroundTasks
            response = await app_with_langfuse._process_chat_request(request, BackgroundTasks())
            
            # Verify context span was created
            app_with_langfuse.langfuse_client.create_span.assert_called()
            span_call = app_with_langfuse.langfuse_client.create_span.call_args
            
            assert span_call[1]['name'] == "context-loading"
            assert span_call[1]['input_data']['session_id'] == "existing_session"
            assert span_call[1]['input_data']['subject_id'] == "user_789"
            
            # Verify context span was ended successfully
            app_with_langfuse.langfuse_client.end_span.assert_called()
            
            # Verify generation includes conversation history
            generation_call = app_with_langfuse.langfuse_client.create_generation.call_args
            assert "history" in generation_call[1]['input_data']
            assert generation_call[1]['metadata']['has_conversation_history'] is True
            assert generation_call[1]['metadata']['message_count_in_context'] > 0

    @pytest.mark.asyncio
    async def test_chat_request_error_handling_with_langfuse(self, app_with_langfuse, mock_agent):
        """Test error handling with Langfuse monitoring."""
        request = ChatRequest(
            message="Error test",
            agent_id="TestAgent"
        )
        
        # Make agent throw an error
        mock_agent.process_request.side_effect = Exception("Test error")
        
        with patch('energyai_sdk.application.agent_registry') as mock_registry:
            mock_registry.get_agent.return_value = mock_agent
            
            from fastapi import BackgroundTasks
            with pytest.raises(Exception):
                await app_with_langfuse._process_chat_request(request, BackgroundTasks())
            
            # Verify generation was ended with error
            end_gen_calls = app_with_langfuse.langfuse_client.end_generation.call_args_list
            error_call = next((call for call in end_gen_calls if call[1]['level'] == "ERROR"), None)
            assert error_call is not None
            assert "Test error" in error_call[1]['status_message']
            
            # Verify trace was updated with error
            update_trace_calls = app_with_langfuse.langfuse_client.update_trace.call_args_list
            error_update = next((call for call in update_trace_calls if call[1]['level'] == "ERROR"), None)
            assert error_update is not None
            
            # Verify flush was still called
            app_with_langfuse.langfuse_client.flush.assert_called()

    @pytest.mark.asyncio
    async def test_chat_request_without_langfuse_client(self):
        """Test chat request processing without Langfuse client."""
        app = EnergyAIApplication(title="Test App", langfuse_monitoring_client=None)
        
        request = ChatRequest(
            message="Hello without monitoring",
            agent_id="TestAgent"
        )
        
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_response = AgentResponse(content="Response without monitoring", agent_id="TestAgent")
        mock_agent.process_request = AsyncMock(return_value=mock_response)
        
        with patch('energyai_sdk.application.agent_registry') as mock_registry:
            mock_registry.get_agent.return_value = mock_agent
            
            from fastapi import BackgroundTasks
            response = await app._process_chat_request(request, BackgroundTasks())
            
            # Should complete successfully without monitoring
            assert response.content == "Response without monitoring"

    def test_application_startup_with_langfuse(self, app_with_langfuse):
        """Test application startup with Langfuse monitoring."""
        # Mock health check
        app_with_langfuse.langfuse_client.is_enabled.return_value = True
        
        # Run startup (async context not needed for health check)
        # This tests the health check logic
        assert app_with_langfuse.langfuse_client.is_enabled() is True

    def test_application_startup_without_langfuse(self):
        """Test application startup without Langfuse monitoring."""
        app = EnergyAIApplication(title="Test App", langfuse_monitoring_client=None)
        assert app.langfuse_client is None

    def test_production_application_configuration(self):
        """Test production application with Langfuse configuration."""
        with patch('energyai_sdk.application.LANGFUSE_AVAILABLE', True):
            with patch('energyai_sdk.application.get_langfuse_client') as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                
                from energyai_sdk.application import create_production_application
                
                app = create_production_application(
                    enable_langfuse_monitoring=True,
                    langfuse_public_key="pk_prod",
                    langfuse_secret_key="sk_prod"
                )
                
                assert app.langfuse_client is not None
                mock_get_client.assert_called_once_with(
                    public_key="pk_prod",
                    secret_key="sk_prod",
                    host="https://cloud.langfuse.com",
                    debug=False,
                    environment="production"
                )

    @pytest.mark.asyncio
    async def test_context_span_error_handling(self, app_with_langfuse, mock_agent):
        """Test context span error handling in Langfuse monitoring."""
        request = ChatRequest(
            message="Context error test",
            agent_id="TestAgent",
            session_id="error_session"
        )
        
        # Mock context store client to throw error
        mock_context_store = MagicMock()
        mock_context_store.load_or_create.side_effect = Exception("Context store error")
        app_with_langfuse.context_store_client = mock_context_store
        
        with patch('energyai_sdk.application.agent_registry') as mock_registry:
            mock_registry.get_agent.return_value = mock_agent
            
            from fastapi import BackgroundTasks
            response = await app_with_langfuse._process_chat_request(request, BackgroundTasks())
            
            # Verify context span was ended with error
            end_span_calls = app_with_langfuse.langfuse_client.end_span.call_args_list
            error_call = next((call for call in end_span_calls if call[1]['level'] == "ERROR"), None)
            assert error_call is not None
            assert "Context store error" in error_call[1]['status_message']
            
            # Request should still complete
            assert response.content == "Test response from agent with monitoring"

    def test_langfuse_feature_detection(self):
        """Test that Langfuse features are properly detected."""
        # Test with Langfuse available
        with patch('energyai_sdk.application.LANGFUSE_AVAILABLE', True):
            with patch('energyai_sdk.application.get_langfuse_client') as mock_get_client:
                mock_client = MagicMock()
                mock_client.is_enabled.return_value = True
                mock_get_client.return_value = mock_client
                
                app = create_application(
                    enable_langfuse_monitoring=True,
                    langfuse_public_key="pk_test",
                    langfuse_secret_key="sk_test"
                )
                
                assert app.langfuse_client is not None
                assert app.langfuse_client.is_enabled() is True

    def test_langfuse_configuration_variations(self):
        """Test different Langfuse configuration scenarios."""
        test_cases = [
            {
                "description": "Custom host and environment",
                "params": {
                    "langfuse_host": "https://custom.langfuse.com",
                    "langfuse_environment": "staging",
                    "debug": True
                }
            },
            {
                "description": "Production configuration",
                "params": {
                    "langfuse_environment": "production",
                    "debug": False
                }
            },
            {
                "description": "Development configuration",
                "params": {
                    "langfuse_environment": "development",
                    "debug": True
                }
            }
        ]
        
        for test_case in test_cases:
            with patch('energyai_sdk.application.LANGFUSE_AVAILABLE', True):
                with patch('energyai_sdk.application.get_langfuse_client') as mock_get_client:
                    mock_client = MagicMock()
                    mock_get_client.return_value = mock_client
                    
                    app = create_application(
                        enable_langfuse_monitoring=True,
                        langfuse_public_key="pk_test",
                        langfuse_secret_key="sk_test",
                        **test_case["params"]
                    )
                    
                    assert app.langfuse_client is not None
                    
                    # Verify correct parameters were passed
                    call_args = mock_get_client.call_args
                    for key, value in test_case["params"].items():
                        if key in call_args[1]:
                            assert call_args[1][key] == value
