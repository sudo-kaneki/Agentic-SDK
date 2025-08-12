# tests/test_core.py
"""
Test core SDK components including decorators, registry, and telemetry.
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from energyai_sdk import (
    AgentRequest,
    AgentResponse,
    PromptTemplate,
    SkillDefinition,
    ToolDefinition,
    agent,
    agent_registry,
    monitor,
    planner,
    prompt,
    skill,
    telemetry_manager,
    tool,
)


class TestDecorators:
    """Test all SDK decorators."""

    def test_tool_decorator_basic(self, initialized_sdk):
        """Test basic tool decorator functionality."""

        @tool(name="test_add", description="Add two numbers")
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

        # Check if tool is registered
        assert "test_add" in agent_registry.tools
        tool_def = agent_registry.tools["test_add"]

        assert isinstance(tool_def, ToolDefinition)
        assert tool_def.name == "test_add"
        assert tool_def.description == "Add two numbers"
        assert tool_def.function == add_numbers

        # Check function still works
        result = add_numbers(5, 3)
        assert result == 8

        # Check metadata
        assert hasattr(add_numbers, "_tool_definition")
        assert hasattr(add_numbers, "_is_energyai_tool")

    def test_tool_decorator_with_parameters(self, initialized_sdk):
        """Test tool decorator with custom parameters."""

        @tool(
            name="custom_calculator",
            description="Custom calculator with operations",
            parameters={
                "operation": {"type": "string", "required": True},
                "x": {"type": "float", "required": True},
                "y": {"type": "float", "required": False, "default": 0},
            },
        )
        def calculator(operation: str, x: float, y: float = 0) -> dict[str, Any]:
            """Perform calculations."""
            operations = {"add": x + y, "multiply": x * y, "power": x**y}
            return {"result": operations.get(operation, 0)}

        tool_def = agent_registry.tools["custom_calculator"]
        assert tool_def.parameters["operation"]["required"] is True
        assert tool_def.parameters["y"]["default"] == 0

    def test_prompt_decorator(self, initialized_sdk):
        """Test prompt decorator functionality."""

        @prompt(
            name="analysis_prompt",
            template="Analyze the following data: {{$data}}\nContext: {{$context}}",
            parameters=["data", "context"],
        )
        def analysis_prompt():
            """Analysis prompt template."""
            pass

        # Check registration
        assert "analysis_prompt" in agent_registry.prompts
        prompt_def = agent_registry.prompts["analysis_prompt"]

        assert isinstance(prompt_def, PromptTemplate)
        assert prompt_def.name == "analysis_prompt"
        assert "{{$data}}" in prompt_def.template
        assert "data" in prompt_def.parameters
        assert "context" in prompt_def.parameters

        # Check metadata
        assert hasattr(analysis_prompt, "_prompt_definition")
        assert hasattr(analysis_prompt, "_is_energyai_prompt")

    def test_skill_decorator(self, initialized_sdk):
        """Test skill decorator functionality."""

        @skill(name="MathSkill", description="Mathematical operations")
        class MathSkill:
            """Mathematical operations skill."""

            @tool(name="square")
            def square(self, x: float) -> float:
                """Calculate square of a number."""
                return x * x

            @tool(name="cube")
            def cube(self, x: float) -> float:
                """Calculate cube of a number."""
                return x * x * x

        # Check registration
        assert "MathSkill" in agent_registry.skills
        skill_def = agent_registry.skills["MathSkill"]

        assert isinstance(skill_def, SkillDefinition)
        assert skill_def.name == "MathSkill"
        assert skill_def.description == "Mathematical operations"

        # Check metadata
        assert hasattr(MathSkill, "_skill_definition")
        assert hasattr(MathSkill, "_is_energyai_skill")

    def test_planner_decorator(self, initialized_sdk):
        """Test planner decorator functionality."""

        @planner(
            name="TestPlanner",
            description="Test planning functionality",
            max_steps=10,
            enable_loops=True,
        )
        async def test_planner(
            objective: str, constraints: list[str] = None
        ) -> list[dict[str, Any]]:
            """Test planner function."""
            return [
                {"step": 1, "action": "analyze", "description": f"Analyze {objective}"},
                {"step": 2, "action": "plan", "description": "Create plan"},
                {"step": 3, "action": "execute", "description": "Execute plan"},
            ]

        # Check registration
        assert "TestPlanner" in agent_registry.planners
        planner_config = agent_registry.planners["TestPlanner"]

        assert planner_config["name"] == "TestPlanner"
        assert planner_config["max_steps"] == 10
        assert planner_config["enable_loops"] is True

        # Check metadata
        assert hasattr(test_planner, "_planner_config")
        assert hasattr(test_planner, "_is_energyai_planner")

    def test_agent_decorator_function(self, initialized_sdk):
        """Test agent decorator on functions."""

        @agent(
            name="TestFunctionAgent",
            description="Test function-based agent",
            system_prompt="You are a test agent",
            tools=["test_tool"],
            skills=["TestSkill"],
        )
        async def test_agent(query: str, context: dict[str, Any] = None) -> str:
            """Test agent function."""
            return f"Processed: {query}"

        # Check registration
        assert "TestFunctionAgent" in agent_registry.agents
        registered_agent = agent_registry.agents["TestFunctionAgent"]

        assert registered_agent == test_agent
        assert hasattr(test_agent, "_agent_config")
        assert test_agent._agent_config["name"] == "TestFunctionAgent"

    def test_monitor_decorator_sync(self):
        """Test monitor decorator on synchronous functions."""

        @monitor("test_sync_operation")
        def sync_function(x: int, y: int) -> int:
            """Test synchronous function."""
            time.sleep(0.01)  # Small delay to test timing
            return x + y

        result = sync_function(5, 3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_monitor_decorator_async(self):
        """Test monitor decorator on asynchronous functions."""
        import asyncio

        @monitor("test_async_operation")
        async def async_function(x: int, y: int) -> int:
            """Test asynchronous function."""
            await asyncio.sleep(0.01)
            return x * y

        result = await async_function(4, 5)
        assert result == 20


class TestAgentRegistry:
    """Test agent registry functionality."""

    def test_register_and_retrieve_agent(self, initialized_sdk, sample_agent):
        """Test agent registration and retrieval."""

        # Agent should be auto-registered
        assert sample_agent.agent_name in agent_registry.agents

        # Test retrieval
        retrieved_agent = agent_registry.get_agent(sample_agent.agent_name)
        assert retrieved_agent == sample_agent

        # Test list agents
        agent_list = agent_registry.list_agents()
        assert sample_agent.agent_name in agent_list

    def test_register_tool(self, initialized_sdk):
        """Test tool registration."""

        def dummy_tool(x: int) -> int:
            return x * 2

        tool_def = ToolDefinition(
            name="dummy_tool",
            description="Dummy tool for testing",
            function=dummy_tool,
            parameters={"x": {"type": "int", "required": True}},
        )

        agent_registry.register_tool(tool_def)

        assert "dummy_tool" in agent_registry.tools
        assert agent_registry.tools["dummy_tool"] == tool_def

    def test_register_prompt(self, initialized_sdk):
        """Test prompt registration."""

        prompt_def = PromptTemplate(
            name="test_prompt",
            template="Hello {{$name}}",
            parameters=["name"],
            description="Test greeting prompt",
        )

        agent_registry.register_prompt(prompt_def)

        assert "test_prompt" in agent_registry.prompts
        assert agent_registry.prompts["test_prompt"] == prompt_def

    def test_register_skill(self, initialized_sdk):
        """Test skill registration."""

        skill_def = SkillDefinition(
            name="TestSkill", description="Test skill", functions=[], metadata={"version": "1.0"}
        )

        agent_registry.register_skill(skill_def)

        assert "TestSkill" in agent_registry.skills
        assert agent_registry.skills["TestSkill"] == skill_def

    def test_get_capabilities(self, initialized_sdk, sample_agent, sample_tool):
        """Test registry capabilities summary."""

        capabilities = agent_registry.get_capabilities()

        assert "agents" in capabilities
        assert "tools" in capabilities
        assert "prompts" in capabilities
        assert "skills" in capabilities
        assert "planners" in capabilities

        assert sample_agent.agent_name in capabilities["agents"]
        assert "test_calculator" in capabilities["tools"]

    def test_registry_cleanup(self, initialized_sdk):
        """Test that registry properly cleans up between tests."""

        # This test relies on the clean_registry fixture
        assert len(agent_registry.agents) >= 0  # May have test agents
        assert len(agent_registry.tools) >= 0  # May have test tools

        # Add something and verify it's there
        @tool(name="temp_tool")
        def temp_tool():
            return "temp"

        assert "temp_tool" in agent_registry.tools

        # The clean_registry fixture will clean this up after the test


class TestTelemetryManager:
    """Test telemetry manager functionality."""

    def test_telemetry_manager_initialization(self):
        """Test telemetry manager basic initialization."""

        assert telemetry_manager is not None
        assert hasattr(telemetry_manager, "azure_tracer")
        assert hasattr(telemetry_manager, "langfuse_client")
        assert hasattr(telemetry_manager, "active_traces")

    @patch("energyai_sdk.AZURE_MONITOR_AVAILABLE", True)
    @patch("energyai_sdk.AzureMonitorTraceExporter")
    @patch("energyai_sdk.trace")
    def test_configure_azure_monitor(self, mock_trace, mock_exporter):
        """Test Azure Monitor configuration."""

        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        telemetry_manager.configure_azure_monitor("InstrumentationKey=test-key", "test-service")

        assert telemetry_manager.azure_tracer == mock_tracer
        mock_exporter.assert_called_once()

    @patch("energyai_sdk.LANGFUSE_AVAILABLE", True)
    @patch("energyai_sdk.Langfuse")
    def test_configure_langfuse(self, mock_langfuse):
        """Test Langfuse configuration."""

        mock_client = Mock()
        mock_langfuse.return_value = mock_client

        telemetry_manager.configure_langfuse(
            "pk_test_key", "sk_test_key", "https://test.langfuse.com", "test"
        )

        assert telemetry_manager.langfuse_client == mock_client
        mock_langfuse.assert_called_once_with(
            public_key="pk_test_key",
            secret_key="sk_test_key",
            host="https://test.langfuse.com",
            environment="test",
        )

    def test_trace_operation_context_manager(self):
        """Test trace operation context manager."""

        with telemetry_manager.trace_operation("test_op", {"key": "value"}) as trace_id:
            assert trace_id is not None
            assert trace_id in telemetry_manager.active_traces

            trace_data = telemetry_manager.active_traces[trace_id]
            assert trace_data["operation_name"] == "test_op"
            assert trace_data["metadata"]["key"] == "value"

        # After context exit, trace should be cleaned up
        assert trace_id not in telemetry_manager.active_traces

    def test_trace_operation_with_exception(self):
        """Test trace operation handling exceptions."""

        try:
            with telemetry_manager.trace_operation("test_error") as trace_id:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Trace should still be cleaned up
        assert trace_id not in telemetry_manager.active_traces


class TestDataStructures:
    """Test core data structures."""

    def test_agent_request_creation(self):
        """Test AgentRequest creation and validation."""

        request = AgentRequest(
            message="Test message",
            agent_id="TestAgent",
            session_id="test_session",
            user_id="test_user",
            metadata={"priority": "high"},
        )

        assert request.message == "Test message"
        assert request.agent_id == "TestAgent"
        assert request.session_id == "test_session"
        assert request.user_id == "test_user"
        assert request.metadata["priority"] == "high"
        assert request.request_id is not None
        assert request.timestamp is not None

    def test_agent_request_defaults(self):
        """Test AgentRequest with default values."""

        request = AgentRequest(message="Test message", agent_id="TestAgent")

        assert request.session_id is None
        assert request.user_id is None
        assert isinstance(request.metadata, dict)
        assert len(request.metadata) == 0
        assert request.request_id is not None

    def test_agent_response_creation(self):
        """Test AgentResponse creation and validation."""

        response = AgentResponse(
            content="Test response",
            agent_id="TestAgent",
            session_id="test_session",
            execution_time_ms=150,
            metadata={"model": "gpt-4o"},
        )

        assert response.content == "Test response"
        assert response.agent_id == "TestAgent"
        assert response.session_id == "test_session"
        assert response.execution_time_ms == 150
        assert response.metadata["model"] == "gpt-4o"
        assert response.error is None
        assert response.timestamp is not None

    def test_agent_response_with_error(self):
        """Test AgentResponse with error information."""

        response = AgentResponse(
            content="Error occurred",
            agent_id="TestAgent",
            error="Test error message",
            metadata={"error_type": "ValueError"},
        )

        assert response.error == "Test error message"
        assert response.metadata["error_type"] == "ValueError"

    def test_tool_definition_creation(self):
        """Test ToolDefinition creation."""

        def test_func(x: int, y: int = 5) -> int:
            return x + y

        tool_def = ToolDefinition(
            name="test_func",
            description="Test function",
            function=test_func,
            parameters={"x": {"type": "int", "required": True}},
            return_type=int,
            is_async=False,
        )

        assert tool_def.name == "test_func"
        assert tool_def.description == "Test function"
        assert tool_def.function == test_func
        assert tool_def.return_type == int
        assert tool_def.is_async is False

    def test_prompt_template_creation(self):
        """Test PromptTemplate creation."""

        prompt_def = PromptTemplate(
            name="greeting",
            template="Hello {{$name}}, welcome to {{$place}}!",
            parameters=["name", "place"],
            description="Greeting prompt",
            execution_settings={"temperature": 0.7},
        )

        assert prompt_def.name == "greeting"
        assert "{{$name}}" in prompt_def.template
        assert "name" in prompt_def.parameters
        assert "place" in prompt_def.parameters
        assert prompt_def.execution_settings["temperature"] == 0.7

    def test_skill_definition_creation(self):
        """Test SkillDefinition creation."""

        def func1():
            return "result1"

        def func2():
            return "result2"

        skill_def = SkillDefinition(
            name="TestSkill",
            description="Test skill with functions",
            functions=[func1, func2],
            metadata={"version": "1.0", "author": "test"},
        )

        assert skill_def.name == "TestSkill"
        assert skill_def.description == "Test skill with functions"
        assert len(skill_def.functions) == 2
        assert func1 in skill_def.functions
        assert func2 in skill_def.functions
        assert skill_def.metadata["version"] == "1.0"
        assert skill_def.metadata["author"] == "test"


class TestInitialization:
    """Test SDK initialization."""

    @patch("energyai_sdk.configure_telemetry")
    def test_initialize_sdk_basic(self, mock_configure_telemetry):
        """Test basic SDK initialization."""

        from energyai_sdk import initialize_sdk

        result = initialize_sdk(log_level="INFO")

        assert isinstance(result, dict)
        assert "version" in result
        assert "telemetry_configured" in result
        assert "capabilities" in result

    @patch("energyai_sdk.configure_telemetry")
    def test_initialize_sdk_with_telemetry(self, mock_configure_telemetry):
        """Test SDK initialization with telemetry configuration."""

        from energyai_sdk import initialize_sdk

        result = initialize_sdk(
            azure_monitor_connection_string="InstrumentationKey=test",
            langfuse_public_key="pk_test",
            langfuse_secret_key="sk_test",
            log_level="DEBUG",
        )

        mock_configure_telemetry.assert_called_once_with(
            "InstrumentationKey=test", "pk_test", "sk_test"
        )

        assert result["telemetry_configured"]["azure_monitor"] is True
        assert result["telemetry_configured"]["langfuse"] is True

    def test_initialize_sdk_logging_configuration(self):
        """Test that SDK initialization properly configures logging."""

        import logging

        from energyai_sdk import initialize_sdk

        initialize_sdk(log_level="WARNING")

        logger = logging.getLogger("energyai_sdk")
        assert logger.level <= logging.WARNING


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for core components working together."""

    def test_tool_skill_agent_integration(self, initialized_sdk, mock_azure_openai_config):
        """Test integration of tools, skills, and agents."""

        # Create tools
        @tool(name="multiply")
        def multiply(x: float, y: float) -> float:
            return x * y

        @tool(name="divide")
        def divide(x: float, y: float) -> float:
            return x / y if y != 0 else 0

        # Create skill
        @skill(name="Calculator")
        class Calculator:
            @tool(name="add_skill")
            def add(self, x: float, y: float) -> float:
                return x + y

        # Create agent with tools and skills using decorator
        @agent(
            name="MathAgent",
            description="Mathematical operations agent",
            system_prompt="You perform mathematical operations.",
            tools=["multiply", "divide"],
            skills=["Calculator"],
        )
        class MathAgent:
            pass

        # Bootstrap the agent
        from energyai_sdk.agents import bootstrap_agents

        agents = bootstrap_agents(azure_openai_config=mock_azure_openai_config)
        agent = agents.get("MathAgent")

        # Verify everything is connected
        if agent:
            assert agent.agent_name == "MathAgent"

        # Verify registry state
        assert "MathAgent" in agent_registry.agents
        assert "multiply" in agent_registry.tools
        assert "divide" in agent_registry.tools
        assert "Calculator" in agent_registry.skills

    @pytest.mark.asyncio
    async def test_monitor_telemetry_integration(self):
        """Test integration of monitoring and telemetry."""
        import asyncio

        call_count = 0

        @monitor("test_integration_op")
        async def monitored_operation(data: str) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return f"Processed: {data}"

        result = await monitored_operation("test_data")

        assert result == "Processed: test_data"
        assert call_count == 1

    def test_multiple_decorators_on_same_function(self, initialized_sdk):
        """Test multiple decorators on the same function."""

        @monitor("decorated_tool_operation")
        @tool(name="decorated_tool", description="Tool with monitoring")
        def decorated_tool(value: str) -> str:
            """Tool with both monitoring and tool decorators."""
            return f"Decorated: {value}"

        # Test function works
        result = decorated_tool("test")
        assert result == "Decorated: test"

        # Test both decorators applied
        assert hasattr(decorated_tool, "_tool_definition")
        assert "decorated_tool" in agent_registry.tools

        # Test tool can be called
        tool_def = agent_registry.tools["decorated_tool"]
        tool_result = tool_def.function("test_value")
        assert tool_result == "Decorated: test_value"


if __name__ == "__main__":
    pytest.main([__file__])
