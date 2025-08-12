# tests/conftest.py
"""
Pytest configuration and shared fixtures for EnergyAI SDK tests.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from energyai_sdk import (
    AgentRequest,
    AgentResponse,
    agent,
    agent_registry,
    initialize_sdk,
    skill,
    tool,
)
from energyai_sdk.agents import bootstrap_agents

# Try to import optional components
try:
    from energyai_sdk.middleware import (
        AgentMiddleware,
        AuthenticationMiddleware,
        CachingMiddleware,
        MiddlewareContext,
        MiddlewarePipeline,
        ValidationMiddleware,
    )

    MIDDLEWARE_AVAILABLE = True
except ImportError:
    MIDDLEWARE_AVAILABLE = False

try:
    from energyai_sdk.application import EnergyAIApplication, create_application

    APPLICATION_AVAILABLE = True
except ImportError:
    APPLICATION_AVAILABLE = False


# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring real API credentials"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in test_*.py files unless marked otherwise
        if "test_" in item.nodeid and not any(
            marker in item.keywords for marker in ["integration", "slow"]
        ):
            item.add_marker(pytest.mark.unit)


# ==============================================================================
# ASYNC SUPPORT
# ==============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# MOCK CONFIGURATIONS
# ==============================================================================


@pytest.fixture
def mock_azure_openai_config():
    """Mock Azure OpenAI configuration for testing."""
    return {
        "deployment_name": "test-gpt-4o",
        "endpoint": "https://test-endpoint.openai.azure.com/",
        "api_key": "test-api-key-12345",
        "api_version": "2024-02-01",
        "service_id": "test-service",
        "is_default": True,
    }


@pytest.fixture
def mock_openai_config():
    """Mock OpenAI configuration for testing."""
    return {
        "deployment_name": "gpt-4o",
        "api_key": "test-openai-key-12345",
        "base_url": None,
        "service_type": "openai",
        "is_default": True,
    }


@pytest.fixture
def mock_telemetry_config():
    """Mock telemetry configuration for testing."""
    return {
        "azure_monitor_connection_string": "InstrumentationKey=test-key-12345",
        "langfuse_public_key": "pk_lf_test_key",
        "langfuse_secret_key": "sk_lf_test_secret",
        "langfuse_host": "https://test.langfuse.com",
        "environment": "test",
    }


# ==============================================================================
# SDK FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the agent registry before each test."""
    # Store original state
    original_agents = agent_registry.agents.copy()
    original_tools = agent_registry.tools.copy()
    original_prompts = agent_registry.prompts.copy()
    original_skills = agent_registry.skills.copy()
    original_planners = agent_registry.planners.copy()

    # Clear registry
    agent_registry.agents.clear()
    agent_registry.tools.clear()
    agent_registry.prompts.clear()
    agent_registry.skills.clear()
    agent_registry.planners.clear()

    yield

    # Restore original state
    agent_registry.agents = original_agents
    agent_registry.tools = original_tools
    agent_registry.prompts = original_prompts
    agent_registry.skills = original_skills
    agent_registry.planners = original_planners


@pytest.fixture
def initialized_sdk():
    """Initialize SDK for testing."""
    initialize_sdk(log_level="DEBUG")
    yield
    # Cleanup is handled by clean_registry


# ==============================================================================
# TOOL AND SKILL FIXTURES
# ==============================================================================


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""

    @tool(name="test_calculator", description="Simple test calculator")
    def test_calculator(a: float, b: float, operation: str = "add") -> dict:
        """Test calculator tool."""
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else None,
        }

        result = operations.get(operation.lower())
        return {
            "result": result,
            "operation": operation,
            "operands": [a, b],
            "success": result is not None,
        }

    return test_calculator


@pytest.fixture
def sample_skill():
    """Create a sample skill for testing."""

    @skill(name="TestMath", description="Test mathematical operations")
    class TestMath:
        """Test skill for mathematical operations."""

        @tool(name="power")
        def power(self, base: float, exponent: float) -> dict:
            """Calculate power operation."""
            result = base**exponent
            return {"result": result, "base": base, "exponent": exponent}

        @tool(name="factorial")
        def factorial(self, n: int) -> dict:
            """Calculate factorial."""
            if n < 0:
                return {"error": "Factorial not defined for negative numbers"}

            result = 1
            for i in range(1, n + 1):
                result *= i

            return {"result": result, "input": n}

    return TestMath


# ==============================================================================
# AGENT FIXTURES
# ==============================================================================


@pytest.fixture
def mock_semantic_kernel():
    """Mock Semantic Kernel for testing."""
    with patch("energyai_sdk.agents.Kernel") as mock_kernel:
        mock_instance = Mock()
        mock_kernel.return_value = mock_instance

        # Mock service methods
        mock_service = Mock()
        mock_service.get_chat_message_content = AsyncMock()
        mock_instance.get_service.return_value = mock_service
        mock_instance.add_service = Mock()

        yield mock_instance


@pytest.fixture
def sample_agent(mock_azure_openai_config):
    """Create a sample agent using decorators."""

    @agent(
        name="TestAgent",
        description="Test agent for pytest",
        system_prompt="You are a helpful test assistant.",
    )
    class TestAgent:
        temperature = 0.7
        max_tokens = 1000

    # Bootstrap the agent to create actual instance
    agents = bootstrap_agents(azure_openai_config=mock_azure_openai_config)
    return agents.get("TestAgent")


@pytest.fixture
def sample_manager_agent(sample_agent, mock_azure_openai_config):
    """Create a sample manager agent for testing."""

    @agent(
        name="SubordinateAgent",
        description="Subordinate test agent",
        system_prompt="You are a subordinate test assistant.",
    )
    class SubordinateAgent:
        pass

    @agent(
        name="ManagerAgent",
        description="Manager test agent",
        system_prompt="You coordinate other agents.",
    )
    class ManagerAgent:
        subordinate_agents = ["TestAgent", "SubordinateAgent"]
        selection_strategy = "prompt"
        max_iterations = 3

    # Bootstrap agents
    agents = bootstrap_agents(azure_openai_config=mock_azure_openai_config)
    return agents.get("ManagerAgent")


# ==============================================================================
# REQUEST/RESPONSE FIXTURES
# ==============================================================================


@pytest.fixture
def sample_agent_request():
    """Create a sample agent request for testing."""
    return AgentRequest(
        message="Hello, this is a test message",
        agent_id="TestAgent",
        session_id="test_session_123",
        user_id="test_user",
        metadata={"test": True, "priority": "normal"},
    )


@pytest.fixture
def sample_agent_response():
    """Create a sample agent response for testing."""
    return AgentResponse(
        content="Hello! I'm a test response.",
        agent_id="TestAgent",
        session_id="test_session_123",
        execution_time_ms=150,
        metadata={"model_used": "gpt-4o", "tokens": 10},
    )


# ==============================================================================
# MIDDLEWARE FIXTURES
# ==============================================================================


@pytest.fixture
def sample_middleware_context(sample_agent_request):
    """Create a sample middleware context for testing."""
    return MiddlewareContext(request=sample_agent_request)


@pytest.fixture
def test_middleware():
    """Create a test middleware for testing."""
    if not MIDDLEWARE_AVAILABLE:
        pytest.skip("Middleware not available")

    class TestMiddleware(AgentMiddleware):
        def __init__(self, name="TestMiddleware"):
            super().__init__(name)
            self.called = False
            self.call_count = 0
            self.priority = 10

        async def process(self, context, next_middleware):
            self.called = True
            self.call_count += 1
            context.metadata["test_middleware_called"] = True
            await next_middleware(context)

    return TestMiddleware()


@pytest.fixture
def middleware_pipeline():
    """Create a middleware pipeline for testing."""
    if not MIDDLEWARE_AVAILABLE:
        pytest.skip("Middleware not available")

    pipeline = MiddlewarePipeline()

    # Add some basic middleware
    pipeline.add_preprocessing(ValidationMiddleware(max_message_length=1000))
    pipeline.add_preprocessing(AuthenticationMiddleware(required_auth=False))
    pipeline.add_postprocessing(CachingMiddleware(cache_ttl_seconds=60))

    return pipeline


# ==============================================================================
# APPLICATION FIXTURES
# ==============================================================================


@pytest.fixture
def test_application():
    """Create a test application for testing."""
    if not APPLICATION_AVAILABLE:
        pytest.skip("Application module not available")

    app = create_application(title="Test Application", debug=True, enable_default_middleware=False)
    return app


@pytest.fixture
def fastapi_test_client(test_application):
    """Create a FastAPI test client."""
    try:
        from fastapi.testclient import TestClient

        return TestClient(test_application.get_fastapi_app())
    except ImportError:
        pytest.skip("FastAPI not available for testing")


# ==============================================================================
# DATA FIXTURES
# ==============================================================================


@pytest.fixture
def sample_energy_data():
    """Sample energy data for testing calculations."""
    return {
        "solar_farm": {
            "capacity_mw": 100,
            "capital_cost": 150_000_000,
            "annual_generation_mwh": 250_000,
            "annual_operating_cost": 2_000_000,
            "capacity_factor": 0.285,
            "lifetime_years": 25,
        },
        "wind_farm": {
            "capacity_mw": 200,
            "capital_cost": 400_000_000,
            "annual_generation_mwh": 700_000,
            "annual_operating_cost": 8_000_000,
            "capacity_factor": 0.40,
            "lifetime_years": 25,
        },
        "battery_storage": {
            "capacity_mw": 50,
            "duration_hours": 4,
            "capital_cost": 75_000_000,
            "annual_revenue": 15_000_000,
            "annual_operating_cost": 1_500_000,
            "lifetime_years": 15,
        },
    }


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing."""
    return {
        "discount_rates": [0.06, 0.08, 0.10, 0.12],
        "electricity_prices": [40, 50, 60, 70, 80],  # $/MWh
        "inflation_rates": [0.02, 0.025, 0.03],
        "tax_rates": [0.21, 0.25, 0.30],
        "cash_flows_25_years": [8_000_000] * 25,  # $8M annually
        "degradation_rates": {"solar": 0.005, "wind": 0.002, "battery": 0.02},
    }


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "historical_prices": [35, 42, 38, 45, 52, 48, 55, 50, 58, 62, 59, 65],
        "demand_data": [1200, 1150, 1300, 1250, 1400, 1350, 1500, 1450],
        "capacity_factors_by_region": {
            "california": {"solar": 0.25, "wind": 0.32},
            "texas": {"solar": 0.28, "wind": 0.38},
            "northeast": {"solar": 0.18, "wind": 0.35},
        },
        "benchmark_prices": {
            "pjm": {"low": 30, "avg": 45, "high": 70},
            "caiso": {"low": 40, "avg": 60, "high": 90},
            "ercot": {"low": 25, "avg": 40, "high": 65},
        },
    }


# ==============================================================================
# MOCK RESPONSE FIXTURES
# ==============================================================================


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing."""

    class MockChatMessageContent:
        def __init__(self, content: str):
            self.content = content

    return MockChatMessageContent("This is a mock response from the AI model.")


@pytest.fixture
def mock_successful_api_calls(mock_openai_response):
    """Mock successful API calls to external services."""
    with (
        patch("energyai_sdk.agents.AzureAIInferenceChatCompletion") as mock_azure,
        patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion") as mock_openai,
    ):

        # Mock Azure OpenAI
        mock_azure_instance = Mock()
        mock_azure_instance.get_chat_message_content = AsyncMock(return_value=mock_openai_response)
        mock_azure.return_value = mock_azure_instance

        # Mock OpenAI
        mock_openai_instance = Mock()
        mock_openai_instance.get_chat_message_content = AsyncMock(return_value=mock_openai_response)
        mock_openai.return_value = mock_openai_instance

        yield {"azure": mock_azure_instance, "openai": mock_openai_instance}


# ==============================================================================
# PERFORMANCE FIXTURES
# ==============================================================================


@pytest.fixture
def performance_monitor():
    """Monitor for tracking test performance."""
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.measurements = {}

        def start(self, operation: str = "default"):
            self.start_time = time.time()
            return self

        def stop(self, operation: str = "default"):
            self.end_time = time.time()
            self.measurements[operation] = self.end_time - self.start_time
            return self.measurements[operation]

        def get_duration(self, operation: str = "default"):
            return self.measurements.get(operation, 0)

    return PerformanceMonitor()


# ==============================================================================
# ENVIRONMENT FIXTURES
# ==============================================================================


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    mock_env = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key-12345",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-gpt-4o",
        "LANGFUSE_PUBLIC_KEY": "pk_test_12345",
        "LANGFUSE_SECRET_KEY": "sk_test_12345",
        "ENERGYAI_APP_DEBUG": "true",
        "ENERGYAI_LOG_LEVEL": "DEBUG",
    }

    with patch.dict(os.environ, mock_env, clear=False):
        yield mock_env


# ==============================================================================
# UTILITY FIXTURES
# ==============================================================================


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file for testing."""
    config_content = """
application:
  title: "Test EnergyAI Platform"
  debug: true
  port: 8001

models:
  - deployment_name: "test-gpt-4o"
    model_type: "azure_openai"
    endpoint: "https://test.openai.azure.com/"
    api_key: "test-key-12345"
    is_default: true

security:
  enable_auth: false
  api_keys: []

telemetry:
  enable_azure_monitor: false
  enable_langfuse: false
"""

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def capture_logs():
    """Capture log messages during tests."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("energyai_sdk")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    yield log_capture

    logger.removeHandler(handler)


# ==============================================================================
# PARAMETRIZE HELPERS
# ==============================================================================


def pytest_generate_tests(metafunc):
    """Generate parameterized tests based on fixtures."""
    if "agent_type" in metafunc.fixturenames:
        metafunc.parametrize("agent_type", ["SemanticKernelAgent", "ManagerAgent"])

    if "model_config_type" in metafunc.fixturenames:
        metafunc.parametrize("model_config_type", ["azure_openai", "openai"])

    if "middleware_type" in metafunc.fixturenames:
        metafunc.parametrize(
            "middleware_type",
            ["AuthenticationMiddleware", "ValidationMiddleware", "CachingMiddleware"],
        )
