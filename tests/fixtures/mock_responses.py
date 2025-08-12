# tests/fixtures/mock_responses.py
"""
Mock responses and test data for EnergyAI SDK tests.
"""

import json
from datetime import datetime, timezone
from typing import Any

from energyai_sdk import AgentResponse


class MockOpenAIResponses:
    """Mock responses for OpenAI API calls."""

    @staticmethod
    def get_chat_completion_response(content: str = "This is a mock response") -> dict[str, Any]:
        """Get mock chat completion response."""
        return {
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    @staticmethod
    def get_streaming_response_chunks(content: str = "Streaming response") -> list[dict[str, Any]]:
        """Get mock streaming response chunks."""
        words = content.split()
        chunks = []

        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-mock{i}",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + " " if i < len(words) - 1 else word},
                        "finish_reason": None if i < len(words) - 1 else "stop",
                    }
                ],
            }
            chunks.append(chunk)

        return chunks

    @staticmethod
    def get_error_response(error_message: str = "Mock API error") -> dict[str, Any]:
        """Get mock error response."""
        return {
            "error": {
                "message": error_message,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }


class MockSemanticKernelResponses:
    """Mock responses for Semantic Kernel components."""

    class MockChatMessageContent:
        """Mock ChatMessageContent object."""

        def __init__(self, content: str):
            self.content = content
            self.role = "assistant"
            self.metadata = {}

    class MockKernel:
        """Mock Semantic Kernel object."""

        def __init__(self):
            self.services = {}
            self.functions = {}

        def add_service(self, service):
            """Mock add_service method."""
            self.services[service.service_id] = service
            return service

        def get_service(self, service_id: str):
            """Mock get_service method."""
            return self.services.get(service_id, MockSemanticKernelResponses.MockChatService())

        def add_function(
            self, plugin_name: str, function_name: str, function, description: str = ""
        ):
            """Mock add_function method."""
            self.functions[f"{plugin_name}.{function_name}"] = function
            return function

    class MockChatService:
        """Mock chat completion service."""

        def __init__(self, response_content: str = "Mock SK response"):
            self.response_content = response_content
            self.service_id = "mock_service"

        async def get_chat_message_content(self, chat_history=None, settings=None, kernel=None):
            """Mock get_chat_message_content method."""
            return MockSemanticKernelResponses.MockChatMessageContent(self.response_content)


class MockEnergyData:
    """Mock energy-related data for testing."""

    @staticmethod
    def get_solar_project_data() -> dict[str, Any]:
        """Get mock solar project data."""
        return {
            "project_name": "Mock Solar Farm",
            "capacity_mw": 100,
            "capacity_factor": 0.25,
            "capex_million": 150,
            "annual_opex_million": 2,
            "location": "California",
            "technology": "crystalline_silicon",
            "lifetime_years": 25,
            "degradation_rate": 0.005,
            "discount_rate": 0.08,
            "electricity_price_mwh": 60,
        }

    @staticmethod
    def get_wind_project_data() -> dict[str, Any]:
        """Get mock wind project data."""
        return {
            "project_name": "Mock Wind Farm",
            "capacity_mw": 200,
            "capacity_factor": 0.40,
            "capex_million": 400,
            "annual_opex_million": 8,
            "location": "Texas",
            "technology": "onshore_wind",
            "turbine_capacity_mw": 2.5,
            "number_of_turbines": 80,
            "hub_height_m": 100,
            "lifetime_years": 25,
        }

    @staticmethod
    def get_battery_project_data() -> dict[str, Any]:
        """Get mock battery storage project data."""
        return {
            "project_name": "Mock Battery Storage",
            "capacity_mw": 50,
            "duration_hours": 4,
            "capex_million": 75,
            "annual_opex_million": 1.5,
            "location": "Arizona",
            "technology": "lithium_ion",
            "efficiency": 0.85,
            "lifetime_years": 15,
            "cycles_per_year": 365,
        }

    @staticmethod
    def get_market_data() -> dict[str, Any]:
        """Get mock energy market data."""
        return {
            "region": "CAISO",
            "historical_prices_mwh": [45, 48, 52, 47, 55, 50, 58, 53, 60, 55, 62, 58],
            "demand_gw": [25, 28, 32, 30, 35, 33, 38, 36, 40, 37, 42, 39],
            "renewable_penetration": 0.35,
            "peak_demand_gw": 50,
            "average_price_mwh": 53,
            "price_volatility": 0.15,
            "carbon_price_ton": 25,
        }

    @staticmethod
    def get_financial_assumptions() -> dict[str, Any]:
        """Get mock financial assumptions."""
        return {
            "discount_rate": 0.08,
            "inflation_rate": 0.025,
            "tax_rate": 0.25,
            "debt_ratio": 0.70,
            "debt_interest_rate": 0.05,
            "project_lifetime_years": 25,
            "currency": "USD",
            "base_year": 2024,
        }


class MockAgentResponses:
    """Mock agent responses for testing."""

    @staticmethod
    def get_successful_response(
        agent_id: str = "MockAgent", content: str = "Mock successful response"
    ) -> AgentResponse:
        """Get mock successful agent response."""
        return AgentResponse(
            content=content,
            agent_id=agent_id,
            session_id="mock_session",
            execution_time_ms=150,
            metadata={
                "model_used": "gpt-4o",
                "tokens_used": 50,
                "tools_called": [],
                "success": True,
            },
            timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def get_error_response(
        agent_id: str = "MockAgent", error_message: str = "Mock error"
    ) -> AgentResponse:
        """Get mock error agent response."""
        return AgentResponse(
            content=f"Error occurred: {error_message}",
            agent_id=agent_id,
            session_id="mock_session",
            execution_time_ms=50,
            error=error_message,
            metadata={"error_type": "MockError", "error_handled": True, "success": False},
            timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def get_tool_usage_response(
        agent_id: str = "MockAgent", tools_used: list[str] = None
    ) -> AgentResponse:
        """Get mock response showing tool usage."""
        tools_used = tools_used or ["mock_calculator", "mock_analyzer"]

        return AgentResponse(
            content=f"Analysis completed using tools: {', '.join(tools_used)}",
            agent_id=agent_id,
            session_id="mock_session",
            execution_time_ms=300,
            metadata={
                "model_used": "gpt-4o",
                "tools_called": tools_used,
                "tool_results": {
                    tool: {"status": "success", "result": f"Mock result from {tool}"}
                    for tool in tools_used
                },
                "success": True,
            },
            timestamp=datetime.now(timezone.utc),
        )


class MockConfigurationData:
    """Mock configuration data for testing."""

    @staticmethod
    def get_test_config() -> dict[str, Any]:
        """Get mock test configuration."""
        return {
            "application": {
                "title": "Test EnergyAI Platform",
                "version": "1.0.0-test",
                "description": "Test environment configuration",
                "debug": True,
                "host": "127.0.0.1",
                "port": 8000,
            },
            "models": [
                {
                    "deployment_name": "test-gpt-4o",
                    "model_type": "azure_openai",
                    "endpoint": "https://test.openai.azure.com/",
                    "api_key": "test-api-key-12345",
                    "api_version": "2024-02-01",
                    "is_default": True,
                }
            ],
            "telemetry": {
                "enable_azure_monitor": False,
                "enable_langfuse": False,
                "sample_rate": 1.0,
            },
            "security": {
                "enable_auth": False,
                "enable_rate_limiting": False,
                "api_keys": [],
                "max_requests_per_minute": 1000,
            },
            "features": {
                "enable_caching": True,
                "cache_ttl_seconds": 300,
                "enable_cors": True,
                "max_message_length": 10000,
            },
        }

    @staticmethod
    def get_production_config() -> dict[str, Any]:
        """Get mock production configuration."""
        return {
            "application": {
                "title": "EnergyAI Production Platform",
                "version": "1.0.0",
                "description": "Production environment configuration",
                "debug": False,
                "host": "0.0.0.0",
                "port": 8080,
            },
            "models": [
                {
                    "deployment_name": "prod-gpt-4o",
                    "model_type": "azure_openai",
                    "endpoint": "${AZURE_OPENAI_ENDPOINT}",
                    "api_key": "${AZURE_OPENAI_API_KEY}",
                    "is_default": True,
                }
            ],
            "telemetry": {
                "enable_azure_monitor": True,
                "azure_monitor_connection_string": "${AZURE_MONITOR_CONNECTION_STRING}",
                "enable_langfuse": True,
                "langfuse_public_key": "${LANGFUSE_PUBLIC_KEY}",
                "langfuse_secret_key": "${LANGFUSE_SECRET_KEY}",
                "sample_rate": 1.0,
            },
            "security": {
                "enable_auth": True,
                "enable_rate_limiting": True,
                "api_keys": ["${API_KEY_1}", "${API_KEY_2}"],
                "max_requests_per_minute": 100,
                "max_requests_per_hour": 5000,
            },
            "features": {
                "enable_caching": True,
                "cache_ttl_seconds": 600,
                "enable_cors": False,
                "max_message_length": 5000,
            },
        }


class MockTelemetryData:
    """Mock telemetry data for testing."""

    @staticmethod
    def get_trace_data() -> dict[str, Any]:
        """Get mock trace data."""
        return {
            "trace_id": "mock-trace-12345",
            "span_id": "mock-span-67890",
            "operation_name": "mock_operation",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc)).isoformat(),
            "duration_ms": 150,
            "status": "success",
            "attributes": {
                "agent_id": "MockAgent",
                "user_id": "test_user",
                "session_id": "test_session",
                "model_used": "gpt-4o",
            },
            "events": [
                {
                    "name": "agent_processing_started",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attributes": {"phase": "preprocessing"},
                },
                {
                    "name": "agent_processing_completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attributes": {"phase": "postprocessing"},
                },
            ],
        }

    @staticmethod
    def get_metric_data() -> dict[str, Any]:
        """Get mock metric data."""
        return {
            "metric_name": "agent_response_time",
            "value": 150,
            "unit": "milliseconds",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dimensions": {"agent_id": "MockAgent", "model": "gpt-4o", "success": True},
        }


class MockWebResponses:
    """Mock web/HTTP responses for testing."""

    @staticmethod
    def get_health_check_response() -> dict[str, Any]:
        """Get mock health check response."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0-test",
            "agents_count": 3,
            "uptime_seconds": 3600,
            "telemetry_configured": False,
            "components": {
                "registry": "healthy",
                "middleware": "healthy",
                "telemetry": "not_configured",
            },
        }

    @staticmethod
    def get_chat_request() -> dict[str, Any]:
        """Get mock chat request."""
        return {
            "message": "Calculate LCOE for a 100MW solar farm",
            "agent_id": "EnergyAnalyst",
            "session_id": "test_session_123",
            "user_id": "test_user",
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False,
            "metadata": {"priority": "normal", "source": "web_interface"},
        }

    @staticmethod
    def get_chat_response() -> dict[str, Any]:
        """Get mock chat response."""
        return {
            "content": "Based on the parameters provided, the LCOE for the 100MW solar farm is approximately $45/MWh.",
            "agent_id": "EnergyAnalyst",
            "session_id": "test_session_123",
            "execution_time_ms": 250,
            "model_used": "gpt-4o",
            "metadata": {
                "tools_used": ["lcoe_calculator"],
                "confidence": 0.95,
                "calculation_verified": True,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }

    @staticmethod
    def get_agent_info() -> dict[str, Any]:
        """Get mock agent info response."""
        return {
            "agent_id": "EnergyAnalyst",
            "name": "Energy Analysis Specialist",
            "description": "Expert in renewable energy financial analysis",
            "type": "SemanticKernelAgent",
            "capabilities": ["chat", "calculation", "analysis"],
            "models": ["gpt-4o"],
            "tools": ["lcoe_calculator", "capacity_factor_analyzer"],
            "skills": ["EnergyEconomics", "TechnicalPerformance"],
            "is_available": True,
            "metadata": {
                "version": "1.0.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "specialization": "renewable_energy_finance",
            },
        }


class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def generate_time_series_data(
        length: int = 12, base_value: float = 50, volatility: float = 0.1
    ) -> list[float]:
        """Generate mock time series data."""
        import random

        data = []
        current_value = base_value

        for _ in range(length):
            # Add some random walk behavior
            change = random.uniform(-volatility, volatility) * current_value
            current_value = max(0, current_value + change)
            data.append(round(current_value, 2))

        return data

    @staticmethod
    def generate_agent_metadata(agent_id: str) -> dict[str, Any]:
        """Generate mock agent metadata."""
        return {
            "agent_id": agent_id,
            "creation_time": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "capabilities": ["chat", "analysis"],
            "configuration": {"temperature": 0.7, "max_tokens": 1000, "timeout_seconds": 30},
            "statistics": {
                "requests_processed": 100,
                "average_response_time_ms": 200,
                "success_rate": 0.98,
                "last_request_time": datetime.now(timezone.utc).isoformat(),
            },
        }

    @staticmethod
    def generate_error_scenarios() -> list[dict[str, Any]]:
        """Generate various error scenarios for testing."""
        return [
            {
                "scenario": "api_key_invalid",
                "error_type": "AuthenticationError",
                "error_message": "Invalid API key provided",
                "http_status": 401,
                "expected_behavior": "return_auth_error",
            },
            {
                "scenario": "model_not_found",
                "error_type": "ModelNotFoundError",
                "error_message": "Specified model deployment not found",
                "http_status": 404,
                "expected_behavior": "fallback_to_default_model",
            },
            {
                "scenario": "rate_limit_exceeded",
                "error_type": "RateLimitError",
                "error_message": "Rate limit exceeded",
                "http_status": 429,
                "expected_behavior": "retry_with_backoff",
            },
            {
                "scenario": "timeout",
                "error_type": "TimeoutError",
                "error_message": "Request timed out",
                "http_status": 504,
                "expected_behavior": "return_timeout_error",
            },
            {
                "scenario": "internal_error",
                "error_type": "InternalServerError",
                "error_message": "Internal server error",
                "http_status": 500,
                "expected_behavior": "return_generic_error",
            },
        ]


if __name__ == "__main__":
    # Example usage of mock data
    print("Mock OpenAI Response:")
    print(json.dumps(MockOpenAIResponses.get_chat_completion_response(), indent=2))

    print("\nMock Energy Data:")
    print(json.dumps(MockEnergyData.get_solar_project_data(), indent=2))

    print("\nMock Configuration:")
    print(json.dumps(MockConfigurationData.get_test_config(), indent=2))
