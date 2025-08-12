"""
Client modules for external service integration.

This package contains clients for:
- Agentic Registry (Cosmos DB) - for fetching agent and tool definitions
- Context Store (Cosmos DB) - for session persistence
- Monitoring - unified monitoring and observability (OpenTelemetry + Langfuse)
- Other external services
"""

from .context_store_client import ContextStoreClient

# Registry client for fetching agent and tool definitions
try:
    from .registry_client import AgentDefinition, MockRegistryClient, RegistryClient, ToolDefinition

    REGISTRY_CLIENT_AVAILABLE = True
except ImportError:
    REGISTRY_CLIENT_AVAILABLE = False

# Unified monitoring client
try:
    from .monitoring import (
        MonitoringClient,
        MonitoringConfig,
        get_monitoring_client,
        initialize_monitoring,
        monitor,
        monitor_agent_execution,
        monitor_tool_execution,
    )

    MONITORING_CLIENT_AVAILABLE = True
except ImportError:
    MONITORING_CLIENT_AVAILABLE = False

__all__ = [
    "ContextStoreClient",
]

if REGISTRY_CLIENT_AVAILABLE:
    __all__.extend(
        [
            "RegistryClient",
            "AgentDefinition",
            "ToolDefinition",
            "MockRegistryClient",
        ]
    )

if MONITORING_CLIENT_AVAILABLE:
    __all__.extend(
        [
            "MonitoringClient",
            "MonitoringConfig",
            "get_monitoring_client",
            "initialize_monitoring",
            "monitor_agent_execution",
            "monitor_tool_execution",
            "monitor",
        ]
    )
