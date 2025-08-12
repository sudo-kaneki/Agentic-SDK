"""
Client modules for external service integration.

This package contains clients for:
- Agentic Registry (Cosmos DB) - for fetching agent and tool definitions
- Context Store (Cosmos DB) - for session persistence
- Monitoring (OpenTelemetry) - for observability and telemetry
- Langfuse - for LLM observability and tracing
- Other external services
"""

from .context_store_client import ContextStoreClient

# Langfuse client - optional dependency
try:
    from .langfuse_client import LangfuseMonitoringClient, configure_langfuse, get_langfuse_client

    LANGFUSE_CLIENT_AVAILABLE = True
except ImportError:
    LANGFUSE_CLIENT_AVAILABLE = False

__all__ = [
    "ContextStoreClient",
]

if LANGFUSE_CLIENT_AVAILABLE:
    __all__.extend(
        [
            "LangfuseMonitoringClient",
            "get_langfuse_client",
            "configure_langfuse",
        ]
    )
