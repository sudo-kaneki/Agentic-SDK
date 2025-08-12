"""
Client modules for external service integration.

This package contains clients for:
- Agentic Registry (Cosmos DB) - for fetching agent and tool definitions
- Context Store (Cosmos DB) - for session persistence
- Monitoring (OpenTelemetry) - for observability and telemetry
- Other external services
"""

from .context_store_client import ContextStoreClient, MockContextStoreClient
from .monitoring import MockMonitoringClient, MonitoringClient, MonitoringConfig
from .registry_client import MockRegistryClient, RegistryClient

__all__ = [
    "RegistryClient",
    "MockRegistryClient",
    "ContextStoreClient",
    "MockContextStoreClient",
    "MonitoringClient",
    "MonitoringConfig",
    "MockMonitoringClient",
]
