# Unified Monitoring System Migration

This document describes the migration from the old fragmented monitoring components to the new unified MonitoringClient system in the EnergyAI SDK.

## Overview

The EnergyAI SDK has been updated to use a unified monitoring and observability system that consolidates previously separate components into a single, cohesive solution.

### Before: Fragmented Monitoring

The old system had multiple separate components:
- `LangfuseMonitoringClient` - Langfuse-specific client
- `TelemetryManager` - Legacy telemetry management
- `ObservabilityManager` - Basic observability wrapper
- Various separate imports and initialization

### After: Unified MonitoringClient

The new system provides:
- **Single `MonitoringClient`** - Unified interface for all monitoring needs
- **Integrated sub-clients** - Langfuse and OpenTelemetry clients managed internally
- **Consistent API** - One interface for all observability features
- **Simplified imports** - All monitoring functions from one module

## Migration Changes

### 1. Core Architecture

#### Old Structure
```
energyai_sdk/
├── clients/
│   ├── langfuse_client.py        # Separate Langfuse client
│   └── monitoring.py             # Basic monitoring
├── observability.py              # Observability manager
└── core.py                       # TelemetryManager
```

#### New Structure
```
energyai_sdk/
├── clients/
│   └── monitoring.py             # Unified MonitoringClient
└── core.py                       # No telemetry manager
```

### 2. Import Changes

#### Old Imports (Deprecated)
```python
# These imports are deprecated
from energyai_sdk import telemetry_manager, TelemetryManager
from energyai_sdk.clients import LangfuseMonitoringClient, get_langfuse_client
from energyai_sdk.observability import ObservabilityManager, get_observability_manager
```

#### New Imports (Recommended)
```python
# Use these imports instead
from energyai_sdk.clients.monitoring import (
    MonitoringClient,
    MonitoringConfig,
    initialize_monitoring,
    get_monitoring_client,
    monitor,
)
```

### 3. Initialization Changes

#### Old Initialization
```python
# Old approach - multiple separate initializations
from energyai_sdk import telemetry_manager
from energyai_sdk.clients import configure_langfuse

telemetry_manager.configure_azure_monitor(connection_string, service_name)
langfuse_client = configure_langfuse(public_key, secret_key, host)
```

#### New Initialization
```python
# New approach - unified configuration
from energyai_sdk.clients.monitoring import MonitoringConfig, initialize_monitoring

config = MonitoringConfig(
    enable_langfuse=True,
    langfuse_public_key="your_key",
    langfuse_secret_key="your_secret",
    enable_opentelemetry=True,
    azure_monitor_connection_string="your_connection"
)

monitoring_client = initialize_monitoring(config)
```

### 4. Usage Changes

#### Old Usage
```python
# Old - separate clients and managers
langfuse_client = get_langfuse_client()
trace = langfuse_client.trace(name="operation")
generation = trace.generation(name="llm_call")

with telemetry_manager.trace_operation("operation") as trace_id:
    # Operation code
    pass
```

#### New Usage
```python
# New - unified client
monitoring_client = get_monitoring_client()
trace = monitoring_client.create_trace(name="operation")
generation = monitoring_client.create_generation(trace, name="llm_call")

with monitoring_client.start_span("operation") as span:
    # Operation code
    pass
```

## Code Changes Made

### 1. Updated Files

#### Core SDK Files
- `energyai_sdk/__init__.py` - Updated exports and imports
- `energyai_sdk/core.py` - Removed TelemetryManager, added KernelManager
- `energyai_sdk/application.py` - Updated to use unified monitoring client
- `energyai_sdk/middleware.py` - Removed telemetry_manager references

#### Client Files
- `energyai_sdk/clients/__init__.py` - Updated exports for unified client
- `energyai_sdk/clients/monitoring.py` - Comprehensive unified monitoring implementation
- `energyai_sdk/clients/registry_client.py` - Updated for new exception types

#### Test Files
- `tests/test_unified_monitoring_integration.py` - New comprehensive tests
- `tests/test_langfuse_integration.py` - Marked as deprecated
- `tests/test_core.py` - Updated TelemetryManager tests

#### Example Files
- `examples/unified_observability_example.py` - Updated for new client
- `examples/registry_based_agents.py` - Uses unified monitoring

#### Documentation
- `docs/GETTING_STARTED.md` - Updated monitoring section
- `docs/REGISTRY_ARCHITECTURE.md` - New registry documentation
- `docs/UNIFIED_MONITORING_MIGRATION.md` - This migration guide

### 2. Removed/Deprecated Components

#### Completely Removed
- `energyai_sdk/observability.py` - Replaced by unified client
- `energyai_sdk/clients/langfuse_client.py` - Functionality moved to monitoring.py
- `TelemetryManager` class from `core.py` - Replaced by MonitoringClient

#### Deprecated (Kept for Compatibility)
- `tests/test_langfuse_integration.py` - Marked as deprecated
- Legacy import paths - Still work but show deprecation warnings

### 3. New Components Added

#### Core Components
- `MonitoringClient` - Unified monitoring interface
- `LangfuseClient` - Internal Langfuse client
- `OpenTelemetryClient` - Internal OpenTelemetry client
- `MonitoringConfig` - Unified configuration dataclass

#### Helper Components
- `MockMonitoringClient` - For testing and development
- `monitor` decorator - Simplified monitoring decorator
- `monitor_agent_execution` - Agent-specific monitoring
- `monitor_tool_execution` - Tool-specific monitoring

## Migration Guide

### For Application Developers

1. **Update imports** to use the new unified monitoring client
2. **Replace initialization** code with unified config approach
3. **Update monitoring calls** to use the new unified API
4. **Test thoroughly** to ensure monitoring still works as expected

### For SDK Contributors

1. **Use `get_monitoring_client()`** instead of separate client instances
2. **Configure via `MonitoringConfig`** for all monitoring needs
3. **Test with `MockMonitoringClient`** for unit tests
4. **Follow unified patterns** in new code

## Benefits of Migration

### 🎯 **Simplified Architecture**
- Single monitoring client instead of multiple components
- Consistent API across all monitoring features
- Reduced complexity in imports and initialization

### 🔧 **Better Configuration**
- Unified configuration object for all monitoring settings
- Environment-based configuration support
- Validation and type safety with dataclasses

### 🚀 **Enhanced Functionality**
- Combined Langfuse and OpenTelemetry in one client
- Automatic health checking across all monitoring systems
- Consistent error handling and logging

### 🧪 **Improved Testing**
- Comprehensive mock client for testing
- Unified test patterns for all monitoring features
- Better integration test coverage

### 📊 **Better Observability**
- Single point of configuration for all monitoring
- Consistent tracing across LLM calls and application metrics
- Unified health checks and status reporting

## Backward Compatibility

### What Still Works
- Basic SDK initialization and core functionality
- Agent and tool decorators
- Application factory functions
- Most existing examples and tests

### What's Deprecated
- Direct imports of `TelemetryManager`, `LangfuseMonitoringClient`, `ObservabilityManager`
- Separate configuration of monitoring components
- Old telemetry management patterns

### What's Removed
- `observability.py` module
- `clients/langfuse_client.py` module
- `TelemetryManager` class

## Future Enhancements

### Planned Features
- **Advanced Analytics** - More sophisticated metrics and dashboards
- **Custom Metrics** - User-defined metrics and alerts
- **Performance Monitoring** - Detailed performance analytics
- **Error Tracking** - Enhanced error reporting and aggregation

### Migration Timeline
- **Phase 1** ✅ - Unified MonitoringClient implementation
- **Phase 2** ✅ - Update all examples and tests
- **Phase 3** ✅ - Update documentation
- **Phase 4** (Future) - Remove deprecated components
- **Phase 5** (Future) - Advanced monitoring features

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Error: ImportError: cannot import name 'TelemetryManager'
# Fix: Use unified monitoring client
from energyai_sdk.clients.monitoring import get_monitoring_client
```

#### Configuration Issues
```python
# Error: Multiple separate configurations
# Fix: Use unified config
config = MonitoringConfig(enable_langfuse=True, enable_opentelemetry=True)
initialize_monitoring(config)
```

#### Missing Monitoring Data
```python
# Issue: Monitoring not initialized
# Fix: Ensure monitoring is initialized before use
client = get_monitoring_client()
if client is None:
    initialize_monitoring(MonitoringConfig())
```

### Getting Help

- Check the updated documentation in `docs/GETTING_STARTED.md`
- Review examples in `examples/unified_observability_example.py`
- Run tests in `tests/test_unified_monitoring_integration.py`
- Review this migration guide for specific changes

## Conclusion

The migration to the unified MonitoringClient provides a more robust, maintainable, and feature-rich monitoring system for the EnergyAI SDK. While some breaking changes were necessary, the new system offers significant improvements in usability, functionality, and maintainability.

All existing functionality is preserved or enhanced, and the new system provides a solid foundation for future monitoring and observability enhancements.
