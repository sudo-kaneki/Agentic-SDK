# Registry-Based Agent Architecture

This document outlines the implementation of the declarative, database-driven agent registry and kernel factory system in the EnergyAI SDK.

## Overview

The registry-based architecture shifts from code-defined agents to a fully declarative approach where agents and tools are defined in JSON documents stored in Cosmos DB and loaded dynamically at runtime.

## Architecture Components

### 1. RegistryClient (`energyai_sdk/clients/registry_client.py`)

**Purpose**: Handles all communication with the Cosmos DB registry to fetch agent and tool definitions.

**Key Methods**:
- `get_agent_by_name(name: str)`: Fetch agent definition by name
- `get_tool_by_name(name: str, version: str)`: Fetch tool definition by name and version
- `get_agent_definition(agent_id: str)`: Fetch agent by ID (point-read)
- `get_tool_definition(tool_id: str)`: Fetch tool by ID (point-read)
- `list_agents()`, `list_tools()`: Query multiple definitions
- `health_check()`: Verify registry connectivity

**Features**:
- Caching with configurable TTL
- Error handling for missing definitions
- Support for both real Cosmos DB and mock implementations
- Async/await support for all operations

### 2. KernelFactory (`energyai_sdk/kernel_factory.py`)

**Purpose**: Builds Semantic Kernel instances dynamically based on agent definitions from the registry.

**Workflow**:
1. Load agent definition from registry
2. Resolve tool dependencies referenced by the agent
3. Convert tool schemas to OpenAPI v3 specifications
4. Create authentication providers for tools
5. Register tools as plugins in the kernel
6. Return fully configured kernel instance

**Key Methods**:
- `create_kernel_for_agent(agent_name: str)`: Main factory method
- `_convert_to_openapi(tool_def: ToolDefinition)`: Convert custom schema to OpenAPI
- `_register_http_tool(kernel, tool_def)`: Register HTTP-based tools as plugins
- `_create_auth_provider(auth_config)`: Create authentication providers

**OpenAPI Conversion**:
- Maps custom tool schemas to standard OpenAPI v3 format
- Supports function-type tools with parameter definitions
- Handles authentication schemes (API key, Bearer token)
- Generates complete OpenAPI specifications with paths, schemas, and security

### 3. KernelManager (`energyai_sdk/core.py`)

**Purpose**: Provides caching layer over KernelFactory for efficient kernel retrieval and lifecycle management.

**Key Methods**:
- `get_kernel_for_agent(agent_name: str)`: Get cached or create new kernel
- `refresh_agent_kernel(agent_name: str)`: Force rebuild kernel
- `clear_cache()`: Clear all or specific agent caches
- `get_cache_stats()`: Get caching statistics

**Features**:
- Intelligent caching to avoid rebuilding kernels
- Force rebuild capability for dynamic updates
- Global singleton instance available as `kernel_manager`
- Cache management and statistics

## Data Models

### AgentDefinition
```python
@dataclass
class AgentDefinition:
    id: str
    name: str
    description: str
    system_prompt: str
    model_config: Dict[str, Any]
    tools: List[str]  # Tool names referenced by this agent
    capabilities: List[str]
    temperature: float = 0.7
    max_tokens: int = 1000
    version: str = "1.0.0"
    # ... metadata fields
```

### ToolDefinition
```python
@dataclass
class ToolDefinition:
    id: str
    name: str
    description: str
    category: str
    schema: Dict[str, Any]  # Custom schema format
    endpoint_url: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    version: str = "1.0.0"
    # ... metadata fields
```

## Usage Example

```python
from energyai_sdk import kernel_manager
from energyai_sdk.clients import RegistryClient

# Get a kernel for an agent (cached or newly created)
kernel = await kernel_manager.get_kernel_for_agent("energy_analyst")

# Use the kernel with Semantic Kernel
result = await kernel.invoke_prompt("Calculate LCOE for a 100MW solar project")
```

## Benefits

### 🔧 Declarative Configuration
- Agents and tools defined in data, not code
- JSON schema-driven approach
- Configuration changes without code deployment

### 🚀 Dynamic Loading
- Add/modify agents without application restart
- Runtime tool discovery and loading
- Version management for tools and agents

### 📦 Scalability
- Centralized registry serves multiple applications
- Shared tool definitions across teams
- Horizontal scaling with Cosmos DB

### 🔄 Efficient Caching
- Kernel instances cached for performance
- Intelligent cache invalidation
- Memory-efficient resource management

### 🛡️ Security
- API keys managed through secret references
- Authentication providers for tool access
- Secure credential handling

## Implementation Details

### OpenAPI Schema Conversion

The `_convert_to_openapi()` method performs critical data transformation:

1. **Input**: Custom tool schema in JSON format
2. **Process**: Maps to OpenAPI v3 specification structure
3. **Output**: Valid OpenAPI JSON string for Semantic Kernel

Example transformation:
```python
# Custom Tool Schema
{
    "type": "function",
    "function": {
        "name": "calculate_lcoe",
        "description": "Calculate Levelized Cost of Energy",
        "parameters": {
            "type": "object",
            "properties": {
                "capex": {"type": "number"},
                "opex": {"type": "number"}
            }
        }
    }
}

# Generated OpenAPI Spec
{
    "openapi": "3.0.1",
    "info": {"title": "Energy Calculator", "version": "1.0.0"},
    "paths": {
        "/calculate_lcoe": {
            "post": {
                "operationId": "calculate_lcoe",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "capex": {"type": "number"},
                                    "opex": {"type": "number"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### Error Handling

- Graceful degradation when registry is unavailable
- Fallback to cached definitions when possible
- Comprehensive error reporting with context
- Optional dependency handling (semantic-kernel, azure-cosmos)

### Testing Support

- `MockRegistryClient` for development and testing
- Sample agent and tool definitions included
- Comprehensive example demonstrating all features

## Configuration

### Real Registry Client
```python
registry_client = RegistryClient(
    cosmos_endpoint="https://your-cosmos.documents.azure.com:443/",
    cosmos_key="your-cosmos-primary-key",
    database_name="AgenticPlatform",
    agents_container="Agents",
    tools_container="Tools"
)
```

### Mock Registry Client (Development)
```python
registry_client = MockRegistryClient()
# Includes sample agent and tool definitions
```

## Future Enhancements

- **Tool Versioning**: Advanced version resolution and compatibility checking
- **Agent Composition**: Support for multi-agent workflows and hierarchies
- **Runtime Updates**: Hot-reload capabilities for agent definition changes
- **Metrics Integration**: Tool usage analytics and performance monitoring
- **Security Enhancements**: Role-based access control for agent definitions

## Files Modified/Created

- `energyai_sdk/clients/registry_client.py` - Registry client implementation
- `energyai_sdk/kernel_factory.py` - New kernel factory module
- `energyai_sdk/core.py` - Updated with KernelManager
- `energyai_sdk/clients/__init__.py` - Updated exports
- `energyai_sdk/__init__.py` - Updated exports
- `examples/registry_based_agents.py` - Comprehensive demonstration
- `docs/REGISTRY_ARCHITECTURE.md` - This documentation

This architecture provides a solid foundation for building scalable, declarative AI agent systems that can evolve and adapt without requiring code changes.
