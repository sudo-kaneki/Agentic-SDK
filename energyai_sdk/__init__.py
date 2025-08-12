# EnergyAI Agentic SDK - Package Initialization and Public API
# File: energyai_sdk/__init__.py

"""
EnergyAI Agentic SDK

A comprehensive SDK for building AI agents specialized in energy analytics and management.
This package provides tools, skills, and frameworks for creating intelligent energy applications.

Main components:
- Core: Registry, Context Store, Kernel Factory, and base agent classes
- Decorators: @agent, @tool, @skill, @prompt, @planner decorators
- Agents: Concrete agent implementations using Semantic Kernel
- Application: FastAPI-based web application framework
- Middleware: Request/response processing pipeline
- Exceptions: Custom error handling classes
"""

__version__ = "1.0.0"
__author__ = "EnergyAI Team"
__license__ = "MIT"

# ==============================================================================
# CORE IMPORTS - Registry, Context, and Base Classes
# ==============================================================================

from .core import (  # Data Models; Base Classes; Registry System; Context Management; Kernel Factory; SDK Initialization
    AgentRegistry,
    AgentRequest,
    AgentResponse,
    ContextStore,
    CoreAgent,
    KernelFactory,
    KernelManager,
    PlannerDefinition,
    PromptTemplate,
    SkillDefinition,
    ToolDefinition,
    agent_registry,  # Global instance
    context_store,  # Global instance
    initialize_sdk,
    kernel_manager,  # Global instance
    monitor,
)
from .decorators import (  # Primary Decorators; Agent Decorators - Your Main Vision!; Utility Functions
    agent,
    get_registered_agent_classes,
    get_registered_planners,
    get_registered_prompts,
    get_registered_skills,
    get_registered_tools,
    is_agent,
    is_planner,
    is_prompt,
    is_skill,
    is_tool,
    master_agent,
    planner,
    prompt,
    skill,
    tool,
)
from .exceptions import (  # Base Exceptions; Agent Exceptions; Registry Exceptions; Tool and Skill Exceptions; Middleware Exceptions; Application Exceptions; Telemetry Exceptions; Model and API Exceptions; Utility Functions
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentModelError,
    AgentNotFoundError,
    AgentTimeoutError,
    APIError,
    APIRateLimitError,
    APITimeoutError,
    ApplicationError,
    ApplicationShutdownError,
    ApplicationStartupError,
    AuthenticationError,
    AuthorizationError,
    ComponentNotFoundError,
    ComponentRegistrationError,
    DuplicateComponentError,
    EnergyAIConfigurationError,
    EnergyAISDKError,
    EnergyAIValidationError,
    ErrorContext,
    MiddlewareError,
    MiddlewareExecutionError,
    ModelConfigurationError,
    ModelError,
    RateLimitExceededError,
    RegistryError,
    SkillError,
    SkillLoadError,
    TelemetryConfigurationError,
    TelemetryError,
    TelemetryTraceError,
    ToolError,
    ToolExecutionError,
    ToolParameterError,
    create_error_response,
    format_exception_chain,
    is_retryable_error,
)

# Kernel Factory module
try:
    from .kernel_factory import KernelFactory as NewKernelFactory

    # Make the new KernelFactory available as well
    _KERNEL_FACTORY_AVAILABLE = True
except ImportError:
    _KERNEL_FACTORY_AVAILABLE = False

# ==============================================================================
# DECORATORS - ALL Component Registration in One Place
# ==============================================================================


# ==============================================================================
# EXCEPTIONS - Error Handling
# ==============================================================================


# ==============================================================================
# CONDITIONAL IMPORTS - Optional Components
# ==============================================================================

# Simple Agents module - minimal Semantic Kernel wrapper
try:
    from .agents import (
        SimpleSemanticKernelAgent as SemanticKernelAgent,  # Simple Agent Implementation; Agent Creation Function; Convenience Functions
    )
    from .agents import bootstrap_agents, get_agent, list_agents

    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False

# Monitoring - Unified monitoring and observability
try:
    from .clients.monitoring import (
        MonitoringClient,
        MonitoringConfig,
        get_monitoring_client,
        initialize_monitoring,
        monitor_agent_execution,
        monitor_tool_execution,
    )

    _MONITORING_AVAILABLE = True
except ImportError:
    _MONITORING_AVAILABLE = False

# Application module - depends on FastAPI
try:
    from .application import (  # Core Application; API Models; Application Factory; Development Server
        AgentInfo,
        ChatRequest,
        ChatResponse,
        DevelopmentServer,
        EnergyAIApplication,
        HealthCheck,
        StreamingChatResponse,
        create_application,
        create_production_application,
        get_application,
        run_development_server,
    )

    _APPLICATION_AVAILABLE = True
except ImportError:
    _APPLICATION_AVAILABLE = False

# Clients module - external service integrations
try:
    from .clients import AgentDefinition, ContextStoreClient, RegistryClient, ToolDefinition

    _CLIENTS_AVAILABLE = True
except ImportError:
    _CLIENTS_AVAILABLE = False

# Legacy compatibility - these are now part of the unified MonitoringClient
try:
    from .clients.monitoring import (
        MonitoringClient as LangfuseMonitoringClient,  # For backward compatibility
    )

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

# Middleware module
try:
    from .middleware import (  # Core Middleware; Middleware Implementations; Pipeline Factory
        AgentMiddleware,
        AuthenticationMiddleware,
        AuthorizationMiddleware,
        CachingMiddleware,
        ErrorHandlingMiddleware,
        MiddlewareContext,
        MiddlewarePipeline,
        RateLimitingMiddleware,
        TelemetryMiddleware,
        ValidationMiddleware,
        create_default_pipeline,
        create_production_pipeline,
    )

    _MIDDLEWARE_AVAILABLE = True
except ImportError:
    _MIDDLEWARE_AVAILABLE = False

# Configuration module
try:
    from .config import (  # Configuration Models; Configuration Manager; Example Functions (if needed)
        ApplicationConfig,
        ConfigurationManager,
        LogLevel,
        ModelConfig,
        ModelType,
        SecurityConfig,
        TelemetryConfig,
        config_manager,  # Global instance
        create_energy_agents,
        create_energy_platform,
        get_settings,
    )

    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

# ==============================================================================
# PUBLIC API DEFINITION
# ==============================================================================

# Core API - Always Available
__all__ = [
    # Version and Metadata
    "__version__",
    "__author__",
    "__license__",
    # Core Data Models
    "AgentRequest",
    "AgentResponse",
    "ToolDefinition",
    "SkillDefinition",
    "PromptTemplate",
    "PlannerDefinition",
    # Base Classes
    "CoreAgent",
    # Registry
    "AgentRegistry",
    "agent_registry",
    # Monitoring (unified system)
    "monitor",
    # Context
    "ContextStore",
    "context_store",
    # Kernel Factory and Manager
    "KernelFactory",
    "KernelManager",
    "kernel_manager",
    # Initialization
    "initialize_sdk",
    "get_available_features",
    "quick_start",
    # Decorators
    "tool",
    "skill",
    "prompt",
    "planner",
    "agent",
    "get_registered_tools",
    "get_registered_skills",
    "get_registered_prompts",
    "get_registered_planners",
    "is_tool",
    "is_skill",
    "is_prompt",
    "is_planner",
    "is_agent",
    # Exceptions
    "EnergyAISDKError",
    "EnergyAIConfigurationError",
    "EnergyAIValidationError",
    "AgentError",
    "AgentNotFoundError",
    "AgentInitializationError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentModelError",
    "RegistryError",
    "ComponentNotFoundError",
    "ComponentRegistrationError",
    "DuplicateComponentError",
    "ToolError",
    "ToolExecutionError",
    "ToolParameterError",
    "SkillError",
    "SkillLoadError",
    "MiddlewareError",
    "MiddlewareExecutionError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitExceededError",
    "ApplicationError",
    "ApplicationStartupError",
    "ApplicationShutdownError",
    "TelemetryError",
    "TelemetryConfigurationError",
    "TelemetryTraceError",
    "ModelError",
    "ModelConfigurationError",
    "APIError",
    "APITimeoutError",
    "APIRateLimitError",
    "format_exception_chain",
    "create_error_response",
    "is_retryable_error",
    "ErrorContext",
]

# Conditional Exports
if _AGENTS_AVAILABLE:
    __all__.extend(
        [
            "SemanticKernelAgent",  # Simple implementation
            "bootstrap_agents",  # Create all agents
            "get_agent",
            "list_agents",
        ]
    )

if _APPLICATION_AVAILABLE:
    __all__.extend(
        [
            "EnergyAIApplication",
            "ChatRequest",
            "ChatResponse",
            "AgentInfo",
            "HealthCheck",
            "StreamingChatResponse",
            "create_application",
            "create_production_application",
            "DevelopmentServer",
            "run_development_server",
            "get_application",
        ]
    )

if _CLIENTS_AVAILABLE:
    __all__.extend(
        [
            "ContextStoreClient",
            "RegistryClient",
            "AgentDefinition",
            "ToolDefinition",
        ]
    )

if _LANGFUSE_AVAILABLE:
    __all__.extend(
        [
            "LangfuseMonitoringClient",  # Backward compatibility alias
        ]
    )

if _MONITORING_AVAILABLE:
    __all__.extend(
        [
            "MonitoringClient",
            "MonitoringConfig",
            "get_monitoring_client",
            "initialize_monitoring",
            "monitor_agent_execution",
            "monitor_tool_execution",
        ]
    )

if _MIDDLEWARE_AVAILABLE:
    __all__.extend(
        [
            "MiddlewareContext",
            "AgentMiddleware",
            "MiddlewarePipeline",
            "AuthenticationMiddleware",
            "AuthorizationMiddleware",
            "RateLimitingMiddleware",
            "ValidationMiddleware",
            "CachingMiddleware",
            "TelemetryMiddleware",
            "ErrorHandlingMiddleware",
            "create_default_pipeline",
            "create_production_pipeline",
        ]
    )

if _CONFIG_AVAILABLE:
    __all__.extend(
        [
            "LogLevel",
            "ModelType",
            "ModelConfig",
            "TelemetryConfig",
            "SecurityConfig",
            "ApplicationConfig",
            "ConfigurationManager",
            "config_manager",
            "get_settings",
            "create_energy_platform",
            "create_energy_agents",
        ]
    )

# ==============================================================================
# COMPATIBILITY AND FEATURE DETECTION
# ==============================================================================


def get_available_features() -> dict:
    """Get information about available features in this installation."""
    return {
        "core": True,  # Always available
        "agents": _AGENTS_AVAILABLE,
        "application": _APPLICATION_AVAILABLE,
        "clients": _CLIENTS_AVAILABLE,
        "middleware": _MIDDLEWARE_AVAILABLE,
        "config": _CONFIG_AVAILABLE,
        "semantic_kernel": _AGENTS_AVAILABLE,  # Same as agents
        "fastapi": _APPLICATION_AVAILABLE,  # Same as application
        "cosmos_db": _CLIENTS_AVAILABLE,  # Same as clients
    }


def check_dependencies(feature: str) -> bool:
    """Check if dependencies for a specific feature are available."""
    features = get_available_features()
    return features.get(feature, False)


def require_feature(feature: str) -> None:
    """Raise an error if a required feature is not available."""
    if not check_dependencies(feature):
        missing_deps = {
            "agents": "semantic-kernel",
            "application": "fastapi, uvicorn",
            "semantic_kernel": "semantic-kernel",
            "fastapi": "fastapi, uvicorn",
        }

        deps = missing_deps.get(feature, f"dependencies for {feature}")
        raise EnergyAIConfigurationError(
            f"Feature '{feature}' is not available. Please install: {deps}",
            error_code="MISSING_DEPENDENCIES",
        )


# ==============================================================================
# PACKAGE METADATA
# ==============================================================================

__package_info__ = {
    "name": "energyai-sdk",
    "version": __version__,
    "description": "AI Agent SDK for Energy Analytics",
    "author": __author__,
    "license": __license__,
    "features": get_available_features(),
    "core_components": ["agent_registry", "kernel_manager", "context_store"],
    "optional_components": [
        "agents" if _AGENTS_AVAILABLE else None,
        "application" if _APPLICATION_AVAILABLE else None,
        "clients" if _CLIENTS_AVAILABLE else None,
        "middleware" if _MIDDLEWARE_AVAILABLE else None,
        "config" if _CONFIG_AVAILABLE else None,
    ],
}

# Clean up None values
__package_info__["optional_components"] = [
    comp for comp in __package_info__["optional_components"] if comp is not None
]

# ==============================================================================
# PACKAGE INITIALIZATION
# ==============================================================================


def _initialize_package():
    """Initialize package components."""
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(f"EnergyAI SDK {__version__} loaded")
    logger.debug(f"Available features: {list(get_available_features().keys())}")

    # Initialize global components
    if not hasattr(agent_registry, "_initialized"):
        agent_registry._initialized = True

    # Initialization is now handled by the MonitoringClient

    if not hasattr(context_store, "_initialized"):
        context_store._initialized = True


# Initialize when package is imported
_initialize_package()

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def get_available_features() -> dict:
    """Get dictionary of available features and their status."""
    return {
        "core": _AGENTS_AVAILABLE,  # Core functionality same as agents
        "agents": _AGENTS_AVAILABLE,
        "application": _APPLICATION_AVAILABLE,
        "clients": _CLIENTS_AVAILABLE,
        "cosmos_db": _CLIENTS_AVAILABLE,  # Same as clients
        "langfuse": _LANGFUSE_AVAILABLE,
        "middleware": _MIDDLEWARE_AVAILABLE,
        "config": _CONFIG_AVAILABLE,
        "monitoring": _MONITORING_AVAILABLE,  # Unified monitoring and observability
        "semantic_kernel": _AGENTS_AVAILABLE,  # Core includes SK
        "fastapi": _APPLICATION_AVAILABLE,  # Application includes FastAPI
    }


def quick_start(log_level: str = "INFO", enable_telemetry: bool = False, **telemetry_config):
    """Quick start function for basic SDK setup."""
    if enable_telemetry:
        initialize_sdk(log_level=log_level, **telemetry_config)
    else:
        initialize_sdk(log_level=log_level)

    # Get monitoring client if available
    monitoring = None
    if _MONITORING_AVAILABLE:
        monitoring = get_monitoring_client()

    return {
        "registry": agent_registry,
        "monitoring": monitoring,
        "context": context_store,
        "features": get_available_features(),
    }


# ==============================================================================
# EXAMPLE USAGE DOCUMENTATION
# ==============================================================================

__doc__ += """

Quick Start Examples:

1. Basic Tool Registration:
    ```python
    from energyai_sdk import tool, agent_registry

    @tool(name="calculate_lcoe", description="Calculate LCOE")
    def calculate_lcoe(capital_cost: float, annual_generation: float):
        return {"lcoe": capital_cost / annual_generation}

    print(agent_registry.list_tools())  # ['calculate_lcoe']
    ```

2. Skill Definition:
    ```python
    from energyai_sdk import skill, tool

    @skill(name="EnergyEconomics", description="Energy financial tools")
    class EnergyEconomics:
        @tool(name="npv")
        def calculate_npv(self, cash_flows, discount_rate):
            # NPV calculation logic
            return {"npv": sum(cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows))}
    ```

3. Agent Creation (if agents module available):
    ```python
    from energyai_sdk import agent, bootstrap_agents

    @agent(
        name="EnergyAnalyst",
        description="Energy market analyst",
        system_prompt="You are an expert energy analyst.",
        tools=["calculate_lcoe"]
    )
    class EnergyAnalyst:
        temperature = 0.7
        max_tokens = 1000

    # Bootstrap agents with configuration
    azure_config = {"deployment_name": "gpt-4o", "api_key": "your-key", "endpoint": "your-endpoint"}
    agents = bootstrap_agents(azure_openai_config=azure_config)
    ```

4. Application Setup (if application module available):
    ```python
    from energyai_sdk import create_application, run_development_server, bootstrap_agents

    # Create agents using decorators and bootstrap them
    azure_config = {"deployment_name": "gpt-4o", "api_key": "your-key", "endpoint": "your-endpoint"}
    agents = bootstrap_agents(azure_openai_config=azure_config)

    app = create_application(title="Energy Analytics API")
    for agent in agents.values():
        app.add_agent(agent)

    # Run development server
    run_development_server(list(agents.values()), host="127.0.0.1", port=8000)
    ```

5. Feature Detection:
    ```python
    from energyai_sdk import get_available_features, require_feature

    features = get_available_features()
    print(f"Available: {features}")

    # Ensure required features are available
    require_feature("agents")  # Raises error if not available
    ```

For more examples, see the examples/ directory.
"""
