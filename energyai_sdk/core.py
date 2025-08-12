# Core components for the EnergyAI SDK

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

try:
    from semantic_kernel import Kernel
    from semantic_kernel.contents.chat_history import ChatHistory

    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    SEMANTIC_KERNEL_AVAILABLE = False

from .clients.monitoring import MonitoringConfig, initialize_monitoring

# Data models


@dataclass
class AgentRequest:
    """Represents a request to an agent."""

    message: str
    agent_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentResponse:
    """Represents a response from an agent."""

    content: str
    agent_id: str
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    execution_time_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ToolDefinition:
    """Definition of a tool that can be used by agents."""

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillDefinition:
    """Definition of a skill containing multiple tools."""

    name: str
    description: str
    functions: list[Callable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """Template for prompts with variables."""

    name: str
    template: str
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerDefinition:
    """Definition of a planner for agent orchestration."""

    name: str
    description: str
    planner_type: str
    configuration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# Agent interface


class CoreAgent(ABC):
    """Base class for all agents in the EnergyAI SDK."""

    def __init__(self, agent_name: str, agent_description: str, system_prompt: str, **kwargs):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"energyai_sdk.agent.{agent_name}")

        # Initialize kernel and context
        if SEMANTIC_KERNEL_AVAILABLE:
            self.kernel = self._initialize_kernel()
            self.chat_history = ChatHistory()
            self.chat_history.add_system_message(system_prompt)

        # Agent metadata
        self.metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent_type": self.__class__.__name__,
        }

        # Configuration
        self._configure_agent(kwargs)

    @abstractmethod
    def _initialize_kernel(self) -> "Kernel":
        """Initialize the Semantic Kernel for this agent."""
        pass

    @abstractmethod
    def _configure_agent(self, config: dict[str, Any]) -> None:
        """Configure agent with provided parameters."""
        pass

    @abstractmethod
    async def _get_kernel_response(self, message: str) -> str:
        """Get response from kernel."""
        pass

    @abstractmethod
    def preprocess_message(self, message: str, context: dict[str, Any]) -> str:
        """Preprocess incoming message."""
        pass

    @abstractmethod
    def postprocess_response(self, response: str, context: dict[str, Any]) -> str:
        """Postprocess outgoing response."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model name."""
        pass

    @abstractmethod
    def reset_context(self) -> None:
        """Reset agent context."""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Get agent capabilities."""
        pass

    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process an agent request."""
        start_time = time.time()

        try:
            # Preprocess message
            context = {**request.metadata, "session_id": request.session_id}
            processed_message = self.preprocess_message(request.message, context)

            # Add message to chat history
            if hasattr(self, "chat_history"):
                self.chat_history.add_user_message(processed_message)

            # Get response from kernel
            raw_response = await self._get_kernel_response(processed_message)

            # Postprocess response
            final_response = self.postprocess_response(raw_response, context)

            # Calculate execution time
            execution_time = int((time.time() - start_time) * 1000)

            # Create response
            return AgentResponse(
                content=final_response,
                agent_id=self.agent_name,
                session_id=request.session_id,
                request_id=request.request_id,
                execution_time_ms=execution_time,
                metadata={
                    "model_used": self.get_default_model(),
                    "processed_message_length": len(processed_message),
                    "response_length": len(final_response),
                    **context,
                },
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error processing request: {e}")

            return AgentResponse(
                content=f"Error processing request: {str(e)}",
                agent_id=self.agent_name,
                session_id=request.session_id,
                request_id=request.request_id,
                execution_time_ms=execution_time,
                error=str(e),
                metadata={"error_type": type(e).__name__},
            )


# Registry system


class AgentRegistry:
    """Central registry for agents, tools, skills, prompts, and planners."""

    def __init__(self):
        self.agents: dict[str, CoreAgent] = {}
        self.tools: dict[str, ToolDefinition] = {}
        self.skills: dict[str, SkillDefinition] = {}
        self.prompts: dict[str, PromptTemplate] = {}
        self.planners: dict[str, PlannerDefinition] = {}
        self.logger = logging.getLogger("energyai_sdk.registry")

    # Agent management
    def register_agent(self, name: str, agent: CoreAgent) -> None:
        """Register an agent."""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")

    def get_agent(self, name: str) -> Optional[CoreAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self.agents.keys())

    def remove_agent(self, name: str) -> bool:
        """Remove an agent."""
        if name in self.agents:
            del self.agents[name]
            self.logger.info(f"Removed agent: {name}")
            return True
        return False

    # Tool management
    def register_tool(self, name: str, tool: ToolDefinition) -> None:
        """Register a tool."""
        self.tools[name] = tool
        self.logger.info(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    # Skill management
    def register_skill(self, name: str, skill: SkillDefinition) -> None:
        """Register a skill."""
        self.skills[name] = skill
        self.logger.info(f"Registered skill: {name}")

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get a skill by name."""
        return self.skills.get(name)

    def list_skills(self) -> list[str]:
        """List all registered skill names."""
        return list(self.skills.keys())

    # Prompt management
    def register_prompt(self, name: str, prompt: PromptTemplate) -> None:
        """Register a prompt template."""
        self.prompts[name] = prompt
        self.logger.info(f"Registered prompt: {name}")

    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.prompts.get(name)

    def list_prompts(self) -> list[str]:
        """List all registered prompt names."""
        return list(self.prompts.keys())

    # Planner management
    def register_planner(self, name: str, planner: PlannerDefinition) -> None:
        """Register a planner."""
        self.planners[name] = planner
        self.logger.info(f"Registered planner: {name}")

    def get_planner(self, name: str) -> Optional[PlannerDefinition]:
        """Get a planner by name."""
        return self.planners.get(name)

    def list_planners(self) -> list[str]:
        """List all registered planner names."""
        return list(self.planners.keys())

    # Registry capabilities
    def get_capabilities(self) -> dict[str, Any]:
        """Get overall registry capabilities."""
        return {
            "agents": self.list_agents(),
            "tools": self.list_tools(),
            "skills": self.list_skills(),
            "prompts": self.list_prompts(),
            "planners": self.list_planners(),
            "total_components": (
                len(self.agents)
                + len(self.tools)
                + len(self.skills)
                + len(self.prompts)
                + len(self.planners)
            ),
        }


# Telemetry


# Telemetry and monitoring functionality has been moved to clients/monitoring.py


# Monitoring decorator
# This is kept for backward compatibility but now delegates to clients/monitoring.py
def monitor(operation_name: str):
    """Decorator for monitoring function execution."""

    # Import here to avoid circular imports
    from .clients.monitoring import monitor as monitoring_decorator

    return monitoring_decorator(operation_name)


# Context store


class ContextStore:
    """Store for managing agent contexts and sessions."""

    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}
        self.user_contexts: dict[str, dict[str, Any]] = {}
        self.logger = logging.getLogger("energyai_sdk.context_store")

    def get_session_context(self, session_id: str) -> dict[str, Any]:
        """Get or create session context."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "messages": [],
                "metadata": {},
            }
        return self.sessions[session_id]

    def update_session_context(self, session_id: str, data: dict[str, Any]):
        """Update session context."""
        context = self.get_session_context(session_id)
        context.update(data)

    def add_message_to_session(self, session_id: str, role: str, content: str):
        """Add a message to session history."""
        context = self.get_session_context(session_id)
        context["messages"].append(
            {"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def get_user_context(self, user_id: str) -> dict[str, Any]:
        """Get or create user context."""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "preferences": {},
                "session_history": [],
                "metadata": {},
            }
        return self.user_contexts[user_id]

    def update_user_context(self, user_id: str, data: dict[str, Any]):
        """Update user context."""
        context = self.get_user_context(user_id)
        context.update(data)

    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Remove expired sessions."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        expired_sessions = []
        for session_id, context in self.sessions.items():
            created_at = datetime.fromisoformat(context["created_at"])
            if created_at < cutoff_time:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# Kernel Management


class KernelManager:
    """
    Manager for Semantic Kernel instances with caching and factory integration.

    This manager acts as a caching layer over the KernelFactory, providing
    efficient kernel retrieval and lifecycle management.
    """

    def __init__(self):
        """Initialize the KernelManager."""
        self.logger = logging.getLogger(__name__)
        self._factory = None
        self._kernel_cache = {}

    def _get_factory(self):
        """Get or create the KernelFactory instance."""
        if self._factory is None:
            from .clients.registry_client import RegistryClient
            from .config import ConfigurationManager
            from .kernel_factory import KernelFactory

            # Initialize with real clients
            config_manager = ConfigurationManager()

            # Try to initialize registry client with configuration
            registry_client = None
            try:
                config = config_manager.get_settings()
                # Check if we have Cosmos DB configuration
                if hasattr(config, "cosmos_endpoint") and hasattr(config, "cosmos_key"):
                    registry_client = RegistryClient(
                        cosmos_endpoint=config.cosmos_endpoint,
                        cosmos_key=config.cosmos_key,
                    )
                elif isinstance(config, dict) and config.get("cosmos_endpoint") and config.get("cosmos_key"):
                    registry_client = RegistryClient(
                        cosmos_endpoint=config["cosmos_endpoint"],
                        cosmos_key=config["cosmos_key"],
                    )
                else:
                    # Use mock client for development/testing
                    from .clients.registry_client import MockRegistryClient
                    registry_client = MockRegistryClient()
                    self.logger.info("Using MockRegistryClient for development")
            except Exception as e:
                self.logger.warning(f"Could not initialize registry client: {e}")
                # Fallback to mock client
                from .clients.registry_client import MockRegistryClient
                registry_client = MockRegistryClient()

            self._factory = KernelFactory(
                registry_client=registry_client,
                config_manager=config_manager,
            )

        return self._factory

    async def get_kernel_for_agent(
        self, agent_name: str, force_rebuild: bool = False
    ) -> Optional["Kernel"]:
        """
        Get a configured kernel for the specified agent.

        Args:
            agent_name: Name of the agent to get kernel for
            force_rebuild: Whether to force rebuild even if cached

        Returns:
            Configured Semantic Kernel instance or None if not available
        """
        if not SEMANTIC_KERNEL_AVAILABLE:
            self.logger.warning("Semantic Kernel not available")
            return None

        try:
            # Check cache first (unless force rebuild)
            if not force_rebuild and agent_name in self._kernel_cache:
                self.logger.debug(f"Using cached kernel for agent: {agent_name}")
                return self._kernel_cache[agent_name]

            # Use factory to create new kernel
            factory = self._get_factory()
            kernel = await factory.create_kernel_for_agent(agent_name, force_rebuild)

            # Cache the result
            if kernel:
                self._kernel_cache[agent_name] = kernel
                self.logger.info(f"Created and cached kernel for agent: {agent_name}")

            return kernel

        except Exception as e:
            self.logger.error(f"Failed to get kernel for agent {agent_name}: {e}")
            return None

    def get_cached_agent_names(self) -> list[str]:
        """Get list of agent names with cached kernels."""
        return list(self._kernel_cache.keys())

    def clear_cache(self, agent_name: Optional[str] = None) -> None:
        """
        Clear kernel cache.

        Args:
            agent_name: Specific agent to clear, or None to clear all
        """
        if agent_name:
            if agent_name in self._kernel_cache:
                del self._kernel_cache[agent_name]
                self.logger.info(f"Cleared cache for agent: {agent_name}")
        else:
            self._kernel_cache.clear()
            self.logger.info("Cleared all kernel cache")

    def invalidate_agent_cache(self, agent_name: str) -> None:
        """
        Invalidate cache for a specific agent.

        Args:
            agent_name: Name of agent to invalidate
        """
        self.clear_cache(agent_name)

    async def refresh_agent_kernel(self, agent_name: str) -> Optional["Kernel"]:
        """
        Refresh kernel for an agent by forcing rebuild.

        Args:
            agent_name: Name of agent to refresh

        Returns:
            New kernel instance
        """
        return await self.get_kernel_for_agent(agent_name, force_rebuild=True)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_agents": len(self._kernel_cache),
            "agent_names": list(self._kernel_cache.keys()),
            "factory_initialized": self._factory is not None,
        }




# SDK initialization


def initialize_sdk(
    log_level: str = "INFO",
    azure_monitor_connection_string: Optional[str] = None,
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: str = "https://cloud.langfuse.com",
    environment: str = "production",
):
    """Initialize the EnergyAI SDK."""

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("energyai_sdk")
    logger.info(f"Initializing EnergyAI SDK with log level: {log_level}")

    # Configure monitoring
    monitoring_config = MonitoringConfig(
        environment=environment,
        enable_langfuse=bool(langfuse_public_key and langfuse_secret_key),
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        enable_opentelemetry=bool(azure_monitor_connection_string),
        azure_monitor_connection_string=azure_monitor_connection_string,
    )

    # Initialize the monitoring client
    initialize_monitoring(monitoring_config)

    logger.info("EnergyAI SDK initialized successfully")


# Global instances

# Global registry instance
agent_registry = AgentRegistry()

# Global context store
context_store = ContextStore()

# Global kernel manager
kernel_manager = KernelManager()

# Note: The global telemetry_manager has been replaced by get_monitoring_client() from clients/monitoring.py
