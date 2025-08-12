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

from .config import ObservabilityConfig

# Import observability manager
from .observability import initialize_observability

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


# Telemetry and monitoring functionality has been moved to observability.py


# Monitoring decorator
# This is kept for backward compatibility but now delegates to observability.py
def monitor(operation_name: str):
    """Decorator for monitoring function execution."""

    # Import here to avoid circular imports
    from .observability import monitor as observability_monitor

    return observability_monitor(operation_name)


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


# Semantic Kernel factory


class KernelFactory:
    """Factory for creating Semantic Kernel instances with dynamic tool loading."""

    @staticmethod
    def create_kernel() -> Optional["Kernel"]:
        """Create a new Semantic Kernel instance."""
        if not SEMANTIC_KERNEL_AVAILABLE:
            return None

        from semantic_kernel import Kernel

        return Kernel()

    @staticmethod
    async def load_tools_from_registry(kernel: "Kernel", registry_client) -> int:
        """Load tools from external registry into the kernel."""
        if not kernel or not registry_client:
            return 0

        try:
            # Import registry client types

            # Fetch tools from registry
            tools = await registry_client.list_tools()
            loaded_count = 0

            for tool_def in tools:
                try:
                    # Convert registry tool definition to kernel-compatible function
                    sk_function = KernelFactory._create_kernel_function_from_registry(tool_def)

                    if sk_function:
                        kernel.add_function(plugin_name="registry_tools", function=sk_function)
                        loaded_count += 1
                        logging.getLogger("energyai_sdk.kernel_factory").info(
                            f"Loaded tool '{tool_def.name}' from registry"
                        )

                except Exception as e:
                    logging.getLogger("energyai_sdk.kernel_factory").error(
                        f"Error loading tool '{tool_def.name}': {e}"
                    )

            return loaded_count

        except Exception as e:
            logging.getLogger("energyai_sdk.kernel_factory").error(
                f"Error loading tools from registry: {e}"
            )
            return 0

    @staticmethod
    def _create_kernel_function_from_registry(tool_def) -> Optional[Any]:
        """Create a Semantic Kernel function from registry tool definition."""
        try:
            import json

            import aiohttp
            from semantic_kernel import kernel_function

            # If tool has an endpoint URL, create a function that calls it
            if tool_def.endpoint_url:

                @kernel_function(description=tool_def.description, name=tool_def.name)
                async def registry_tool_function(**kwargs) -> str:
                    """Dynamically created function from registry tool."""
                    try:
                        async with aiohttp.ClientSession() as session:
                            # Prepare request data
                            request_data = kwargs

                            # Add authentication if configured
                            headers = {"Content-Type": "application/json"}
                            if tool_def.auth_config:
                                if "api_key" in tool_def.auth_config:
                                    headers["Authorization"] = (
                                        f"Bearer {tool_def.auth_config['api_key']}"
                                    )
                                elif "headers" in tool_def.auth_config:
                                    headers.update(tool_def.auth_config["headers"])

                            # Make HTTP request to tool endpoint
                            async with session.post(
                                tool_def.endpoint_url, json=request_data, headers=headers
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    return json.dumps(result)
                                else:
                                    error_text = await response.text()
                                    return json.dumps(
                                        {
                                            "error": f"Tool request failed with status {response.status}",
                                            "details": error_text,
                                        }
                                    )

                    except Exception as e:
                        return json.dumps({"error": f"Failed to execute registry tool: {str(e)}"})

                return registry_tool_function

            else:
                # For tools without endpoints, create a placeholder function
                @kernel_function(
                    description=f"{tool_def.description} (Registry tool without endpoint)",
                    name=tool_def.name,
                )
                async def placeholder_tool(**kwargs) -> str:
                    return json.dumps(
                        {
                            "message": f"Tool '{tool_def.name}' is defined in registry but has no executable endpoint",
                            "tool_id": tool_def.id,
                            "parameters_received": kwargs,
                        }
                    )

                return placeholder_tool

        except Exception as e:
            logging.getLogger("energyai_sdk.kernel_factory").error(
                f"Error creating kernel function for tool '{tool_def.name}': {e}"
            )
            return None

    @staticmethod
    def configure_azure_openai_service(
        kernel: "Kernel",
        deployment_name: str,
        endpoint: str,
        api_key: str,
        api_version: str = "2024-02-01",
        service_id: Optional[str] = None,
    ):
        """Configure Azure OpenAI service on kernel."""
        try:
            from semantic_kernel.connectors.ai.azure_ai_inference import (
                AzureAIInferenceChatCompletion,
            )

            service = AzureAIInferenceChatCompletion(
                ai_model_id=deployment_name,
                api_key=api_key,
                endpoint=endpoint,
                api_version=api_version,
                service_id=service_id or deployment_name,
            )

            kernel.add_service(service)
        except ImportError:
            logging.getLogger("energyai_sdk.kernel_factory").error(
                "Azure AI Inference connector not available"
            )

    @staticmethod
    def configure_openai_service(
        kernel: "Kernel",
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        service_id: Optional[str] = None,
    ):
        """Configure OpenAI service on kernel."""
        try:
            from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

            service = OpenAIChatCompletion(
                ai_model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                service_id=service_id or model_id,
            )

            kernel.add_service(service)
        except ImportError:
            logging.getLogger("energyai_sdk.kernel_factory").error("OpenAI connector not available")


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

    # Configure observability
    observability_config = ObservabilityConfig(
        environment=environment,
        enable_langfuse=bool(langfuse_public_key and langfuse_secret_key),
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        enable_opentelemetry=bool(azure_monitor_connection_string),
        azure_monitor_connection_string=azure_monitor_connection_string,
    )

    # Initialize the observability manager
    initialize_observability(observability_config)

    logger.info("EnergyAI SDK initialized successfully")


# Global instances

# Global registry instance
agent_registry = AgentRegistry()

# Global context store
context_store = ContextStore()

# Note: The global telemetry_manager has been replaced by get_observability_manager() from observability.py
