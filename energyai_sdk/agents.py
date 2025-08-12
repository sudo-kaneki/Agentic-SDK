# Semantic Kernel agent implementation

import logging
from typing import Any, Optional

try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.contents.chat_history import ChatHistory

    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    SEMANTIC_KERNEL_AVAILABLE = False

from .core import CoreAgent, agent_registry
from .decorators import get_pending_agents

# Agent implementation


class SimpleSemanticKernelAgent(CoreAgent):
    """
    Minimal Semantic Kernel agent implementation.

    Focus: Simple wrapper around SK, no unnecessary complexity.
    """

    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        system_prompt: str,
        model_config: dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[list[str]] = None,
    ):
        super().__init__(agent_name, agent_description, system_prompt)
        self.model_config = model_config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_names = tools or []

        # Register with global registry
        agent_registry.register_agent(agent_name, self)

    def _initialize_kernel(self) -> "Kernel":
        """Initialize minimal Semantic Kernel."""
        if not SEMANTIC_KERNEL_AVAILABLE:
            raise ImportError(
                "Semantic Kernel not available. Install with: pip install semantic-kernel"
            )

        kernel = Kernel()

        # Add model service
        if self.model_config.get("service_type") == "openai":
            service = OpenAIChatCompletion(
                ai_model_id=self.model_config["deployment_name"],
                api_key=self.model_config["api_key"],
            )
        else:
            service = AzureAIInferenceChatCompletion(
                ai_model_id=self.model_config["deployment_name"],
                api_key=self.model_config["api_key"],
                endpoint=self.model_config["endpoint"],
            )

        kernel.add_service(service)

        # Add tools from registry
        for tool_name in self.tool_names:
            tool_def = agent_registry.get_tool(tool_name)
            if tool_def:
                kernel.add_function(
                    plugin_name="tools",
                    function_name=tool_name,
                    function=tool_def.function,
                    description=tool_def.description,
                )

        return kernel

    def _configure_agent(self, config: dict[str, Any]) -> None:
        """Simple configuration."""
        pass

    async def _get_kernel_response(self, message: str) -> str:
        """Get response from Semantic Kernel."""
        service = self.kernel.get_service(self.model_config["deployment_name"])
        result = await service.get_chat_message_content(
            chat_history=self.chat_history, kernel=self.kernel
        )
        return result.content

    def preprocess_message(self, message: str, context: dict[str, Any]) -> str:
        """Simple preprocessing."""
        return message.strip()

    def postprocess_response(self, response: str, context: dict[str, Any]) -> str:
        """Simple postprocessing."""
        return response.strip()

    def get_default_model(self) -> str:
        """Get model name."""
        return self.model_config["deployment_name"]

    def reset_context(self) -> None:
        """Reset chat history."""
        if SEMANTIC_KERNEL_AVAILABLE:
            self.chat_history = ChatHistory()
            self.chat_history.add_system_message(self.system_prompt)

    def get_capabilities(self) -> dict[str, Any]:
        """Get agent info."""
        return {
            "name": self.agent_name,
            "description": self.agent_description,
            "tools": self.tool_names,
            "model": self.get_default_model(),
        }


# Bootstrap function


def bootstrap_agents(
    azure_openai_config: Optional[dict[str, Any]] = None,
    openai_config: Optional[dict[str, Any]] = None,
) -> dict[str, CoreAgent]:
    """
    Create all decorated agents.

    Args:
        azure_openai_config: Azure OpenAI configuration
        openai_config: OpenAI configuration

    Returns:
        Dictionary of created agents
    """
    if not (azure_openai_config or openai_config):
        raise ValueError("Must provide model configuration")

    # Use provided config
    model_config = azure_openai_config or openai_config
    if azure_openai_config:
        model_config["service_type"] = "azure_openai"
    else:
        model_config["service_type"] = "openai"

    created_agents = {}
    pending_agents = get_pending_agents()

    # Create all standard agents first
    for agent_name, cls in pending_agents.items():
        config = cls._agent_config

        if config.get("agent_type") != "master":
            # Extract class attributes
            system_prompt = getattr(cls, "system_prompt", config["system_prompt"])
            temperature = getattr(cls, "temperature", 0.7)
            max_tokens = getattr(cls, "max_tokens", 1000)

            # Create simple agent
            agent = SimpleSemanticKernelAgent(
                agent_name=config["name"],
                agent_description=config["description"],
                system_prompt=system_prompt,
                model_config=model_config,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=config["tools"],
            )

            created_agents[agent_name] = agent

    # Create master agents (simplified - just delegate to subordinates)
    for agent_name, cls in pending_agents.items():
        config = cls._agent_config

        if config.get("agent_type") == "master":
            system_prompt = getattr(cls, "system_prompt", config["system_prompt"])

            # For simplicity, master agents just delegate to subordinates
            subordinate_names = config.get("subordinates", [])
            available_subordinates = [
                created_agents[name] for name in subordinate_names if name in created_agents
            ]

            if available_subordinates:
                # Create a simple master that delegates to first subordinate
                master = SimpleSemanticKernelAgent(
                    agent_name=config["name"],
                    agent_description=config["description"],
                    system_prompt=system_prompt,
                    model_config=model_config,
                    tools=[],
                )
                master._subordinates = available_subordinates  # Store subordinates
                created_agents[agent_name] = master

                logger = logging.getLogger("energyai_sdk.agents")
                logger.info(
                    f"Created master agent: {agent_name} with {len(available_subordinates)} subordinates"
                )

    return created_agents


# Convenience functions

# Aliases for backward compatibility
SemanticKernelAgent = SimpleSemanticKernelAgent


def get_agent(name):
    return agent_registry.get_agent(name)


def list_agents():
    return agent_registry.list_agents()


# Import agent decorator functions from decorators module

# Example usage

if __name__ == "__main__":
    print("✅ Simple Semantic Kernel implementation loaded!")
    print("Decorators are in decorators.py")
    print("Use bootstrap_agents(config) to create agents from decorators.")
