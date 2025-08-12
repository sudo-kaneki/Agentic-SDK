"""
Kernel Factory for EnergyAI SDK.

This module provides the KernelFactory class for building Semantic Kernel instances
dynamically based on agent definitions from the registry. The factory handles:

1. Loading agent definitions from the registry
2. Loading tool definitions for each agent
3. Converting custom tool schemas to OpenAPI specifications
4. Creating and configuring kernel instances with plugins
5. Setting up authentication providers for tools
"""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.functions.kernel_function_from_openapi import kernel_function_from_openapi

    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    SEMANTIC_KERNEL_AVAILABLE = False

from .clients.registry_client import AgentDefinition, RegistryClient, ToolDefinition
from .config import ConfigurationManager
from .exceptions import EnergyAISDKError


class KernelFactory:
    """
    Factory for creating Semantic Kernel instances based on agent definitions.

    This factory encapsulates the logic for:
    - Loading agent definitions from the registry
    - Resolving tool dependencies
    - Converting tool schemas to OpenAPI specifications
    - Building configured kernel instances
    """

    def __init__(
        self,
        registry_client: Optional[RegistryClient] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the KernelFactory.

        Args:
            registry_client: Client for accessing the agent/tool registry
            config_manager: Configuration manager for SDK settings
        """
        self.logger = logging.getLogger(__name__)

        if not SEMANTIC_KERNEL_AVAILABLE:
            raise EnergyAISDKError(
                "Semantic Kernel not available. Install with: pip install semantic-kernel"
            )

        self.registry_client = registry_client
        self.config_manager = config_manager or ConfigurationManager()

        # Cache for built kernels to avoid rebuilding
        self._kernel_cache: Dict[str, "Kernel"] = {}
        # Cache for built tools to avoid rebuilding
        self._tool_cache: Dict[str, Any] = {}

    async def create_kernel_for_agent(
        self, agent_name: str, force_rebuild: bool = False
    ) -> Optional["Kernel"]:
        """
        Create a configured Semantic Kernel instance for the specified agent.

        Args:
            agent_name: Name of the agent to create kernel for
            force_rebuild: Whether to force rebuild even if cached

        Returns:
            Configured Semantic Kernel instance

        Raises:
            EnergyAISDKError: If agent not found or kernel creation fails
        """
        if not force_rebuild and agent_name in self._kernel_cache:
            self.logger.debug(f"Using cached kernel for agent: {agent_name}")
            return self._kernel_cache[agent_name]

        self.logger.info(f"Creating kernel for agent: {agent_name}")

        try:
            # Step 1: Load agent definition from registry
            if not self.registry_client:
                raise EnergyAISDKError("Registry client not configured")

            agent_def = await self.registry_client.get_agent_by_name(agent_name)
            if not agent_def:
                raise EnergyAISDKError(f"Agent '{agent_name}' not found in registry")

            # Step 2: Create base kernel with LLM service
            kernel = Kernel()
            await self._add_llm_service(kernel, agent_def)

            # Step 3: Load and register tools as plugins
            await self._load_and_register_tools(kernel, agent_def)

            # Step 4: Cache the kernel
            self._kernel_cache[agent_name] = kernel

            self.logger.info(f"Successfully created kernel for agent: {agent_name}")
            return kernel

        except Exception as e:
            self.logger.error(f"Failed to create kernel for agent {agent_name}: {e}")
            raise EnergyAISDKError(f"Kernel creation failed: {e}") from e

    async def _add_llm_service(self, kernel: "Kernel", agent_def: AgentDefinition) -> None:
        """
        Add the LLM service to the kernel based on agent configuration.

        Args:
            kernel: Kernel to configure
            agent_def: Agent definition containing model configuration
        """
        try:
            model_config = agent_def.model_config

            # Get configuration from settings
            config = self.config_manager.get_settings()

            # Determine model type and configure accordingly
            deployment_name = model_config.get("deployment_name", "gpt-4")
            model_type = model_config.get("model_type", "azure_openai")
            temperature = agent_def.temperature
            
            if model_type == "azure_openai":
                await self._configure_azure_openai(
                    kernel, deployment_name, temperature, config, agent_def
                )
            elif model_type == "openai":
                await self._configure_openai(
                    kernel, deployment_name, temperature, config, agent_def
                )
            else:
                self.logger.warning(f"Unknown model type: {model_type}")

            self.logger.debug(
                f"Added LLM service for deployment: {model_config.get('deployment_name', 'gpt-4')}"
            )

        except Exception as e:
            self.logger.error(f"Failed to add LLM service: {e}")
            raise EnergyAISDKError(f"LLM service configuration failed: {e}") from e

    async def _configure_azure_openai(
        self,
        kernel: "Kernel",
        deployment_name: str,
        temperature: float,
        config: Any,
        agent_def: AgentDefinition,
    ):
        """Configure Azure OpenAI service."""
        if hasattr(config, "azure_openai_endpoint"):
            endpoint = config.azure_openai_endpoint
            api_key = config.azure_openai_api_key
            api_version = getattr(config, "azure_openai_api_version", "2024-02-01")
        else:
            endpoint = config.get("azure_openai_endpoint")
            api_key = config.get("azure_openai_api_key")
            api_version = config.get("azure_openai_api_version", "2024-02-01")

        if not endpoint or not api_key:
            raise EnergyAISDKError("Azure OpenAI configuration missing")

        service = AzureAIInferenceChatCompletion(
            ai_model_id=deployment_name,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            service_id=f"{agent_def.name}_{deployment_name}",
        )
        kernel.add_service(service)
        self.logger.info(f"Configured Azure OpenAI service: {deployment_name}")

    async def _configure_openai(
        self,
        kernel: "Kernel",
        deployment_name: str,
        temperature: float,
        config: Any,
        agent_def: AgentDefinition,
    ):
        """Configure OpenAI service."""
        if hasattr(config, "openai_api_key"):
            api_key = config.openai_api_key
            base_url = getattr(config, "openai_base_url", None)
        else:
            api_key = config.get("openai_api_key")
            base_url = config.get("openai_base_url")

        if not api_key:
            raise EnergyAISDKError("OpenAI API key missing")

        service = OpenAIChatCompletion(
            ai_model_id=deployment_name,
            api_key=api_key,
            base_url=base_url,
            service_id=f"{agent_def.name}_{deployment_name}",
        )
        kernel.add_service(service)
        self.logger.info(f"Configured OpenAI service: {deployment_name}")

    async def _load_and_register_tools(self, kernel: "Kernel", agent_def: AgentDefinition) -> None:
        """
        Load tools from registry and register them as plugins in the kernel.

        Args:
            kernel: Kernel to add plugins to
            agent_def: Agent definition containing tool references
        """
        if not agent_def.tools:
            self.logger.info("No tools specified for agent")
            return

        for tool_name in agent_def.tools:
            try:
                await self._register_tool_as_plugin(kernel, tool_name)
            except Exception as e:
                self.logger.warning(f"Failed to register tool {tool_name}: {e}")
                # Continue with other tools rather than failing completely

    async def _register_tool_as_plugin(
        self, kernel: "Kernel", tool_name: str, version: str = "1.0.0"
    ) -> None:
        """
        Register a single tool as a plugin in the kernel.

        Args:
            kernel: Kernel to add plugin to
            tool_name: Name of the tool to register
            version: Version of the tool (default: "1.0.0")
        """
        # Load tool definition from registry
        tool_def = await self.registry_client.get_tool_by_name(tool_name, version)
        if not tool_def:
            raise EnergyAISDKError(f"Tool '{tool_name}' version '{version}' not found in registry")

        try:
            # Check cache first
            cache_key = f"{tool_name}_plugin"
            if cache_key in self._tool_cache:
                plugin = self._tool_cache[cache_key]
                kernel.add_plugin(plugin)
                self.logger.debug(f"Using cached tool: {tool_name}")
                return

            # Convert tool schema to OpenAPI and create plugin
            openapi_spec = self._convert_to_openapi(tool_def)
            plugin = await kernel_function_from_openapi(
                plugin_name=tool_def.name,
                openapi_document=openapi_spec,
                execution_settings=None,
            )

            # Cache and add to kernel
            self._tool_cache[cache_key] = plugin
            kernel.add_plugin(plugin)
            
            self.logger.info(f"Successfully loaded tool: {tool_name}")

        except Exception as e:
            self.logger.error(f"Failed to load tool {tool_name}: {e}")
            raise


    def clear_tool_cache(self, tool_name: Optional[str] = None):
        """Clear tool cache."""
        if tool_name:
            cache_key = f"{tool_name}_plugin"
            if cache_key in self._tool_cache:
                del self._tool_cache[cache_key]
                self.logger.info(f"Cleared tool cache for: {tool_name}")
        else:
            self._tool_cache.clear()
            self.logger.info("Cleared all tool cache")

    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "cached_agents": len(self._kernel_cache),
            "cached_tools": len(self._tool_cache),
            "agent_names": list(self._kernel_cache.keys()),
            "tool_names": list(self._tool_cache.keys()),
            "registry_available": self.registry_client is not None,
            "semantic_kernel_available": SEMANTIC_KERNEL_AVAILABLE,
        }

    async def get_available_agents(self) -> List[str]:
        """Get list of available agent names from the registry."""
        if not self.registry_client:
            return []

        try:
            agent_defs = await self.registry_client.list_agents()
            return [agent.name for agent in agent_defs]
        except Exception as e:
            self.logger.error(f"Failed to get available agents: {e}")
            return []

    async def get_available_tools(self) -> List[str]:
        """Get list of available tool names from the registry."""
        if not self.registry_client:
            return []

        try:
            tool_defs = await self.registry_client.list_tools()
            return [tool.name for tool in tool_defs]
        except Exception as e:
            self.logger.error(f"Failed to get available tools: {e}")
            return []


    def _convert_to_openapi(self, tool_def: ToolDefinition) -> str:
        """
        Convert a tool definition to a valid OpenAPI v3 specification.

        This is the critical data transformation that maps our custom tool schema
        to the OpenAPI standard required by Semantic Kernel.

        Args:
            tool_def: Tool definition to convert

        Returns:
            OpenAPI specification as JSON string
        """
        try:
            # Extract function details from tool schema
            tool_schema = tool_def.schema
            
            # Handle different schema formats
            if "function" in tool_schema:
                # Tool Call format: {"type": "function", "function": {...}}
                func_def = tool_schema["function"]
                func_name = func_def.get("name", tool_def.name)
                description = func_def.get("description", tool_def.description)
                parameters = func_def.get("parameters", {})
            else:
                # Direct schema format
                func_name = tool_def.name
                description = tool_def.description
                parameters = tool_schema

            # Build OpenAPI specification
            openapi_spec = {
                "openapi": "3.0.0",
                "info": {
                    "title": tool_def.name,
                    "description": description,
                    "version": tool_def.version,
                },
                "servers": [],
                "paths": {
                    f"/{func_name}": {
                        "post": {
                            "operationId": func_name,
                            "summary": description,
                            "description": description,
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": parameters
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Successful operation",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "description": "Tool execution result"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            # Add endpoint URL if available
            if tool_def.endpoint_url:
                openapi_spec["servers"] = [{"url": tool_def.endpoint_url}]

            # Add authentication if specified
            if tool_def.auth_config:
                auth_config = tool_def.auth_config
                if auth_config.get("type") == "bearer":
                    openapi_spec["components"] = {
                        "securitySchemes": {
                            "bearerAuth": {
                                "type": "http",
                                "scheme": "bearer"
                            }
                        }
                    }
                    openapi_spec["paths"][f"/{func_name}"]["post"]["security"] = [
                        {"bearerAuth": []}
                    ]
                elif auth_config.get("type") == "api_key":
                    openapi_spec["components"] = {
                        "securitySchemes": {
                            "apiKeyAuth": {
                                "type": "apiKey",
                                "in": auth_config.get("in", "header"),
                                "name": auth_config.get("name", "X-API-Key")
                            }
                        }
                    }
                    openapi_spec["paths"][f"/{func_name}"]["post"]["security"] = [
                        {"apiKeyAuth": []}
                    ]

            openapi_json = json.dumps(openapi_spec, indent=2)
            self.logger.debug(f"Generated OpenAPI spec for {tool_def.name}: {len(openapi_json)} chars")
            
            return openapi_json

        except Exception as e:
            self.logger.error(f"Failed to convert tool {tool_def.name} to OpenAPI: {e}")
            raise EnergyAISDKError(f"OpenAPI conversion failed: {e}") from e





    def clear_cache(self) -> None:
        """Clear the kernel cache."""
        self._kernel_cache.clear()
        self.logger.info("Kernel cache cleared")

    def get_cached_agents(self) -> List[str]:
        """Get list of agents with cached kernels."""
        return list(self._kernel_cache.keys())
