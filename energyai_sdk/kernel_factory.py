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
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.connectors.openapi_plugin.authentication.api_key_authentication_provider import (
        ApiKeyAuthenticationProvider,
    )

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
        self._kernel_cache: Dict[str, sk.Kernel] = {}

    async def create_kernel_for_agent(
        self, agent_name: str, force_rebuild: bool = False
    ) -> sk.Kernel:
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
            kernel = sk.Kernel()
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

    async def _add_llm_service(self, kernel: sk.Kernel, agent_def: AgentDefinition) -> None:
        """
        Add the LLM service to the kernel based on agent configuration.

        Args:
            kernel: Kernel to configure
            agent_def: Agent definition containing model configuration
        """
        try:
            model_config = agent_def.model_config

            # Get Azure OpenAI configuration from settings
            config = self.config_manager.get_config()

            # Create Azure Chat Completion service
            azure_chat_completion = AzureChatCompletion(
                deployment_name=model_config.get("deployment_name", "gpt-4"),
                endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
            )

            kernel.add_service(azure_chat_completion)

            self.logger.debug(
                f"Added LLM service for deployment: {model_config.get('deployment_name', 'gpt-4')}"
            )

        except Exception as e:
            self.logger.error(f"Failed to add LLM service: {e}")
            raise EnergyAISDKError(f"LLM service configuration failed: {e}") from e

    async def _load_and_register_tools(self, kernel: sk.Kernel, agent_def: AgentDefinition) -> None:
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
        self, kernel: sk.Kernel, tool_name: str, version: str = "1.0.0"
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

        # Check if this is an HTTP-based tool that can be converted to OpenAPI
        tool_schema = tool_def.schema
        if self._is_http_tool(tool_schema):
            await self._register_http_tool(kernel, tool_def)
        else:
            # For non-HTTP tools, we would need different registration logic
            self.logger.warning(
                f"Tool {tool_name} is not an HTTP tool, skipping OpenAPI registration"
            )

    def _is_http_tool(self, tool_schema: Dict[str, Any]) -> bool:
        """
        Check if a tool is HTTP-based and can be converted to OpenAPI.

        Args:
            tool_schema: Tool schema definition

        Returns:
            True if tool is HTTP-based, False otherwise
        """
        # This is a simplified check - in practice you'd have more sophisticated logic
        return (
            tool_schema.get("type") == "function"
            and "endpoint_url" in tool_schema.get("metadata", {})
        ) or ("openapi" in tool_schema or "http" in tool_schema.get("type", "").lower())

    async def _register_http_tool(self, kernel: sk.Kernel, tool_def: ToolDefinition) -> None:
        """
        Register an HTTP-based tool as an OpenAPI plugin.

        Args:
            kernel: Kernel to add plugin to
            tool_def: Tool definition to register
        """
        try:
            # Convert tool definition to OpenAPI specification
            openapi_spec = self._convert_to_openapi(tool_def)

            # Create authentication provider if needed
            auth_provider = None
            if tool_def.auth_config:
                auth_provider = self._create_auth_provider(tool_def.auth_config)

            # Register the plugin with Semantic Kernel
            await kernel.add_plugin_from_openapi(
                plugin_name=tool_def.name.lower().replace(" ", "_"),
                openapi_document=openapi_spec,
                auth_provider=auth_provider,
            )

            self.logger.info(f"Successfully registered HTTP tool: {tool_def.name}")

        except Exception as e:
            self.logger.error(f"Failed to register HTTP tool {tool_def.name}: {e}")
            raise

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
            # Extract base information
            tool_schema = tool_def.schema
            endpoint_url = tool_def.endpoint_url or tool_schema.get("metadata", {}).get(
                "endpoint_url"
            )

            if not endpoint_url:
                raise EnergyAISDKError(f"No endpoint URL found for tool: {tool_def.name}")

            # Build OpenAPI specification
            openapi_spec = {
                "openapi": "3.0.1",
                "info": {
                    "title": tool_def.name,
                    "description": tool_def.description,
                    "version": tool_def.version,
                },
                "servers": [
                    {
                        "url": endpoint_url,
                        "description": f"API server for {tool_def.name}",
                    }
                ],
                "paths": {},
                "components": {
                    "schemas": {},
                    "securitySchemes": {},
                },
            }

            # Convert function schema to OpenAPI path
            if tool_schema.get("type") == "function":
                self._add_function_to_openapi(openapi_spec, tool_schema["function"], tool_def)
            else:
                # Handle other schema types as needed
                self._add_generic_tool_to_openapi(openapi_spec, tool_schema, tool_def)

            # Add security schemes if authentication is configured
            if tool_def.auth_config:
                self._add_security_to_openapi(openapi_spec, tool_def.auth_config)

            return json.dumps(openapi_spec, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to convert tool {tool_def.name} to OpenAPI: {e}")
            raise EnergyAISDKError(f"OpenAPI conversion failed: {e}") from e

    def _add_function_to_openapi(
        self,
        openapi_spec: Dict[str, Any],
        function_schema: Dict[str, Any],
        tool_def: ToolDefinition,
    ) -> None:
        """
        Add a function-type tool to the OpenAPI specification.

        Args:
            openapi_spec: OpenAPI spec to modify
            function_schema: Function schema from tool definition
            tool_def: Complete tool definition
        """
        function_name = function_schema.get("name", tool_def.name.lower().replace(" ", "_"))
        parameters = function_schema.get("parameters", {})

        # Create path for the function (assuming POST method)
        path = f"/{function_name}"

        # Build request body schema from parameters
        request_schema = {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
        }

        # Add the path to OpenAPI spec
        openapi_spec["paths"][path] = {
            "post": {
                "operationId": function_name,
                "summary": function_schema.get("description", tool_def.description),
                "description": function_schema.get("description", tool_def.description),
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": request_schema}},
                },
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "result": {"type": "string"},
                                        "data": {"type": "object"},
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Bad request"},
                    "500": {"description": "Internal server error"},
                },
            }
        }

        # Add schema to components if it has properties
        if request_schema.get("properties"):
            schema_name = f"{function_name.title()}Request"
            openapi_spec["components"]["schemas"][schema_name] = request_schema

    def _add_generic_tool_to_openapi(
        self, openapi_spec: Dict[str, Any], tool_schema: Dict[str, Any], tool_def: ToolDefinition
    ) -> None:
        """
        Add a generic tool to the OpenAPI specification.

        Args:
            openapi_spec: OpenAPI spec to modify
            tool_schema: Tool schema from definition
            tool_def: Complete tool definition
        """
        # This is a fallback for non-function schemas
        # You would implement specific logic based on your tool schema format
        tool_name = tool_def.name.lower().replace(" ", "_")
        path = f"/{tool_name}"

        openapi_spec["paths"][path] = {
            "post": {
                "operationId": tool_name,
                "summary": tool_def.description,
                "description": tool_def.description,
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": tool_schema.get("parameters", {"type": "object"})
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        }

    def _add_security_to_openapi(
        self, openapi_spec: Dict[str, Any], auth_config: Dict[str, Any]
    ) -> None:
        """
        Add security schemes to the OpenAPI specification.

        Args:
            openapi_spec: OpenAPI spec to modify
            auth_config: Authentication configuration
        """
        auth_type = auth_config.get("type", "").lower()

        if auth_type == "api_key":
            scheme_name = "ApiKeyAuth"
            openapi_spec["components"]["securitySchemes"][scheme_name] = {
                "type": "apiKey",
                "in": auth_config.get("location", "header"),
                "name": auth_config.get("key_name", "X-API-Key"),
            }

            # Apply security to all operations
            for path_obj in openapi_spec["paths"].values():
                for method_obj in path_obj.values():
                    if "security" not in method_obj:
                        method_obj["security"] = []
                    method_obj["security"].append({scheme_name: []})

        elif auth_type == "bearer":
            scheme_name = "BearerAuth"
            openapi_spec["components"]["securitySchemes"][scheme_name] = {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": auth_config.get("format", "JWT"),
            }

            # Apply security to all operations
            for path_obj in openapi_spec["paths"].values():
                for method_obj in path_obj.values():
                    if "security" not in method_obj:
                        method_obj["security"] = []
                    method_obj["security"].append({scheme_name: []})

    def _create_auth_provider(self, auth_config: Dict[str, Any]) -> Optional[Any]:
        """
        Create an authentication provider based on the auth configuration.

        Args:
            auth_config: Authentication configuration from tool definition

        Returns:
            Authentication provider instance or None
        """
        auth_type = auth_config.get("type", "").lower()

        if auth_type == "api_key":
            # Get the API key from configuration or secret reference
            api_key = None

            if "secret_ref" in auth_config:
                # Load from secret reference (implement based on your secret management)
                secret_ref = auth_config["secret_ref"]
                api_key = self._get_secret_value(secret_ref)
            elif "value" in auth_config:
                # Direct value (not recommended for production)
                api_key = auth_config["value"]

            if api_key and SEMANTIC_KERNEL_AVAILABLE:
                from semantic_kernel.connectors.openapi_plugin.authentication.api_key_authentication_provider import (
                    ApiKeyAuthenticationProvider,
                )

                return ApiKeyAuthenticationProvider(lambda: api_key)

        return None

    def _get_secret_value(self, secret_ref: str) -> Optional[str]:
        """
        Retrieve a secret value from the configured secret store.

        Args:
            secret_ref: Reference to the secret

        Returns:
            Secret value or None if not found
        """
        # This would integrate with your secret management system
        # For now, return None as a placeholder
        self.logger.warning(f"Secret reference not implemented: {secret_ref}")
        return None

    def clear_cache(self) -> None:
        """Clear the kernel cache."""
        self._kernel_cache.clear()
        self.logger.info("Kernel cache cleared")

    def get_cached_agents(self) -> List[str]:
        """Get list of agents with cached kernels."""
        return list(self._kernel_cache.keys())
