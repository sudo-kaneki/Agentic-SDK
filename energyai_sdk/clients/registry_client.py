"""
Registry Client for Agentic Registry in Cosmos DB.

This client handles fetching agent and tool definitions from the external registry,
enabling dynamic agent and tool creation at runtime.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from azure.cosmos import exceptions as cosmos_exceptions
    from azure.cosmos.aio import CosmosClient

    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False

from ..exceptions import EnergyAISDKError


@dataclass
class ToolDefinition:
    """Tool definition from the registry."""

    id: str
    name: str
    description: str
    category: str
    schema: Dict[str, Any]
    endpoint_url: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class AgentDefinition:
    """Agent definition from the registry."""

    id: str
    name: str
    description: str
    system_prompt: str
    model_config: Dict[str, Any]
    tools: List[str]
    capabilities: List[str]
    temperature: float = 0.7
    max_tokens: int = 1000
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.capabilities is None:
            self.capabilities = []


class RegistryClient:
    """
    Client for the Agentic Registry in Cosmos DB.

    Handles fetching agent and tool definitions from the centralized registry,
    enabling dynamic agent creation and tool loading at runtime.
    """

    def __init__(
        self,
        cosmos_endpoint: str,
        cosmos_key: str,
        database_name: str = "energyai_platform",
        agents_container: str = "agents",
        tools_container: str = "tools",
    ):
        """
        Initialize the Registry Client.

        Args:
            cosmos_endpoint: Cosmos DB endpoint URL
            cosmos_key: Cosmos DB primary key
            database_name: Database name (default: energyai_platform)
            agents_container: Agents container name (default: agents)
            tools_container: Tools container name (default: tools)
        """
        if not COSMOS_AVAILABLE:
            raise EnergyAISDKError(
                "Azure Cosmos DB SDK not available. Install with: pip install azure-cosmos"
            )

        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        self.database_name = database_name
        self.agents_container = agents_container
        self.tools_container = tools_container

        self.logger = logging.getLogger(__name__)
        self._client = None
        self._database = None
        self._agents_container = None
        self._tools_container = None

        # Cache for frequently accessed definitions
        self._tool_cache: Dict[str, ToolDefinition] = {}
        self._agent_cache: Dict[str, AgentDefinition] = {}
        self._cache_ttl = 300  # 5 minutes

    async def _get_client(self):
        """Get or create Cosmos DB client."""
        if self._client is None:
            self._client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
            self._database = self._client.get_database_client(self.database_name)
            self._agents_container = self._database.get_container_client(self.agents_container)
            self._tools_container = self._database.get_container_client(self.tools_container)

        return self._client

    async def get_agent_by_name(self, name: str) -> Optional[AgentDefinition]:
        """
        Fetch an agent definition by name from the registry.

        Args:
            name: The agent name

        Returns:
            AgentDefinition if found, None otherwise
        """
        return await self.get_agent_definition(name)

    async def get_tool_by_name(self, name: str, version: str = "1.0.0") -> Optional[ToolDefinition]:
        """
        Fetch a tool definition by name and version from the registry.

        Args:
            name: The tool name
            version: The tool version (default: "1.0.0")

        Returns:
            ToolDefinition if found, None otherwise
        """
        try:
            await self._get_client()

            # Query by name and version
            query = "SELECT * FROM c WHERE c.name = @name AND c.version = @version"
            parameters = [{"name": "@name", "value": name}, {"name": "@version", "value": version}]

            results = []
            async for item in self._tools_container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            ):
                tool_def = ToolDefinition(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    category=item["category"],
                    schema=item["schema"],
                    endpoint_url=item.get("endpoint_url"),
                    auth_config=item.get("auth_config"),
                    version=item.get("version", "1.0.0"),
                    tags=item.get("tags", []),
                    created_at=(
                        datetime.fromisoformat(item["created_at"])
                        if item.get("created_at")
                        else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(item["updated_at"])
                        if item.get("updated_at")
                        else None
                    ),
                )
                results.append(tool_def)
                break  # Return first match

            if results:
                self.logger.info(f"Retrieved tool by name: {name} v{version}")
                return results[0]
            else:
                self.logger.warning(f"Tool not found: {name} v{version}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching tool by name {name} v{version}: {e}")
            raise EnergyAISDKError(f"Failed to fetch tool by name: {e}") from e

    async def get_tool_definition(self, tool_id: str) -> Optional[ToolDefinition]:
        """
        Fetch a tool definition by ID from the registry.

        Args:
            tool_id: The tool identifier

        Returns:
            ToolDefinition if found, None otherwise
        """
        try:
            # Check cache first
            if tool_id in self._tool_cache:
                cached_tool = self._tool_cache[tool_id]
                # Simple cache expiration (in production, use proper TTL)
                if (
                    cached_tool.created_at
                    and (datetime.now(timezone.utc) - cached_tool.created_at).seconds
                    < self._cache_ttl
                ):
                    return cached_tool

            await self._get_client()

            # Query Cosmos DB
            response = await self._tools_container.read_item(item=tool_id, partition_key=tool_id)

            tool_def = ToolDefinition(
                id=response["id"],
                name=response["name"],
                description=response["description"],
                category=response["category"],
                schema=response["schema"],
                endpoint_url=response.get("endpoint_url"),
                auth_config=response.get("auth_config"),
                version=response.get("version", "1.0.0"),
                tags=response.get("tags", []),
                created_at=(
                    datetime.fromisoformat(response["created_at"])
                    if response.get("created_at")
                    else None
                ),
                updated_at=(
                    datetime.fromisoformat(response["updated_at"])
                    if response.get("updated_at")
                    else None
                ),
            )

            # Cache the result
            self._tool_cache[tool_id] = tool_def

            self.logger.info(f"Retrieved tool definition: {tool_id}")
            return tool_def

        except cosmos_exceptions.CosmosResourceNotFoundError:
            self.logger.warning(f"Tool not found in registry: {tool_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching tool definition {tool_id}: {e}")
            raise EnergyAISDKError(f"Failed to fetch tool definition: {e}") from e

    async def get_agent_definition(self, agent_id: str) -> Optional[AgentDefinition]:
        """
        Fetch an agent definition by ID from the registry.

        Args:
            agent_id: The agent identifier

        Returns:
            AgentDefinition if found, None otherwise
        """
        try:
            # Check cache first
            if agent_id in self._agent_cache:
                cached_agent = self._agent_cache[agent_id]
                if (
                    cached_agent.created_at
                    and (datetime.now(timezone.utc) - cached_agent.created_at).seconds
                    < self._cache_ttl
                ):
                    return cached_agent

            await self._get_client()

            # Query Cosmos DB
            response = await self._agents_container.read_item(item=agent_id, partition_key=agent_id)

            agent_def = AgentDefinition(
                id=response["id"],
                name=response["name"],
                description=response["description"],
                system_prompt=response["system_prompt"],
                model_config=response["model_config"],
                tools=response.get("tools", []),
                capabilities=response.get("capabilities", []),
                temperature=response.get("temperature", 0.7),
                max_tokens=response.get("max_tokens", 1000),
                version=response.get("version", "1.0.0"),
                tags=response.get("tags", []),
                created_at=(
                    datetime.fromisoformat(response["created_at"])
                    if response.get("created_at")
                    else None
                ),
                updated_at=(
                    datetime.fromisoformat(response["updated_at"])
                    if response.get("updated_at")
                    else None
                ),
            )

            # Cache the result
            self._agent_cache[agent_id] = agent_def

            self.logger.info(f"Retrieved agent definition: {agent_id}")
            return agent_def

        except cosmos_exceptions.CosmosResourceNotFoundError:
            self.logger.warning(f"Agent not found in registry: {agent_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching agent definition {agent_id}: {e}")
            raise EnergyAISDKError(f"Failed to fetch agent definition: {e}") from e

    async def list_tools(
        self, category: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 100
    ) -> List[ToolDefinition]:
        """
        List available tools from the registry.

        Args:
            category: Filter by tool category
            tags: Filter by tags (any match)
            limit: Maximum number of tools to return

        Returns:
            List of ToolDefinition objects
        """
        try:
            await self._get_client()

            # Build query
            query = "SELECT * FROM c"
            parameters = []

            conditions = []
            if category:
                conditions.append("c.category = @category")
                parameters.append({"name": "@category", "value": category})

            if tags:
                tag_conditions = []
                for i, tag in enumerate(tags):
                    param_name = f"@tag{i}"
                    tag_conditions.append(f"ARRAY_CONTAINS(c.tags, {param_name})")
                    parameters.append({"name": param_name, "value": tag})
                conditions.append(f"({' OR '.join(tag_conditions)})")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += f" ORDER BY c.created_at DESC OFFSET 0 LIMIT {limit}"

            tools = []
            async for item in self._tools_container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            ):
                tool_def = ToolDefinition(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    category=item["category"],
                    schema=item["schema"],
                    endpoint_url=item.get("endpoint_url"),
                    auth_config=item.get("auth_config"),
                    version=item.get("version", "1.0.0"),
                    tags=item.get("tags", []),
                    created_at=(
                        datetime.fromisoformat(item["created_at"])
                        if item.get("created_at")
                        else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(item["updated_at"])
                        if item.get("updated_at")
                        else None
                    ),
                )
                tools.append(tool_def)

            self.logger.info(f"Retrieved {len(tools)} tools from registry")
            return tools

        except Exception as e:
            self.logger.error(f"Error listing tools: {e}")
            raise EnergyAISDKError(f"Failed to list tools: {e}") from e

    async def list_agents(
        self, tags: Optional[List[str]] = None, limit: int = 100
    ) -> List[AgentDefinition]:
        """
        List available agents from the registry.

        Args:
            tags: Filter by tags (any match)
            limit: Maximum number of agents to return

        Returns:
            List of AgentDefinition objects
        """
        try:
            await self._get_client()

            # Build query
            query = "SELECT * FROM c"
            parameters = []

            if tags:
                tag_conditions = []
                for i, tag in enumerate(tags):
                    param_name = f"@tag{i}"
                    tag_conditions.append(f"ARRAY_CONTAINS(c.tags, {param_name})")
                    parameters.append({"name": param_name, "value": tag})
                query += " WHERE " + " OR ".join(tag_conditions)

            query += f" ORDER BY c.created_at DESC OFFSET 0 LIMIT {limit}"

            agents = []
            async for item in self._agents_container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            ):
                agent_def = AgentDefinition(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    system_prompt=item["system_prompt"],
                    model_config=item["model_config"],
                    tools=item.get("tools", []),
                    capabilities=item.get("capabilities", []),
                    temperature=item.get("temperature", 0.7),
                    max_tokens=item.get("max_tokens", 1000),
                    version=item.get("version", "1.0.0"),
                    tags=item.get("tags", []),
                    created_at=(
                        datetime.fromisoformat(item["created_at"])
                        if item.get("created_at")
                        else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(item["updated_at"])
                        if item.get("updated_at")
                        else None
                    ),
                )
                agents.append(agent_def)

            self.logger.info(f"Retrieved {len(agents)} agents from registry")
            return agents

        except Exception as e:
            self.logger.error(f"Error listing agents: {e}")
            raise EnergyAISDKError(f"Failed to list agents: {e}") from e

    async def health_check(self) -> bool:
        """
        Check if the registry service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self._get_client()

            # Simple query to test connectivity
            query = "SELECT VALUE COUNT(1) FROM c"
            result = [
                item
                async for item in self._tools_container.query_items(
                    query=query, enable_cross_partition_query=True
                )
            ]

            self.logger.info("Registry health check passed")
            return True

        except Exception as e:
            self.logger.error(f"Registry health check failed: {e}")
            return False

    async def close(self):
        """Close the Cosmos DB client."""
        if self._client:
            await self._client.close()
            self._client = None
            self._database = None
            self._agents_container = None
            self._tools_container = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Mock client for development/testing
class MockRegistryClient(RegistryClient):
    """Mock registry client for development and testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._tool_cache = {}
        self._agent_cache = {}

    async def get_agent_by_name(self, name: str) -> Optional[AgentDefinition]:
        """Mock agent fetching by name."""
        return await self.get_agent_definition(name)

    async def get_tool_by_name(self, name: str, version: str = "1.0.0") -> Optional[ToolDefinition]:
        """Mock tool fetching by name and version."""
        # For mock, we'll map name to tool_id
        tool_id_map = {
            "energy_calculator": "energy_calculator",
            "carbon_calculator": "carbon_calculator",
        }
        tool_id = tool_id_map.get(name)
        if tool_id:
            return await self.get_tool_definition(tool_id)
        return None

    async def get_tool_definition(self, tool_id: str) -> Optional[ToolDefinition]:
        """Mock tool fetching with sample data."""
        sample_tools = {
            "energy_calculator": ToolDefinition(
                id="energy_calculator",
                name="Energy Calculator",
                description="Calculate energy metrics for renewable projects",
                category="energy",
                schema={
                    "type": "function",
                    "function": {
                        "name": "calculate_lcoe",
                        "description": "Calculate Levelized Cost of Energy",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "capex": {"type": "number", "description": "Capital expenditure"},
                                "opex": {"type": "number", "description": "Operating expenditure"},
                                "generation": {
                                    "type": "number",
                                    "description": "Annual generation MWh",
                                },
                            },
                            "required": ["capex", "opex", "generation"],
                        },
                    },
                },
                endpoint_url="https://api.energyai.com/tools/calculator",
                version="1.2.0",
                tags=["energy", "finance", "renewables"],
            )
        }

        return sample_tools.get(tool_id)

    async def get_agent_definition(self, agent_id: str) -> Optional[AgentDefinition]:
        """Mock agent fetching with sample data."""
        sample_agents = {
            "energy_analyst": AgentDefinition(
                id="energy_analyst",
                name="Energy Analyst",
                description="Expert energy analyst for renewable projects",
                system_prompt="You are an expert energy analyst specializing in renewable energy projects.",
                model_config={"deployment_name": "gpt-4", "temperature": 0.3},
                tools=["energy_calculator", "carbon_calculator"],
                capabilities=["financial_analysis", "technical_analysis"],
                temperature=0.3,
                max_tokens=2000,
                tags=["energy", "analysis", "finance"],
            )
        }

        return sample_agents.get(agent_id)

    async def list_tools(
        self, category: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 100
    ) -> List[ToolDefinition]:
        """Mock tool listing."""
        return [await self.get_tool_definition("energy_calculator")]

    async def list_agents(
        self, tags: Optional[List[str]] = None, limit: int = 100
    ) -> List[AgentDefinition]:
        """Mock agent listing."""
        return [await self.get_agent_definition("energy_analyst")]

    async def health_check(self) -> bool:
        """Mock health check."""
        return True

    async def close(self):
        """Mock close."""
        pass
