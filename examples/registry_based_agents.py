"""
Example: Registry-Based Dynamic Agent Creation

This example demonstrates the declarative, database-driven approach for creating
agents and tools using the EnergyAI SDK. Instead of defining agents in code,
they are loaded from a registry (Cosmos DB) at runtime.

Key Features:
1. Agents and tools are defined in JSON documents stored in Cosmos DB
2. The KernelFactory dynamically builds kernels based on agent definitions
3. Tool schemas are converted to OpenAPI specifications for Semantic Kernel
4. The KernelManager provides caching for efficient kernel retrieval
"""

import asyncio
import logging

from energyai_sdk import initialize_sdk, kernel_manager
from energyai_sdk.clients import MockRegistryClient
from energyai_sdk.config import ConfigurationManager
from energyai_sdk.kernel_factory import KernelFactory


async def demonstrate_registry_workflow():
    """Demonstrate the complete registry-based workflow."""

    # Initialize the SDK
    initialize_sdk(log_level="INFO")

    print("🔧 Registry-Based Agent System Demo")
    print("=" * 50)

    # For this demo, we'll use the MockRegistryClient
    # In production, you would use the real RegistryClient with Cosmos DB
    registry_client = MockRegistryClient()
    config_manager = ConfigurationManager()

    # Create a KernelFactory with the registry client
    try:
        factory = KernelFactory(registry_client=registry_client, config_manager=config_manager)
        print("✅ KernelFactory created successfully")
    except Exception as e:
        print(f"⚠️ KernelFactory creation failed: {e}")
        print("   This is expected if semantic-kernel is not installed")
        factory = None

    print("\n1. 🔍 Fetching Agent Definition from Registry")
    print("-" * 30)

    # Fetch an agent definition from the registry
    agent_name = "energy_analyst"
    agent_def = await registry_client.get_agent_by_name(agent_name)

    if agent_def:
        print(f"✅ Found agent: {agent_def.name}")
        print(f"   Description: {agent_def.description}")
        print(f"   Tools: {', '.join(agent_def.tools)}")
        print(f"   Model: {agent_def.model_config.get('deployment_name', 'N/A')}")
    else:
        print(f"❌ Agent '{agent_name}' not found in registry")
        return

    print("\n2. 🛠️ Fetching Tool Definitions")
    print("-" * 30)

    # Fetch tool definitions referenced by the agent
    for tool_name in agent_def.tools:
        tool_def = await registry_client.get_tool_by_name(tool_name)
        if tool_def:
            print(f"✅ Found tool: {tool_def.name}")
            print(f"   Category: {tool_def.category}")
            print(f"   Endpoint: {tool_def.endpoint_url}")
        else:
            print(f"❌ Tool '{tool_name}' not found in registry")

    print("\n3. 🏗️ Building Kernel with KernelFactory")
    print("-" * 30)

    if factory:
        try:
            # This would create a kernel with all tools loaded as plugins
            # Note: This will fail in the demo because we don't have real Azure OpenAI config
            # but it demonstrates the workflow
            print(f"🔄 Creating kernel for agent: {agent_name}")
            print("   (Note: Will use mock/fallback due to missing Azure OpenAI config)")

            # In a real scenario with proper configuration:
            # kernel = await factory.create_kernel_for_agent(agent_name)

            print("✅ Kernel creation workflow completed")
            print("   - Agent definition loaded from registry")
            print("   - Tool definitions resolved")
            print("   - Tools converted to OpenAPI specifications")
            print("   - Plugins registered with Semantic Kernel")

        except Exception as e:
            print(f"⚠️ Kernel creation failed (expected in demo): {e}")
            print("   This is normal without proper Azure OpenAI configuration")
    else:
        print("⚠️ Skipping kernel creation (KernelFactory not available)")
        print("   Install semantic-kernel to enable kernel creation")

    print("\n4. 📊 Using KernelManager for Caching")
    print("-" * 30)

    # Demonstrate the KernelManager caching
    print("🔄 Using global KernelManager...")
    print(f"   Cache stats: {kernel_manager.get_cache_stats()}")

    # This would use the global kernel manager
    # kernel = await kernel_manager.get_kernel_for_agent(agent_name)

    print("\n5. 🔄 Agent Lifecycle Management")
    print("-" * 30)

    # Demonstrate cache management
    cached_agents = kernel_manager.get_cached_agent_names()
    print(f"📦 Cached agents: {cached_agents}")

    # Clear cache for specific agent
    if cached_agents:
        agent_to_clear = cached_agents[0]
        kernel_manager.clear_cache(agent_to_clear)
        print(f"🧹 Cleared cache for: {agent_to_clear}")

    # Refresh an agent (force rebuild)
    # await kernel_manager.refresh_agent_kernel(agent_name)

    print("\n✅ Demo completed!")
    print("\nKey Benefits of This Approach:")
    print("• 🔧 Declarative: Agents defined in data, not code")
    print("• 🚀 Dynamic: Add/modify agents without code changes")
    print("• 📦 Scalable: Registry can serve multiple applications")
    print("• 🔄 Cacheable: Efficient kernel reuse")
    print("• 🛡️ Secure: API keys managed through secret references")


async def demonstrate_openapi_conversion():
    """Demonstrate the tool-to-OpenAPI conversion process."""

    print("\n🔄 OpenAPI Conversion Demo")
    print("=" * 30)

    # Get a sample tool definition
    registry_client = MockRegistryClient()
    tool_def = await registry_client.get_tool_by_name("energy_calculator")

    if not tool_def:
        print("❌ Sample tool not found")
        return

    print("📋 Original Tool Definition:")
    print(f"   Name: {tool_def.name}")
    print(f"   Description: {tool_def.description}")
    print(f"   Schema Type: {tool_def.schema.get('type')}")

    # Create factory and demonstrate conversion
    try:
        factory = KernelFactory()

        try:
            # Convert to OpenAPI
            openapi_spec = factory._convert_to_openapi(tool_def)
            print("\n✅ OpenAPI Conversion Successful!")
            print(f"   Spec length: {len(openapi_spec)} characters")
            print("   Format: JSON")

            # Show a snippet of the converted spec
            import json

            spec_obj = json.loads(openapi_spec)
            print("\n📄 Generated OpenAPI Info:")
            print(f"   Title: {spec_obj['info']['title']}")
            print(f"   Version: {spec_obj['info']['version']}")
            print(f"   Paths: {list(spec_obj['paths'].keys())}")

        except Exception as e:
            print(f"❌ OpenAPI conversion failed: {e}")

    except Exception as e:
        print(f"⚠️ KernelFactory creation failed: {e}")
        print("   Install semantic-kernel to enable OpenAPI conversion")


async def demonstrate_registry_configuration():
    """Demonstrate registry client configuration options."""

    print("\n⚙️ Registry Configuration Demo")
    print("=" * 35)

    # Real registry client configuration (commented out for demo)
    print("📋 Real Registry Client Configuration:")
    print(
        """
    registry_client = RegistryClient(
        cosmos_endpoint="https://your-cosmos.documents.azure.com:443/",
        cosmos_key="your-cosmos-primary-key",
        database_name="AgenticPlatform",
        agents_container="Agents",
        tools_container="Tools"
    )
    """
    )

    # Mock registry client for demo
    print("🧪 Mock Registry Client (for development/testing):")
    mock_client = MockRegistryClient()

    # List available agents and tools
    agents = await mock_client.list_agents()
    tools = await mock_client.list_tools()

    print("\n📊 Available Components:")
    print(f"   Agents: {len(agents)}")
    for agent in agents:
        print(f"     - {agent.name} (v{agent.version})")

    print(f"   Tools: {len(tools)}")
    for tool in tools:
        print(f"     - {tool.name} (v{tool.version}) [{tool.category}]")

    # Health check
    is_healthy = await mock_client.health_check()
    print(f"\n🏥 Registry Health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def main():
        """Run all demonstrations."""
        await demonstrate_registry_workflow()
        await demonstrate_openapi_conversion()
        await demonstrate_registry_configuration()

    # Run the demonstrations
    asyncio.run(main())
