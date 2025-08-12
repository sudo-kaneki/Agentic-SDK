"""
Example demonstrating Azure integration with EnergyAI SDK.

This example shows how to use the new external service clients:
- RegistryClient for fetching agent/tool definitions from Cosmos DB
- ContextStoreClient for session persistence
- MonitoringClient for OpenTelemetry observability

Usage:
    python examples/azure_integration_example.py
"""

import asyncio
from datetime import datetime, timezone

from energyai_sdk.application import create_production_application
from energyai_sdk.clients import (
    MockContextStoreClient,
    MockMonitoringClient,
    MockRegistryClient,
    MonitoringConfig,
)
from energyai_sdk.core import KernelFactory


async def demonstrate_registry_client():
    """Demonstrate using the Registry Client to fetch tools and agents."""
    print("🔧 Registry Client Example")
    print("=" * 50)

    # Use mock client for demo (replace with real client in production)
    async with MockRegistryClient() as registry_client:

        # List available tools
        print("📋 Listing available tools:")
        tools = await registry_client.list_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # Get specific tool
        print("\n🔍 Fetching specific tool:")
        energy_calc = await registry_client.get_tool_definition("energy_calculator")
        if energy_calc:
            print(f"  Tool: {energy_calc.name}")
            print(f"  Version: {energy_calc.version}")
            print(f"  Tags: {', '.join(energy_calc.tags)}")

        # List available agents
        print("\n🤖 Listing available agents:")
        agents = await registry_client.list_agents()
        for agent in agents:
            print(f"  - {agent.name}: {agent.description}")

        # Health check
        is_healthy = await registry_client.health_check()
        print(f"\n✅ Registry health: {'Healthy' if is_healthy else 'Unhealthy'}")


async def demonstrate_context_store():
    """Demonstrate using the Context Store for session persistence."""
    print("\n💾 Context Store Example")
    print("=" * 50)

    async with MockContextStoreClient() as context_store:

        # Create a new session
        session_id = "demo_session_123"
        subject_id = "user_456"

        print(f"🆕 Creating session: {session_id}")
        session_doc = await context_store.create_session(
            session_id=session_id,
            subject_id=subject_id,
            initial_context={"conversation_started": datetime.now(timezone.utc).isoformat()},
        )
        print(f"  Created at: {session_doc.created_at}")

        # Add some conversation context
        print("\n💬 Adding conversation messages:")
        await context_store.append_to_context(
            session_id=session_id,
            key="messages",
            value={
                "user": "What's the LCOE for a 100MW solar farm?",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        await context_store.append_to_context(
            session_id=session_id,
            key="messages",
            value={
                "assistant": "I'll calculate the LCOE for your 100MW solar farm...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Retrieve session
        retrieved_session = await context_store.get_session(session_id)
        if retrieved_session:
            print(f"  Messages in context: {len(retrieved_session.context.get('messages', []))}")

        # List sessions for subject
        print(f"\n📋 Sessions for subject {subject_id}:")
        sessions = await context_store.list_sessions_by_subject(subject_id)
        for session in sessions:
            print(f"  - Session {session.session_id}: {session.created_at}")

        # Health check
        is_healthy = await context_store.health_check()
        print(f"\n✅ Context store health: {'Healthy' if is_healthy else 'Unhealthy'}")


def demonstrate_monitoring():
    """Demonstrate the Monitoring Client for observability."""
    print("\n📊 Monitoring Client Example")
    print("=" * 50)

    # Initialize monitoring with mock client
    monitoring_config = MonitoringConfig(service_name="energyai-demo", environment="development")

    monitoring_client = MockMonitoringClient(monitoring_config)

    # Record some metrics
    print("📈 Recording metrics:")
    monitoring_client.record_metric("demo_counter", 1.0, {"operation": "test"})
    monitoring_client.record_metric("response_time", 150.0, {"endpoint": "/chat"})

    # Create a traced operation
    print("\n🔍 Creating traced operation:")
    with monitoring_client.start_span("demo_operation", service="demo") as span:
        # Simulate some work
        import time

        time.sleep(0.1)
        print("  Completed traced operation")

    # Show recorded data (mock client only)
    traces = monitoring_client.get_recorded_traces()
    metrics = monitoring_client.get_recorded_metrics()

    print(f"  Traces recorded: {len(traces)}")
    print(f"  Metrics recorded: {len(metrics)}")

    # Health check
    is_healthy = monitoring_client.health_check()
    print(f"\n✅ Monitoring health: {'Healthy' if is_healthy else 'Unhealthy'}")


async def demonstrate_dynamic_tool_loading():
    """Demonstrate dynamic tool loading from registry."""
    print("\n⚙️  Dynamic Tool Loading Example")
    print("=" * 50)

    # Create a kernel
    kernel = KernelFactory.create_kernel()
    if not kernel:
        print("❌ Semantic Kernel not available")
        return

    # Load tools from registry
    registry_client = MockRegistryClient()
    loaded_count = await KernelFactory.load_tools_from_registry(kernel, registry_client)

    print(f"🔧 Loaded {loaded_count} tools from registry")

    # List loaded functions
    try:
        plugins = kernel.plugins
        print("📦 Available plugins:")
        for plugin_name, plugin in plugins.items():
            print(f"  - {plugin_name}:")
            for func_name, func in plugin.functions.items():
                print(f"    - {func_name}")
    except Exception as e:
        print(f"  Note: Function listing requires Semantic Kernel: {e}")


async def demonstrate_production_app():
    """Demonstrate creating a production application with external clients."""
    print("\n🚀 Production Application Example")
    print("=" * 50)

    # For demo purposes, we'll use mock configuration
    # In production, you would use real Azure endpoints and keys

    try:
        app = create_production_application(
            api_keys=["demo_api_key_123"],
            # In production, use real values:
            # cosmos_endpoint="https://your-account.documents.azure.com:443/",
            # cosmos_key="your_cosmos_key",
            # azure_monitor_connection_string="InstrumentationKey=your_key",
            # otlp_endpoint="https://your-otlp-endpoint"
        )

        print("✅ Production application created successfully")
        print(f"  Title: {app.title}")
        print(f"  Components: {len(app.components_status)} configured")

        # Show component status (would need to start the app first)
        print("  External service integration: Ready")

    except Exception as e:
        print(f"❌ Error creating production app: {e}")
        print("  (This is expected in demo mode without real Azure credentials)")


async def main():
    """Run all demonstrations."""
    print("🌟 EnergyAI SDK - Azure Integration Demo")
    print("=" * 60)
    print("This demo shows the new external service integration features:")
    print("- Externalized Agentic Registry (Cosmos DB)")
    print("- Externalized Context Store (Cosmos DB)")
    print("- Integrated Observability (OpenTelemetry)")
    print("=" * 60)

    try:
        # Run all demonstrations
        await demonstrate_registry_client()
        await demonstrate_context_store()
        demonstrate_monitoring()
        await demonstrate_dynamic_tool_loading()
        await demonstrate_production_app()

        print("\n🎉 All demonstrations completed successfully!")
        print("\nNext Steps:")
        print("1. Set up real Azure Cosmos DB accounts")
        print("2. Configure OpenTelemetry endpoints")
        print("3. Update application configuration with real credentials")
        print("4. Deploy using create_production_application()")

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
