"""
Example: Unified Monitoring in EnergyAI SDK

This example demonstrates how to use the unified MonitoringClient
to monitor and trace agent interactions, including:

1. Langfuse for LLM-specific observability
2. OpenTelemetry for general application monitoring
3. Azure Monitor integration
"""

import asyncio
import logging
import os
from datetime import datetime

from energyai_sdk import agent, monitor, tool
from energyai_sdk.application import create_application
from energyai_sdk.clients.monitoring import get_monitoring_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define energy calculation tools
@tool(name="calculate_energy_savings", description="Calculate potential energy savings")
@monitor("tool.calculate_energy_savings")  # Add monitoring to track tool performance
def calculate_energy_savings(current_usage: float, efficiency_factor: float = 0.25) -> dict:
    """Calculate potential energy savings based on efficiency improvements."""
    savings = current_usage * efficiency_factor
    co2_reduction = savings * 0.4  # kg CO2 per kWh saved

    return {
        "original_usage": current_usage,
        "savings_kwh": savings,
        "savings_percentage": efficiency_factor * 100,
        "co2_reduction_kg": co2_reduction,
        "annual_savings_kwh": savings * 12,  # Assuming monthly usage
    }


# Define energy advisor agent
@agent(
    name="EnergyAdvisor",
    description="Energy efficiency advisor with comprehensive monitoring",
    system_prompt="""You are an expert energy advisor helping users optimize their energy usage.

    Provide practical advice based on the user's energy consumption data and needs.
    Use the available tools to calculate potential savings and environmental impact.

    Always be encouraging and focus on actionable recommendations.""",
    tools=["calculate_energy_savings"],
)
class EnergyAdvisor:
    """Energy advisor agent with comprehensive monitoring."""

    @monitor("agent.get_advice")  # Monitor this method to track performance
    def get_advice(self, usage_data: dict) -> str:
        """Generate personalized energy advice."""
        total_usage = sum(usage_data.values())
        highest_usage = max(usage_data.items(), key=lambda x: x[1])

        return (
            f"Based on your total usage of {total_usage} kWh, focusing on {highest_usage[0]} "
            f"which accounts for {highest_usage[1]} kWh would yield the best savings."
        )


async def demonstrate_unified_monitoring():
    """Demonstrate the unified monitoring system."""

    print("🔭 EnergyAI SDK - Unified Monitoring Demo")
    print("=" * 50)

    # Set up environment variables for demonstration
    # In production, these would be properly configured
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk_demo_key")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk_demo_key")
    os.environ.setdefault("AZURE_MONITOR_CONNECTION_STRING", "InstrumentationKey=demo-key")

    # Create application with unified monitoring
    print("\n📊 Creating application with unified monitoring...")
    app = create_application(
        title="Energy Advisor Platform",
        enable_observability=True,  # Enable the unified monitoring system
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        azure_monitor_connection_string=os.getenv("AZURE_MONITOR_CONNECTION_STRING"),
    )

    # Initialize the application
    await app._startup()

    # Get the monitoring client
    monitoring_client = get_monitoring_client()
    if monitoring_client:
        print(f"✅ Monitoring client initialized: {type(monitoring_client).__name__}")
        health = monitoring_client.health_check()
        print(f"✅ Monitoring health: {health}")
        print(
            f"✅ Langfuse enabled: {hasattr(monitoring_client, 'langfuse_client') and monitoring_client.langfuse_client is not None}"
        )
        print(
            f"✅ OpenTelemetry enabled: {hasattr(monitoring_client, 'otel_client') and monitoring_client.otel_client is not None}"
        )
    else:
        print("❌ Monitoring client not initialized")

    # Demonstrate tracing with the unified system
    print("\n🔍 Creating a trace for user interaction...")

    # Create a trace for the entire user interaction
    trace = None
    if monitoring_client:
        trace = monitoring_client.create_trace(
            name="energy-advisor-session",
            user_id="demo-user-123",
            session_id=f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            metadata={"demo": True, "scenario": "energy_savings_calculation"},
        )
        if trace:
            print("✅ Trace created successfully")
        else:
            print("⚠️  Trace creation skipped (Langfuse not configured)")

    # Simulate a user query with tracing
    print("\n👤 User: How much could I save if I improve my home's energy efficiency?")

    # Create a generation for the LLM call
    generation = None
    if trace and monitoring_client:
        generation = monitoring_client.create_generation(
            trace,
            name="energy-advisor-response",
            input_data={"query": "How much could I save if I improve my home's energy efficiency?"},
            model="gpt-4",
            model_parameters={"temperature": 0.7, "max_tokens": 300},
        )
        if generation:
            print("✅ Generation created for LLM call")
        else:
            print("⚠️  Generation creation skipped (Langfuse not configured)")

    # Simulate the agent's response
    response = (
        "Based on average home energy usage, you could save approximately 25% on your "
        "energy bills by implementing efficiency improvements like better insulation, "
        "LED lighting, and smart thermostats. Let me calculate the exact savings for you."
    )

    print(f"🤖 EnergyAdvisor: {response}")

    # End the generation with the response
    if generation and monitoring_client:
        monitoring_client.end_generation(
            generation,
            output=response,
            usage={"prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225},
        )
        print("✅ Generation ended with response")

    # Simulate a tool call with span tracing
    print("\n🧮 Calculating potential savings...")

    # Create a span for the tool execution
    with (
        monitoring_client.start_span(
            "tool.calculate_energy_savings", tool_name="calculate_energy_savings"
        )
        if monitoring_client
        else nullcontext()
    ):
        # Simulate tool execution
        savings = calculate_energy_savings(1000, 0.25)
        print(f"📊 Savings calculation: {savings}")

    # Update the trace with final results
    if trace and monitoring_client:
        monitoring_client.update_trace(
            trace,
            output="Energy savings calculation completed successfully",
            metadata={"savings_calculated": True, "savings_kwh": savings["savings_kwh"]},
        )
        print("✅ Trace updated with final results")

    # Flush telemetry data
    if monitoring_client:
        monitoring_client.flush()
        print("✅ Telemetry data flushed")

    # Shutdown the application
    await app._shutdown()
    print("\n✅ Application shutdown complete")


# Helper context manager for when observability is not available
class nullcontext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


async def demonstrate_monitoring_decorator():
    """Demonstrate the @monitor decorator with the unified system."""

    print("\n🔄 Monitoring Decorator Demo")
    print("=" * 30)

    @monitor("demo.monitored_function")
    async def monitored_function(param: str):
        """A function monitored with the @monitor decorator."""
        print(f"Executing monitored function with param: {param}")
        await asyncio.sleep(0.5)  # Simulate work
        return f"Result for {param}"

    print("Calling monitored function...")
    result = await monitored_function("test_param")
    print(f"Result: {result}")

    print("\n✅ Monitoring decorator demonstration complete")


if __name__ == "__main__":
    """
    Run the unified monitoring demonstration.

    This example shows:
    1. How to configure the unified monitoring system (Langfuse + OpenTelemetry)
    2. How to use Langfuse for LLM-specific monitoring
    3. How to use OpenTelemetry for general application monitoring
    4. How to use the @monitor decorator for function-level monitoring

    Setup for production use:
    1. Set environment variables:
       - LANGFUSE_PUBLIC_KEY: Your Langfuse public key
       - LANGFUSE_SECRET_KEY: Your Langfuse secret key
       - AZURE_MONITOR_CONNECTION_STRING: Your Azure Monitor connection string

    2. Enable monitoring in your application:
       app = create_application(
           enable_observability=True,
           langfuse_public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
           langfuse_secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
           azure_monitor_connection_string=os.getenv('AZURE_MONITOR_CONNECTION_STRING'),
       )

    3. Access the monitoring client:
       monitoring_client = get_monitoring_client()

    4. Use the monitoring client to create traces, spans, etc.:
       trace = monitoring_client.create_trace(...)
       generation = monitoring_client.create_generation(...)
       with monitoring_client.start_span(...):
           # Do work
    """

    async def run_demos():
        await demonstrate_unified_monitoring()
        await demonstrate_monitoring_decorator()

    asyncio.run(run_demos())
