"""
Example: EnergyAI Application with Unified Monitoring Integration

This example demonstrates how to create an EnergyAI application with comprehensive
monitoring using the new unified MonitoringClient that integrates both Langfuse
and OpenTelemetry for complete observability into agent interactions and LLM calls.
"""

import asyncio
import logging
import os
from datetime import datetime

from fastapi import BackgroundTasks

from energyai_sdk import agent, tool
from energyai_sdk.application import ChatRequest, create_application

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define energy calculation tools
@tool(name="calculate_renewable_savings", description="Calculate savings from renewable energy")
def calculate_renewable_savings(current_bill: float, renewable_percentage: float = 0.3) -> dict:
    """Calculate potential savings from renewable energy adoption."""
    renewable_savings = current_bill * renewable_percentage * 0.85  # 85% efficiency
    annual_savings = renewable_savings * 12
    co2_reduction = renewable_savings * 0.5  # kg CO2 per dollar saved

    return {
        "monthly_savings": round(renewable_savings, 2),
        "annual_savings": round(annual_savings, 2),
        "co2_reduction_kg": round(co2_reduction, 2),
        "renewable_percentage": renewable_percentage,
        "efficiency_factor": 0.85,
        "currency": "USD",
    }


@tool(
    name="energy_audit_recommendations", description="Get personalized energy audit recommendations"
)
def energy_audit_recommendations(home_type: str = "apartment", square_feet: int = 1000) -> dict:
    """Generate personalized energy audit recommendations."""
    base_recommendations = [
        "Install LED lighting throughout",
        "Upgrade to smart thermostat",
        "Seal air leaks around windows",
        "Add insulation to attic",
    ]

    home_specific = {
        "house": ["Install solar panels", "Upgrade HVAC system", "Energy-efficient windows"],
        "apartment": ["Use smart power strips", "Energy-efficient appliances", "Window treatments"],
        "condo": ["Balcony solar panels", "Smart water heater", "Energy monitoring system"],
    }

    cost_per_sqft = {"house": 15, "apartment": 8, "condo": 12}
    estimated_cost = square_feet * cost_per_sqft.get(home_type, 10)
    potential_savings = estimated_cost * 0.2  # 20% annual savings

    return {
        "home_type": home_type,
        "square_feet": square_feet,
        "basic_recommendations": base_recommendations,
        "specific_recommendations": home_specific.get(home_type, []),
        "estimated_cost": estimated_cost,
        "potential_annual_savings": round(potential_savings, 2),
        "payback_period_years": (
            round(estimated_cost / potential_savings, 1) if potential_savings > 0 else "N/A"
        ),
    }


# Define energy advisor agent with Langfuse monitoring
@agent(
    name="EnergyAdvisorMonitored",
    description="Energy efficiency advisor with comprehensive monitoring",
    system_prompt="""You are an expert energy efficiency advisor helping users reduce their energy consumption and costs.

    You provide actionable, personalized advice based on user needs and circumstances. Use the available tools
    to calculate savings and provide recommendations.

    Always be encouraging and focus on practical steps users can take immediately. Reference any previous
    conversation context to provide personalized follow-up advice.

    When discussing renewable energy, emphasize both environmental and financial benefits.""",
    tools=["calculate_renewable_savings", "energy_audit_recommendations"],
)
class EnergyAdvisorMonitored:
    """Energy efficiency advisor with Langfuse monitoring capabilities."""

    def get_description(self) -> str:
        return "I provide personalized energy efficiency advice with comprehensive monitoring and analytics."


async def demonstrate_unified_monitoring():
    """Demonstrate unified monitoring integration with the EnergyAI application."""

    print("🔍 EnergyAI Unified Monitoring Demo")
    print("=" * 40)

    # Set up mock credentials for demo
    # In production, these would be real credentials
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk_demo_key_for_testing")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk_demo_key_for_testing")

    try:
        # Create application with unified monitoring enabled
        print("📊 Creating EnergyAI application with unified monitoring...")
        app = create_application(
            title="EnergyAI Monitoring Platform",
            debug=True,
            enable_context_store=True,
            enable_observability=True,  # Enable unified monitoring
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host="https://cloud.langfuse.com",
            langfuse_environment="development",
        )

        print(f"✅ Application created: {app.title}")
        print(f"✅ Context store enabled: {app.context_store_client is not None}")

        # Check monitoring client status
        from energyai_sdk.clients.monitoring import get_monitoring_client

        monitoring_client = get_monitoring_client()
        print(f"✅ Unified monitoring enabled: {monitoring_client is not None}")
        if monitoring_client:
            health = monitoring_client.health_check()
            print(f"✅ Monitoring health: {health}")

        # Initialize the app
        await app._startup()
        print(f"✅ Application started. Components: {app.components_status}")

        # Simulate monitored conversations
        session_id = f"monitoring_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "demo_user_monitoring"

        print(f"\n🗣️  Starting monitored conversation session: {session_id}")
        print(f"👤 User: {user_id}")

        # Conversation scenarios to demonstrate different types of monitoring
        conversation_scenarios = [
            {
                "message": "Hi! I want to reduce my energy bills. My current monthly bill is $150.",
                "description": "Initial request with context",
                "expected_tools": ["calculate_renewable_savings"],
            },
            {
                "message": "That sounds great! I live in a 1200 sq ft house. What specific improvements should I make?",
                "description": "Follow-up requesting detailed recommendations",
                "expected_tools": ["energy_audit_recommendations"],
            },
            {
                "message": "How much would solar panels save me specifically? I'm interested in 50% renewable energy.",
                "description": "Specific calculation request with parameters",
                "expected_tools": ["calculate_renewable_savings"],
            },
            {
                "message": "Thanks for all the advice! Can you summarize the key next steps for me?",
                "description": "Summary request using conversation history",
                "expected_tools": [],
            },
        ]

        print("\n" + "=" * 60)
        print("MONITORED CONVERSATION SIMULATION")
        print("=" * 60)

        for i, scenario in enumerate(conversation_scenarios, 1):
            print(f"\n🔄 Turn {i}: {scenario['description']}")
            print(f"👤 User: {scenario['message']}")

            # Create chat request
            request = ChatRequest(
                message=scenario["message"],
                agent_id="EnergyAdvisorMonitored",
                session_id=session_id,
                subject_id=user_id,
                temperature=0.7,
                max_tokens=500,
                metadata={
                    "scenario": scenario["description"],
                    "expected_tools": scenario["expected_tools"],
                },
            )

            try:
                # Process the request with Langfuse monitoring
                response = await app._process_chat_request(request, BackgroundTasks())

                print(f"🤖 EnergyAdvisor: {response.content}")
                print(f"⏱️  Response time: {response.execution_time_ms}ms")

                # Show monitoring status
                if monitoring_client:
                    print("📊 Monitoring trace created successfully")
                    print("   - Trace includes: session context, agent invocation, tool calls")
                    print("   - Generation tracked with model parameters and usage")
                    print("   - Error handling and performance metrics captured")
                    print("   - Both Langfuse and OpenTelemetry data collected")
                else:
                    print("⚠️  Unified monitoring not available (expected for demo)")

                # Show session stats
                if app.context_store_client:
                    try:
                        history = app.context_store_client.get_conversation_history(
                            session_id, user_id
                        )
                        print(f"📈 Session stats: {len(history)} messages in history")
                    except:
                        print("📈 Session stats: Context store not available")

            except Exception as e:
                print(f"❌ Error in conversation turn: {e}")
                # Error would also be captured in Langfuse trace

            print("-" * 50)

        # Demonstrate monitoring features
        print("\n📊 UNIFIED MONITORING FEATURES")
        print("=" * 50)

        monitoring_features = [
            "🎯 Dual Tracing: Each conversation tracked in both Langfuse and OpenTelemetry",
            "🔄 Session Tracking: Multi-turn conversations linked by session_id",
            "🤖 LLM Generation Monitoring: Model calls tracked with inputs/outputs (Langfuse)",
            "⚡ Performance Metrics: Response times and system metrics (OpenTelemetry)",
            "🛠️  Tool Usage: Function calls and results monitored",
            "❌ Error Tracking: Exceptions captured in both systems",
            "🏷️  Rich Metadata: Context and tags for filtering and analysis",
            "📈 Dual Analytics: Langfuse for LLM insights, OpenTelemetry for system metrics",
            "🔧 Easy Configuration: Single config for both monitoring systems",
            "🧪 Mock Support: Full testing capabilities with MockMonitoringClient",
        ]

        for feature in monitoring_features:
            print(f"   {feature}")

        # Close the session
        if app.context_store_client:
            try:
                print(f"\n🔒 Closing monitored session {session_id}")
                app.context_store_client.close_session(session_id, user_id)
            except:
                print("🔒 Session closure not available (context store disabled)")

        # Shutdown application
        await app._shutdown()
        print("\n✅ Application shutdown complete")

    except Exception as e:
        print(f"❌ Monitoring demo failed: {e}")
        import traceback

        traceback.print_exc()


async def demonstrate_monitoring_configuration():
    """Demonstrate different unified monitoring configuration options."""

    print("\n⚙️  Unified Monitoring Configuration Demo")
    print("=" * 45)

    # Show different ways to configure unified monitoring
    configurations = [
        {
            "name": "Development Configuration (Langfuse Only)",
            "params": {
                "enable_observability": True,
                "langfuse_host": "https://cloud.langfuse.com",
                "langfuse_environment": "development",
                "debug": True,
            },
        },
        {
            "name": "Production Configuration (Langfuse + Azure Monitor)",
            "params": {
                "enable_observability": True,
                "langfuse_host": "https://cloud.langfuse.com",
                "langfuse_environment": "production",
                "azure_monitor_connection_string": "InstrumentationKey=your-key",
                "debug": False,
            },
        },
        {
            "name": "OpenTelemetry Only Configuration",
            "params": {
                "enable_observability": True,
                "langfuse_public_key": None,  # Disable Langfuse
                "langfuse_secret_key": None,
                "azure_monitor_connection_string": "InstrumentationKey=your-key",
                "debug": False,
            },
        },
    ]

    for config in configurations:
        print(f"\n📋 {config['name']}:")
        for key, value in config["params"].items():
            print(f"   {key}: {value}")

        # Show how to create app with this config
        print("   Usage:")
        print("   app = create_application(")
        for key, value in config["params"].items():
            if isinstance(value, str):
                print(f'       {key}="{value}",')
            elif value is None:
                print(f"       {key}=None,")
            else:
                print(f"       {key}={value},")
        if config["params"].get("langfuse_public_key") is not None:
            print("       langfuse_public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),")
            print("       langfuse_secret_key=os.getenv('LANGFUSE_SECRET_KEY')")
        print("   )")

    print("\n💡 Environment Variables:")
    print("   LANGFUSE_PUBLIC_KEY: Your Langfuse public key")
    print("   LANGFUSE_SECRET_KEY: Your Langfuse secret key")
    print("   LANGFUSE_HOST: Langfuse server URL (optional)")
    print("   AZURE_MONITOR_CONNECTION_STRING: Azure Application Insights connection string")
    print("   OTLP_ENDPOINT: Custom OpenTelemetry collector endpoint")

    print("\n🔧 Direct MonitoringClient Usage:")
    print("   from energyai_sdk.clients.monitoring import MonitoringClient, MonitoringConfig")
    print("   ")
    print("   config = MonitoringConfig(")
    print("       enable_langfuse=True,")
    print("       langfuse_public_key='your-key',")
    print("       langfuse_secret_key='your-secret',")
    print("       enable_opentelemetry=True,")
    print("       azure_monitor_connection_string='InstrumentationKey=your-key'")
    print("   )")
    print("   ")
    print("   monitoring_client = MonitoringClient(config)")
    print("   health = monitoring_client.health_check()")
    print("   print(f'Monitoring health: {health}')")


if __name__ == "__main__":
    """
    Run the unified monitoring demonstration.

    This example shows:
    1. How to create an application with unified monitoring (Langfuse + OpenTelemetry)
    2. How conversations are traced and monitored in both systems
    3. How agent invocations and tool calls are captured
    4. How errors and performance metrics are tracked
    5. How to configure monitoring for different environments

    Setup for production use:
    1. Get credentials from monitoring providers:
       - Langfuse: https://cloud.langfuse.com
       - Azure Monitor: https://portal.azure.com

    2. Set environment variables:
       - LANGFUSE_PUBLIC_KEY: Your Langfuse public key
       - LANGFUSE_SECRET_KEY: Your Langfuse secret key
       - AZURE_MONITOR_CONNECTION_STRING: Your Azure connection string
       - OTLP_ENDPOINT: Custom OpenTelemetry collector endpoint (optional)

    3. Enable monitoring in your application:
       app = create_application(
           enable_observability=True,
           langfuse_public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
           langfuse_secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
           azure_monitor_connection_string=os.getenv('AZURE_MONITOR_CONNECTION_STRING')
       )

    4. Use monitoring dashboards to analyze:
       - Langfuse: LLM conversations, agent flows, generation costs
       - Azure Monitor/OpenTelemetry: System performance, errors, metrics
       - Combined insights for comprehensive observability
    """

    async def run_demos():
        await demonstrate_unified_monitoring()
        await demonstrate_monitoring_configuration()

    asyncio.run(run_demos())
