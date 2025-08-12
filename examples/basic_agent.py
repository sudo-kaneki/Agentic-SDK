# examples/basic_agent.py
"""
Basic Agent Example

This example demonstrates how to create a simple agent with tools
using the decorator approach and run it in a development server.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path to import energyai_sdk
sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk import AgentRequest, agent, initialize_sdk, tool
from energyai_sdk.agents import bootstrap_agents

# Try to import optional components
try:
    from energyai_sdk.application import create_application, run_development_server
    from energyai_sdk.clients import (
        MockContextStoreClient,
        MockMonitoringClient,
        MockRegistryClient,
    )

    APPLICATION_AVAILABLE = True
except ImportError:
    APPLICATION_AVAILABLE = False
    print("Note: Application module not available. Agent creation will work but no web server.")


# Create a simple calculation tool
@tool(name="simple_calculator", description="Perform basic mathematical calculations")
def simple_calculator(operation: str, a: float, b: float) -> dict:
    """
    Perform basic mathematical operations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        Dictionary with calculation result
    """
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else None,
    }

    result = operations.get(operation.lower())

    return {
        "operation": operation,
        "operand_a": a,
        "operand_b": b,
        "result": result,
        "success": result is not None,
        "error": "Division by zero" if operation.lower() == "divide" and b == 0 else None,
    }


# Create an energy unit converter tool
@tool(name="energy_unit_converter", description="Convert between different energy units")
def energy_unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert energy values between different units.

    Args:
        value: The energy value to convert
        from_unit: Source unit (kwh, mwh, gwh, btu, joules)
        to_unit: Target unit (kwh, mwh, gwh, btu, joules)

    Returns:
        Dictionary with conversion result
    """
    # Conversion factors to kWh (base unit)
    to_kwh = {
        "kwh": 1.0,
        "mwh": 1000.0,
        "gwh": 1_000_000.0,
        "btu": 0.000293071,  # BTU to kWh
        "joules": 2.77778e-7,  # Joules to kWh
    }

    from_unit_lower = from_unit.lower()
    to_unit_lower = to_unit.lower()

    if from_unit_lower not in to_kwh or to_unit_lower not in to_kwh:
        return {
            "error": f"Unsupported unit. Supported: {list(to_kwh.keys())}",
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": None,
            "success": False,
        }

    # Convert to kWh, then to target unit
    kwh_value = value * to_kwh[from_unit_lower]
    result = kwh_value / to_kwh[to_unit_lower]

    return {
        "original_value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": round(result, 6),
        "success": True,
        "conversion_factor": to_kwh[from_unit_lower] / to_kwh[to_unit_lower],
    }


# Create a basic agent using the decorator approach
@agent(
    name="BasicAssistant",
    description="A helpful assistant with basic calculation and energy conversion capabilities",
    system_prompt="""You are a helpful assistant with access to calculation and energy conversion tools.

    You can:
    - Perform basic mathematical calculations (add, subtract, multiply, divide)
    - Convert between energy units (MWh, GWh, kWh, BTU, Joules)

    When users ask for calculations or conversions, use the appropriate tools and explain the results clearly.
    Always show your work and provide context for the calculations.""",
    tools=["simple_calculator", "energy_unit_converter"],
)
class BasicAssistant:
    """A helpful assistant with calculation and energy conversion capabilities."""

    temperature = 0.7
    max_tokens = 1000


def create_basic_agent_config():
    """Create the configuration for the basic agent."""
    return {
        "deployment_name": "gpt-4o",
        "endpoint": "https://your-endpoint.openai.azure.com/",
        "api_key": "your-api-key-here",
        "api_version": "2024-02-01",
    }


async def test_agent_locally():
    """Test the agent locally without starting a server."""

    # Initialize SDK with debug logging
    initialize_sdk(log_level="DEBUG")

    print("🧪 Testing Basic Agent Tools Directly (No AI needed)")
    print("=" * 60)

    # Test tools directly first
    print("\n📊 Testing Calculator Tool:")
    calc_result = simple_calculator("add", 15, 25)
    print(f"15 + 25 = {calc_result}")

    print("\n⚡ Testing Energy Converter Tool:")
    conv_result = energy_unit_converter(100, "GWh", "MWh")
    print(f"100 GWh = {conv_result}")

    print("\n🤖 For AI-powered responses, configure your API keys and run:")
    print("python basic_agent.py --mode ai-test")


async def test_with_ai():
    """Test the agent with AI capabilities."""

    # Initialize SDK with debug logging
    initialize_sdk(log_level="DEBUG")

    # Configure Azure OpenAI (replace with your credentials)
    azure_config = create_basic_agent_config()

    print("🤖 Testing Basic Agent with AI capabilities...")
    print("=" * 60)
    print("⚠️  Note: This requires valid Azure OpenAI credentials")

    try:
        # Bootstrap the agent
        agents = bootstrap_agents(azure_openai_config=azure_config)
        agent = agents.get("BasicAssistant")

        if not agent:
            print("❌ Agent not created. Check your configuration.")
            return

        print("✅ Agent created successfully!")

        # Test cases
        test_cases = [
            {"message": "Calculate 15 + 25", "description": "Basic addition"},
            {"message": "Convert 100 GWh to MWh", "description": "Energy unit conversion"},
            {
                "message": "What's 50 divided by 10, and then convert the result from MWh to kWh?",
                "description": "Combined calculation and conversion",
            },
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['description']}")
            print(f"Input: {test_case['message']}")

            try:
                request = AgentRequest(
                    message=test_case["message"],
                    agent_id="BasicAssistant",
                    session_id=f"test_session_{i}",
                )

                response = await agent.process_request(request)

                print(f"Response: {response.content}")
                print(f"Execution time: {response.execution_time_ms}ms")

                if response.error:
                    print(f"Error: {response.error}")

            except Exception as e:
                print(f"Error processing request: {e}")

            print("-" * 40)

    except Exception as e:
        print(f"❌ Error: {e}")
        print(
            "💡 Make sure to configure your Azure OpenAI credentials in create_basic_agent_config()"
        )


def run_development_mode():
    """Run the agent in development server mode with Azure integration demo."""

    if not APPLICATION_AVAILABLE:
        print("❌ Web server not available. Install FastAPI and uvicorn:")
        print("pip install fastapi uvicorn")
        return

    # Initialize SDK
    initialize_sdk(log_level="DEBUG")

    print("🚀 Starting Basic Agent with Azure Integration Demo")
    print("=" * 60)

    # Configure Azure OpenAI
    azure_config = create_basic_agent_config()

    try:
        # Bootstrap agents
        agents = bootstrap_agents(azure_openai_config=azure_config)

        if "BasicAssistant" not in agents:
            print("❌ BasicAssistant not created. Check configuration.")
            return

        # Create Azure service clients (using mock for demo)
        print("🔧 Setting up Azure service integration (mock clients for demo)...")
        registry_client = MockRegistryClient()
        context_store_client = MockContextStoreClient()
        monitoring_client = MockMonitoringClient()

        # Create application with Azure integration
        app = create_application(
            title="Basic Agent with Azure Integration",
            description="Demonstrates Azure integration features",
            registry_client=registry_client,
            context_store_client=context_store_client,
            monitoring_client=monitoring_client,
            debug=True,
        )

        # Add our agent
        for agent in agents.values():
            app.add_agent(agent)

        print("✅ Azure integration configured:")
        print("   📋 Registry Client: Ready (mock)")
        print("   💾 Context Store: Ready (mock)")
        print("   📊 Monitoring: Ready (mock)")

        print("\n🌐 Starting Development Server with Azure Features...")
        print("📍 Available at: http://localhost:8000")
        print("📚 API Docs at: http://localhost:8000/docs")
        print("\n🆕 New Azure Integration Endpoints:")
        print("   GET  /sessions/{session_id} - Retrieve session context")
        print("   POST /sessions/{session_id} - Create new session")
        print("   POST /registry/reload      - Reload from registry")
        print("   GET  /health              - Enhanced health check")

        print("\n🧪 Test session persistence:")
        print("   1. Send a chat message with 'session_id' parameter")
        print("   2. Check session context at /sessions/{session_id}")
        print("   3. Continue conversation - context will be preserved!")

        # Run development server with the enhanced app
        if hasattr(app, "get_fastapi_app") and app.get_fastapi_app():
            import uvicorn

            uvicorn.run(app.get_fastapi_app(), host="127.0.0.1", port=8000, reload=True)
        else:
            # Fallback to basic server
            run_development_server(
                agents=list(agents.values()), host="127.0.0.1", port=8000, reload=True
            )

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Make sure your Azure OpenAI credentials are configured")
        print("💡 For production Azure integration, see examples/production_azure_platform.py")


def main():
    """Main function to run the example."""
    import argparse

    parser = argparse.ArgumentParser(description="Basic Agent Example")
    parser.add_argument(
        "--mode",
        choices=["test", "ai-test", "server"],
        default="test",
        help="Run mode: 'test' for tool testing, 'ai-test' for AI testing, 'server' for development server",
    )

    args = parser.parse_args()

    if args.mode == "test":
        # Run local tool tests (no AI needed)
        asyncio.run(test_agent_locally())
    elif args.mode == "ai-test":
        # Test with AI capabilities
        asyncio.run(test_with_ai())
    else:
        # Run development server
        try:
            run_development_mode()
        except KeyboardInterrupt:
            print("\n👋 Shutting down server...")
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            print("\n🔧 Make sure you have:")
            print("1. Set proper Azure OpenAI credentials in create_basic_agent_config()")
            print("2. Installed all required dependencies: pip install fastapi uvicorn")
            print("3. Check the Getting_started.md guide")


if __name__ == "__main__":
    main()
