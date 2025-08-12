"""
Example: Stateful EnergyAI Application with Context Store Integration

This example demonstrates how to create an EnergyAI application with integrated
ContextStoreClient for stateful, multi-turn conversations.
"""

import asyncio
import logging
import os
from datetime import datetime

from energyai_sdk import agent, tool
from energyai_sdk.application import create_application

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define energy calculation tools
@tool(name="calculate_energy_cost", description="Calculate total energy cost")
def calculate_energy_cost(kwh: float, rate_per_kwh: float = 0.12) -> dict:
    """Calculate energy cost based on consumption and rate."""
    total_cost = kwh * rate_per_kwh
    savings_potential = total_cost * 0.15  # Assume 15% savings potential

    return {
        "kwh_consumed": kwh,
        "rate_per_kwh": rate_per_kwh,
        "total_cost": round(total_cost, 2),
        "estimated_savings": round(savings_potential, 2),
        "currency": "USD",
    }


@tool(name="get_efficiency_tips", description="Get energy efficiency recommendations")
def get_efficiency_tips(focus_area: str = "general") -> dict:
    """Get energy efficiency tips for specific areas."""
    tips_by_area = {
        "heating": [
            "Set thermostat to 68°F in winter",
            "Use programmable thermostats",
            "Seal air leaks around windows and doors",
            "Insulate attics and basements",
        ],
        "cooling": [
            "Set thermostat to 78°F in summer",
            "Use ceiling fans to circulate air",
            "Close blinds during hot days",
            "Maintain air conditioning units regularly",
        ],
        "lighting": [
            "Switch to LED bulbs",
            "Use natural light when possible",
            "Install motion sensors",
            "Turn off lights when leaving rooms",
        ],
        "general": [
            "Unplug devices when not in use",
            "Use energy-efficient appliances",
            "Wash clothes in cold water",
            "Air dry clothes instead of using dryer",
        ],
    }

    return {
        "focus_area": focus_area,
        "tips": tips_by_area.get(focus_area, tips_by_area["general"]),
        "potential_savings_percent": "10-30% depending on implementation",
    }


# Define energy consultant agent
@agent(
    name="EnergyConsultant",
    description="Energy efficiency expert providing personalized advice",
    system_prompt="""You are an expert energy consultant helping users optimize their energy usage.

    You provide practical, actionable advice based on user needs. When users ask about costs,
    use the calculate_energy_cost tool. When they need efficiency tips, use get_efficiency_tips.

    Be conversational and remember the context of your discussion. Reference previous
    conversations to provide personalized recommendations.""",
    tools=["calculate_energy_cost", "get_efficiency_tips"],
)
class EnergyConsultant:
    """Energy efficiency consultant agent with stateful conversations."""

    def get_description(self) -> str:
        return "I help optimize energy usage and reduce costs through personalized recommendations."


async def demonstrate_stateful_application():
    """Demonstrate the stateful application in action."""

    print("🏭 EnergyAI Stateful Application Demo")
    print("=" * 50)

    # Mock environment variables for demo (in production, set these properly)
    os.environ.setdefault("COSMOS_ENDPOINT", "https://demo-cosmos.documents.azure.com/")
    os.environ.setdefault("COSMOS_KEY", "demo_key_for_testing")

    try:
        # Create application with context store
        print("📱 Creating EnergyAI application with context store...")
        app = create_application(
            title="Energy Efficiency Platform", debug=True, enable_context_store=True
        )

        print(f"✅ Application created: {app.title}")
        print(f"✅ Context store enabled: {app.context_store_client is not None}")

        # Initialize the app
        await app._startup()
        print(f"✅ Application started. Status: {app.components_status}")

        # Simulate a multi-turn conversation
        session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "demo_user"

        print(f"\n🗣️  Starting conversation session: {session_id}")
        print(f"👤 User: {user_id}")

        # Conversation turns
        conversation_turns = [
            {
                "message": "Hi! I'm looking to reduce my energy bills. Can you help?",
                "description": "Initial greeting and request",
            },
            {
                "message": "I use about 800 kWh per month. How much am I spending?",
                "description": "Cost calculation request with context",
            },
            {
                "message": "That's expensive! What can I do to reduce heating costs specifically?",
                "description": "Follow-up question building on previous discussion",
            },
            {
                "message": "Great tips! If I implement these, how much could I save on my 800 kWh usage?",
                "description": "Final question referencing earlier conversation",
            },
        ]

        print("\n" + "=" * 60)
        print("CONVERSATION SIMULATION")
        print("=" * 60)

        for i, turn in enumerate(conversation_turns, 1):
            print(f"\n🔄 Turn {i}: {turn['description']}")
            print(f"👤 User: {turn['message']}")

            # Create chat request
            from energyai_sdk.application import ChatRequest

            request = ChatRequest(
                message=turn["message"],
                agent_id="EnergyConsultant",
                session_id=session_id,
                subject_id=user_id,
                temperature=0.7,
            )

            try:
                # Process the request (this will use context from previous turns)
                from fastapi import BackgroundTasks

                response = await app._process_chat_request(request, BackgroundTasks())

                print(f"🤖 EnergyConsultant: {response.content}")
                print(f"⏱️  Response time: {response.execution_time_ms}ms")

                # Show session stats
                if app.context_store_client:
                    history = app.context_store_client.get_conversation_history(session_id, user_id)
                    print(f"📊 Session stats: {len(history)} messages in history")

            except Exception as e:
                print(f"❌ Error in conversation turn: {e}")
                # Continue with next turn

            print("-" * 40)

        # Show complete conversation history
        if app.context_store_client:
            print("\n📚 COMPLETE CONVERSATION HISTORY")
            print("=" * 50)

            history = app.context_store_client.get_conversation_history(session_id, user_id)
            for i, msg in enumerate(history, 1):
                sender = msg.get("sender", "unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")[:19]  # Remove microseconds
                agent_name = msg.get("agent_name", "")

                if sender == "user":
                    print(f"{i:2d}. [{timestamp}] 👤 User: {content}")
                elif sender == "agent":
                    print(f"{i:2d}. [{timestamp}] 🤖 {agent_name}: {content}")

            # Close the session
            print(f"\n🔒 Closing session {session_id}")
            app.context_store_client.close_session(session_id, user_id)

        # Shutdown application
        await app._shutdown()
        print("\n✅ Application shutdown complete")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


async def demonstrate_api_integration():
    """Demonstrate how to use the stateful app via API endpoints."""

    print("\n🌐 API Integration Demo")
    print("=" * 30)

    try:
        # This would typically be done via HTTP requests in production
        app = create_application(enable_context_store=True)
        await app._startup()

        # Simulate API calls (in production these would be HTTP requests)
        session_id = "api_demo_session"
        subject_id = "api_user"

        print(f"📱 Creating session via API: {session_id}")

        # Simulate session creation
        if app.context_store_client:
            session_doc = app.context_store_client.load_or_create(session_id, subject_id)
            print(f"✅ Session created/loaded: {session_doc['id']}")

            # Simulate getting history
            history = app.context_store_client.get_conversation_history(session_id, subject_id)
            print(f"📊 Current history: {len(history)} messages")

        print("\n💡 In production, you would:")
        print("  POST /sessions/{session_id}?subject_id={subject_id}")
        print("  POST /chat (with session_id in request)")
        print("  GET /sessions/{session_id}/history")
        print("  POST /sessions/{session_id}/close")

        await app._shutdown()

    except Exception as e:
        print(f"❌ API demo failed: {e}")


if __name__ == "__main__":
    """
    Run the stateful application demonstration.

    This example shows:
    1. How to create an application with ContextStoreClient
    2. How conversations maintain context across turns
    3. How agents can reference previous conversation history
    4. How to manage sessions programmatically

    Setup for production use:
    1. Set environment variables:
       - COSMOS_ENDPOINT: Your Azure Cosmos DB endpoint
       - COSMOS_KEY: Your Azure Cosmos DB key
       - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
       - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key

    2. Ensure Cosmos DB has:
       - Database: 'AgenticPlatform'
       - Container: 'Context'
       - Partition key: '/subject/id'

    3. Deploy with FastAPI/uvicorn for HTTP API access
    """

    async def run_demos():
        await demonstrate_stateful_application()
        await demonstrate_api_integration()

    asyncio.run(run_demos())
