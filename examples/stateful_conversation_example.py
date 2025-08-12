"""
Example: Stateful Conversations with ContextStoreClient

This example demonstrates how to use the ContextStoreClient to maintain
conversation history across multiple interactions with agents.
"""

import asyncio
import uuid
from energyai_sdk import agent, tool, bootstrap_agents
from energyai_sdk.clients import ContextStoreClient
from energyai_sdk.config import ConfigurationManager


# Define some simple tools
@tool
def calculate_energy_cost(kwh: float, rate_per_kwh: float = 0.12) -> dict:
    """Calculate energy cost based on consumption."""
    total_cost = kwh * rate_per_kwh
    return {
        "kwh_consumed": kwh,
        "rate_per_kwh": rate_per_kwh,
        "total_cost": round(total_cost, 2)
    }


@tool
def get_energy_tips() -> dict:
    """Get energy saving tips."""
    tips = [
        "Use LED bulbs instead of incandescent",
        "Unplug devices when not in use",
        "Use a programmable thermostat",
        "Seal air leaks around windows and doors",
        "Use energy-efficient appliances"
    ]
    return {"tips": tips}


# Define an energy consultant agent
@agent(name="EnergyConsultant", tools=["calculate_energy_cost", "get_energy_tips"])
class EnergyConsultant:
    """Energy efficiency consultant agent."""
    
    def get_description(self) -> str:
        return "I help with energy efficiency questions and calculations."


class StatefulConversationManager:
    """
    Manager for stateful conversations using ContextStoreClient.
    """
    
    def __init__(self):
        """Initialize the conversation manager."""
        try:
            # Initialize configuration (modify as needed for your setup)
            self.config_manager = ConfigurationManager()
            
            # Initialize ContextStoreClient
            self.context_client = ContextStoreClient(self.config_manager)
            
            # Bootstrap agents
            self.agents = bootstrap_agents({
                "deployment_name": "gpt-4o",
                "endpoint": "your-azure-openai-endpoint",
                "api_key": "your-api-key"
            })
            
            print("✅ Stateful conversation manager initialized")
            
        except Exception as e:
            print(f"❌ Error initializing: {e}")
            # For demo purposes, we'll continue without Cosmos DB
            self.context_client = None
            self.agents = {}
    
    async def handle_conversation_turn(
        self, 
        session_id: str, 
        user_id: str, 
        user_input: str,
        agent_name: str = "EnergyConsultant"
    ) -> dict:
        """
        Handle a single turn in a conversation with context persistence.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            user_input: User's message
            agent_name: Name of the agent to use
            
        Returns:
            Dictionary with response and session info
        """
        try:
            # Load or create session context
            if self.context_client:
                context_doc = self.context_client.load_or_create(session_id, user_id)
                
                # Get conversation history for context
                history = context_doc.get("thread", [])
                
                # Build context from recent history (last 10 messages)
                recent_history = history[-10:] if len(history) > 10 else history
                context_str = self._build_context_string(recent_history)
                
                print(f"📚 Loaded session {session_id} with {len(history)} messages")
            else:
                context_doc = None
                context_str = "No previous context available."
                print("⚠️ Running without persistent context")
            
            # Prepare the input with context
            contextual_input = f"""
Previous conversation context:
{context_str}

Current user message: {user_input}

Please respond naturally, taking into account the conversation history.
"""
            
            # Get agent response (simulated for demo)
            if agent_name in self.agents:
                # In a real implementation, you would call the agent here
                agent_response = await self._simulate_agent_response(
                    agent_name, contextual_input, user_input
                )
            else:
                agent_response = f"Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}"
            
            # Save the conversation turn
            if self.context_client and context_doc:
                updated_doc = self.context_client.update_and_save(
                    context_doc, user_input, agent_response, agent_name
                )
                
                session_info = {
                    "session_id": session_id,
                    "total_messages": len(updated_doc["thread"]),
                    "last_updated": updated_doc["updated_at"],
                    "persistent": True
                }
            else:
                session_info = {
                    "session_id": session_id,
                    "total_messages": "N/A",
                    "last_updated": "N/A",
                    "persistent": False
                }
            
            return {
                "response": agent_response,
                "session_info": session_info,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error in conversation turn: {e}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "session_info": {"error": str(e)},
                "success": False
            }
    
    def _build_context_string(self, history: list) -> str:
        """Build a context string from conversation history."""
        if not history:
            return "This is the start of our conversation."
        
        context_parts = []
        for msg in history:
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")
            agent_name = msg.get("agent_name", "")
            
            if sender == "user":
                context_parts.append(f"User: {content}")
            elif sender == "agent":
                if agent_name:
                    context_parts.append(f"{agent_name}: {content}")
                else:
                    context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts[-5:])  # Last 5 exchanges
    
    async def _simulate_agent_response(
        self, agent_name: str, contextual_input: str, user_input: str
    ) -> str:
        """
        Simulate agent response (replace with actual agent call).
        
        In a real implementation, this would call the actual agent with the
        contextual input and return the response.
        """
        # Simple keyword-based responses for demo
        user_lower = user_input.lower()
        
        if "cost" in user_lower or "calculate" in user_lower:
            if "100" in user_input:
                result = calculate_energy_cost(100)
                return f"Based on 100 kWh consumption: ${result['total_cost']} at ${result['rate_per_kwh']}/kWh"
            else:
                return "I can help calculate energy costs. Please provide your kWh consumption."
        
        elif "tips" in user_lower or "save" in user_lower:
            tips = get_energy_tips()
            return f"Here are some energy saving tips:\n" + "\n".join(f"• {tip}" for tip in tips["tips"][:3])
        
        elif "hello" in user_lower or "hi" in user_lower:
            return "Hello! I'm your Energy Consultant. I can help with energy calculations and efficiency tips."
        
        else:
            return "I'm here to help with energy efficiency questions. You can ask about costs, calculations, or energy saving tips!"
    
    def get_session_history(self, session_id: str, user_id: str) -> list:
        """Get conversation history for a session."""
        if not self.context_client:
            return []
        
        try:
            return self.context_client.get_conversation_history(session_id, user_id)
        except Exception as e:
            print(f"❌ Error retrieving history: {e}")
            return []
    
    def close_session(self, session_id: str, user_id: str) -> bool:
        """Close a conversation session."""
        if not self.context_client:
            return False
        
        try:
            self.context_client.close_session(session_id, user_id)
            print(f"✅ Session {session_id} closed")
            return True
        except Exception as e:
            print(f"❌ Error closing session: {e}")
            return False


async def demo_stateful_conversation():
    """Demonstrate stateful conversation capabilities."""
    print("🚀 EnergyAI SDK - Stateful Conversation Demo")
    print("=" * 50)
    
    # Initialize conversation manager
    manager = StatefulConversationManager()
    
    # Generate a session ID for this demo
    session_id = str(uuid.uuid4())[:8]
    user_id = "demo_user"
    
    print(f"📱 Starting session: {session_id}")
    print(f"👤 User: {user_id}")
    print()
    
    # Simulate a multi-turn conversation
    conversation_turns = [
        "Hi there! I'm new to energy efficiency.",
        "Can you give me some tips to save energy?",
        "That's helpful! How much would it cost if I use 100 kWh?",
        "What about 150 kWh?",
        "Thanks! Can you summarize what we discussed?"
    ]
    
    for i, user_input in enumerate(conversation_turns, 1):
        print(f"🔄 Turn {i}")
        print(f"👤 User: {user_input}")
        
        # Handle the conversation turn
        result = await manager.handle_conversation_turn(
            session_id, user_id, user_input
        )
        
        if result["success"]:
            print(f"🤖 Agent: {result['response']}")
            
            session_info = result["session_info"]
            if session_info.get("persistent"):
                print(f"📊 Session: {session_info['total_messages']} messages, "
                      f"updated: {session_info['last_updated'][:19]}")
            else:
                print("📊 Session: Running without persistence")
        else:
            print(f"❌ Error: {result['response']}")
        
        print("-" * 30)
    
    # Show conversation history
    print("\n📚 Conversation History:")
    history = manager.get_session_history(session_id, user_id)
    
    if history:
        for i, msg in enumerate(history, 1):
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
            timestamp = msg.get("timestamp", "")[:19]
            
            print(f"{i:2d}. [{timestamp}] {sender.title()}: {content}")
    else:
        print("No persistent history available")
    
    # Close the session
    print(f"\n🔒 Closing session {session_id}")
    manager.close_session(session_id, user_id)
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    """
    Run the stateful conversation demo.
    
    Setup required:
    1. Install azure-cosmos: pip install azure-cosmos
    2. Set environment variables:
       - COSMOS_ENDPOINT: Your Cosmos DB endpoint
       - COSMOS_KEY: Your Cosmos DB key
       - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
       - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    3. Ensure Cosmos DB has database 'AgenticPlatform' and container 'Context'
    """
    asyncio.run(demo_stateful_conversation())
