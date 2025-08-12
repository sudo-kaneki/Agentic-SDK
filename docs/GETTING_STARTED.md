# Getting Started with EnergyAI SDK 🚀

**Welcome to EnergyAI SDK!** This guide will take you from zero to creating your first AI agents in just 15 minutes.

## 📋 Prerequisites

- Python 3.8 or higher
- Basic Python knowledge (functions, classes, decorators)
- OpenAI or Azure OpenAI account (for actual AI functionality)

## 🛠️ Installation

### Step 1: Install Semantic Kernel
```bash
pip install semantic-kernel
```

### Step 2: Install EnergyAI SDK
```bash
# Clone the repository
git clone <repository-url>
cd energyai_sdk_project

# Install in development mode
pip install -e .
```

### Step 3: Verify Installation
```bash
python -c "from energyai_sdk import agent, tool; print('✅ EnergyAI SDK installed successfully!')"
```

## 🎯 Your First Agent (5 minutes)

Let's create your first AI agent that can perform basic calculations.

### Create `my_first_agent.py`:
```python
from energyai_sdk import agent, tool, bootstrap_agents, initialize_sdk

# Initialize the SDK
initialize_sdk(log_level="INFO")

# Step 1: Create a tool (function that the agent can use)
@tool(name="add_numbers", description="Add two numbers together")
def add_numbers(a: float, b: float) -> dict:
    """Add two numbers and return the result."""
    result = a + b
    return {
        "result": result,
        "operation": "addition",
        "operands": [a, b]
    }

@tool(name="multiply_numbers", description="Multiply two numbers")
def multiply_numbers(a: float, b: float) -> dict:
    """Multiply two numbers and return the result."""
    result = a * b
    return {
        "result": result,
        "operation": "multiplication",
        "operands": [a, b]
    }

# Step 2: Create an agent that can use these tools
@agent(
    name="MathAssistant",
    description="A helpful math assistant that can perform calculations",
    system_prompt="You are a friendly math assistant. Help users with calculations and explain your work.",
    tools=["add_numbers", "multiply_numbers"]  # Tools this agent can use
)
class MathAssistant:
    """A simple math assistant agent."""
    # Agent configuration
    temperature = 0.1  # Lower temperature for more consistent math
    max_tokens = 500

# Step 3: Test without AI (tools work independently)
print("🧪 Testing tools directly:")
print(f"5 + 3 = {add_numbers(5, 3)}")
print(f"4 × 6 = {multiply_numbers(4, 6)}")

print("\\n✅ Agent created successfully!")
print("\\n💡 To use with AI, configure your API keys and run bootstrap_agents()")
```

### Run it:
```bash
python my_first_agent.py
```

You should see:
```
🧪 Testing tools directly:
5 + 3 = {'result': 8.0, 'operation': 'addition', 'operands': [5.0, 3.0]}
4 × 6 = {'result': 24.0, 'operation': 'multiplication', 'operands': [4.0, 6.0]}

✅ Agent created successfully!
💡 To use with AI, configure your API keys and run bootstrap_agents()
```

## 🔑 Adding AI Capabilities

To make your agent actually intelligent, you need to configure AI services:

### Option 1: Azure OpenAI (Recommended)
```python
# Add this to your script after the agent definition:

# Configure Azure OpenAI
azure_config = {
    "deployment_name": "gpt-4o",  # Your deployment name
    "endpoint": "https://your-resource.openai.azure.com/",
    "api_key": "your-api-key-here",
    "api_version": "2024-02-01"
}

# Create the actual AI agent
try:
    agents = bootstrap_agents(azure_openai_config=azure_config)
    math_assistant = agents["MathAssistant"]
    print("🤖 AI agent created successfully!")

    # Test the agent (this requires async)
    # We'll show how to use this in the next section
except Exception as e:
    print(f"⚠️  AI not configured: {e}")
    print("Agent created but will work in non-AI mode for now.")
```

### Option 2: OpenAI
```python
openai_config = {
    "api_key": "your-openai-api-key-here",
    "model": "gpt-4"
}

agents = bootstrap_agents(openai_config=openai_config)
```

## 💬 Using Your Agent

### Create `chat_with_agent.py`:
```python
import asyncio
from energyai_sdk import agent, tool, bootstrap_agents, initialize_sdk, AgentRequest

# Initialize SDK
initialize_sdk(log_level="INFO")

# Define tools and agent (same as before)
@tool(name="add_numbers")
def add_numbers(a: float, b: float) -> dict:
    return {"result": a + b, "operation": "addition"}

@tool(name="multiply_numbers")
def multiply_numbers(a: float, b: float) -> dict:
    return {"result": a * b, "operation": "multiplication"}

@agent(
    name="MathAssistant",
    description="A helpful math assistant",
    system_prompt="You are a friendly math assistant. Use the available tools to help with calculations.",
    tools=["add_numbers", "multiply_numbers"]
)
class MathAssistant:
    temperature = 0.1
    max_tokens = 500

async def chat_with_math_assistant():
    """Chat with your AI agent."""

    # Configure your AI service
    config = {
        "deployment_name": "gpt-4o",
        "endpoint": "https://your-resource.openai.azure.com/",
        "api_key": "your-api-key"
    }

    try:
        # Create agents
        agents = bootstrap_agents(azure_openai_config=config)
        math_assistant = agents["MathAssistant"]

        # Create a request
        request = AgentRequest(
            message="Hi! Can you help me calculate what 15 plus 27 equals?",
            agent_id="MathAssistant"
        )

        # Get response
        response = await math_assistant.process_request(request)
        print(f"🤖 Agent: {response.content}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to configure your API keys!")

# Run the chat
if __name__ == "__main__":
    asyncio.run(chat_with_math_assistant())
```

## 🎭 Creating Skills (Tool Collections)

Skills group related tools together. Perfect for organizing functionality:

### Create `energy_skills.py`:
```python
from energyai_sdk import skill, tool, agent, bootstrap_agents, initialize_sdk

initialize_sdk(log_level="INFO")

# Create a skill with multiple related tools
@skill(name="EnergyCalculations", description="Tools for energy analysis")
class EnergyCalculations:
    """Collection of energy calculation tools."""

    @tool(name="calculate_lcoe", description="Calculate Levelized Cost of Energy")
    def calculate_lcoe(self, capital_cost: float, annual_generation: float,
                      lifetime_years: int = 25, discount_rate: float = 0.08) -> dict:
        """Calculate LCOE for renewable energy projects."""
        # Simple LCOE calculation
        annual_payment = capital_cost * (discount_rate * (1 + discount_rate)**lifetime_years) / ((1 + discount_rate)**lifetime_years - 1)
        lcoe = annual_payment / annual_generation

        return {
            "lcoe_per_mwh": round(lcoe, 2),
            "capital_cost": capital_cost,
            "annual_generation": annual_generation,
            "lifetime_years": lifetime_years,
            "discount_rate": discount_rate
        }

    @tool(name="capacity_factor", description="Calculate capacity factor")
    def capacity_factor(self, actual_generation: float,
                       rated_capacity: float) -> dict:
        """Calculate capacity factor for renewable energy systems."""
        max_generation = rated_capacity * 8760  # 24/7 for a year
        cf = actual_generation / max_generation

        return {
            "capacity_factor": round(cf, 3),
            "capacity_factor_percent": round(cf * 100, 1),
            "actual_generation_mwh": actual_generation,
            "rated_capacity_mw": rated_capacity,
            "theoretical_max_mwh": max_generation
        }

# Create an agent that uses the tools from the skill
@agent(
    name="EnergyAnalyst",
    description="Expert in renewable energy financial analysis",
    system_prompt="You are an expert energy analyst. Use your tools to help with renewable energy calculations and provide insights.",
    tools=["calculate_lcoe", "capacity_factor"]  # Use tools from the EnergyCalculations skill
)
class EnergyAnalyst:
    temperature = 0.3

# Test the skill directly
print("🧪 Testing EnergyCalculations skill:")
skill_instance = EnergyCalculations()

# Test LCOE calculation
lcoe_result = skill_instance.calculate_lcoe(
    capital_cost=100_000_000,  # $100M solar project
    annual_generation=250_000,  # 250,000 MWh/year
    lifetime_years=25,
    discount_rate=0.08
)
print(f"LCOE Result: {lcoe_result}")

# Test capacity factor
cf_result = skill_instance.capacity_factor(
    actual_generation=250_000,  # 250,000 MWh/year actual
    rated_capacity=100  # 100 MW rated capacity
)
print(f"Capacity Factor: {cf_result}")
```

## 👑 Master Agents (Coordination)

Master agents coordinate multiple specialized agents:

### Create `master_agent_example.py`:
```python
from energyai_sdk import agent, master_agent, tool, bootstrap_agents, initialize_sdk

initialize_sdk(log_level="INFO")

# Create specialized tools
@tool(name="technical_analysis")
def technical_analysis(technology: str, capacity: float, location: str) -> dict:
    """Analyze technical aspects of renewable energy project."""
    return {
        "technology": technology,
        "capacity_mw": capacity,
        "location": location,
        "analysis": f"Technical analysis for {capacity}MW {technology} project in {location}",
        "feasibility": "High" if capacity < 500 else "Medium"
    }

@tool(name="financial_analysis")
def financial_analysis(capex: float, opex: float, revenue: float) -> dict:
    """Analyze financial aspects of energy project."""
    annual_profit = revenue - opex
    payback_period = capex / annual_profit if annual_profit > 0 else None

    return {
        "capex": capex,
        "annual_opex": opex,
        "annual_revenue": revenue,
        "annual_profit": annual_profit,
        "payback_years": round(payback_period, 1) if payback_period else "N/A",
        "recommendation": "Proceed" if payback_period and payback_period < 10 else "Reconsider"
    }

# Create specialized agents
@agent(
    name="TechnicalExpert",
    description="Technical analysis specialist for renewable energy",
    system_prompt="You are a technical expert in renewable energy systems. Focus on engineering and performance aspects.",
    tools=["technical_analysis"]
)
class TechnicalExpert:
    temperature = 0.2

@agent(
    name="FinancialExpert",
    description="Financial analysis specialist for energy projects",
    system_prompt="You are a financial analyst specializing in energy investments. Focus on costs, returns, and financial viability.",
    tools=["financial_analysis"]
)
class FinancialExpert:
    temperature = 0.2

# Create master agent that coordinates the specialists
@master_agent(
    name="ProjectManager",
    description="Master coordinator for energy project analysis",
    system_prompt="You coordinate technical and financial analysis for energy projects. Gather input from specialists and provide comprehensive recommendations.",
    subordinates=["TechnicalExpert", "FinancialExpert"]
)
class ProjectManager:
    temperature = 0.4
    selection_strategy = "prompt"  # Let AI choose which subordinate to use
    max_iterations = 3

# Test the tools directly
print("🧪 Testing coordination tools:")
print("Technical:", technical_analysis("Solar", 100, "California"))
print("Financial:", financial_analysis(150_000_000, 2_000_000, 12_000_000))

print("\\n✅ Master agent system created!")
print("💡 The ProjectManager can now coordinate TechnicalExpert and FinancialExpert")
```

## 🌐 Creating a Web Interface (Optional)

Turn your agents into a web service:

### Create `web_service.py`:
```python
from energyai_sdk import agent, tool, initialize_sdk
from energyai_sdk.agents import bootstrap_agents

# Only try to import if FastAPI is available
try:
    from energyai_sdk.application import create_application, run_development_server
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("Web features not available. Install with: pip install fastapi uvicorn")

initialize_sdk(log_level="INFO")

# Create your agents (same as before)
@tool(name="greet_user")
def greet_user(name: str, language: str = "english") -> dict:
    greetings = {
        "english": f"Hello {name}! How can I help you today?",
        "spanish": f"¡Hola {name}! ¿Cómo puedo ayudarte hoy?",
        "french": f"Bonjour {name}! Comment puis-je vous aider aujourd'hui?"
    }
    return {
        "greeting": greetings.get(language.lower(), greetings["english"]),
        "name": name,
        "language": language
    }

@agent(
    name="GreeterBot",
    description="Friendly greeting bot",
    system_prompt="You are a friendly assistant that greets users in different languages.",
    tools=["greet_user"]
)
class GreeterBot:
    temperature = 0.7

if WEB_AVAILABLE:
    def start_web_server():
        """Start the web server with your agents."""

        # Configure AI (replace with your keys)
        config = {
            "deployment_name": "gpt-4o",
            "endpoint": "https://your-resource.openai.azure.com/",
            "api_key": "your-api-key"
        }

        try:
            # Create agents
            agents = bootstrap_agents(azure_openai_config=config)

            # Start development server
            run_development_server(
                agents=list(agents.values()),
                host="0.0.0.0",
                port=8000,
                reload=True
            )
        except Exception as e:
            print(f"❌ Could not start web server: {e}")
            print("💡 Make sure your API keys are configured")

    if __name__ == "__main__":
        print("🌐 Starting web server...")
        print("📍 Open http://localhost:8000 in your browser")
        print("📚 API docs: http://localhost:8000/docs")
        start_web_server()
else:
    print("⚠️  Web server not available - install FastAPI and uvicorn")
```

## 🎯 Next Steps

### 🔧 Development Setup
```bash
# Set up development environment
python scripts/setup_dev.py

# Run all tests
python scripts/run_tests.py
```

### 📚 Learn More
- Explore [examples/](examples/) for complete working examples
- Check out `examples/decorator_concept_demo.py` for advanced features
- Look at `examples/energy_skills.py` for domain-specific tools

### 🚀 Production Deployment
```bash
# Build for production
python scripts/deploy.py build

# Deploy to PyPI (when ready)
python scripts/deploy.py upload
```

## 🔭 Unified Observability System

EnergyAI SDK provides a comprehensive unified observability system that integrates:

1. **Langfuse** for LLM-specific observability (traces, generations, spans)
2. **OpenTelemetry** for general application monitoring (metrics, traces, logs)
3. **Azure Monitor** integration for cloud deployments

### 1. Enable Observability

```python
from energyai_sdk import initialize_sdk

# Enable observability during SDK initialization
initialize_sdk(
    langfuse_public_key="your_langfuse_public_key",
    langfuse_secret_key="your_langfuse_secret_key",
    azure_monitor_connection_string="your_connection_string",
    environment="production"  # or "development", "staging"
)
```

### 2. Monitor Functions and Methods

```python
from energyai_sdk import monitor

@monitor("energy_calculation")
def calculate_energy_metrics(data):
    # Your calculation logic
    return result
```

### 3. Advanced Observability

```python
from energyai_sdk import get_observability_manager

# Get the global observability manager
observability = get_observability_manager()

# Create a trace for a user session
trace = observability.create_trace(
    name="user-session",
    user_id="user123",
    session_id="session456",
    metadata={"source": "web_app"}
)

# Create a generation for an LLM call
generation = observability.create_generation(
    trace,
    name="energy-advice-generation",
    input_data={"query": "How can I reduce my energy bill?"},
    model="gpt-4",
    model_parameters={"temperature": 0.7}
)

# End the generation with the result
observability.end_generation(
    generation,
    output="Here are 5 ways to reduce your energy bill...",
    usage={"prompt_tokens": 150, "completion_tokens": 200, "total_tokens": 350}
)

# Use spans for general operations
with observability.start_span("data_processing", source="sensor_data"):
    # Process data
    processed_data = process_data(raw_data)

# Update the trace with final results
observability.update_trace(
    trace,
    metadata={"processing_complete": True}
)

# Flush telemetry data
observability.flush()
```

### 4. Application Integration

```python
from energyai_sdk.application import create_application

# Create application with unified observability
app = create_application(
    enable_observability=True,
    enable_langfuse_monitoring=True,
    langfuse_public_key="your_langfuse_public_key",
    langfuse_secret_key="your_langfuse_secret_key",
    azure_monitor_connection_string="your_connection_string",
    langfuse_environment="production"
)
```

### 💡 Pro Tips

1. **Start Simple**: Begin with basic tools and agents, then add complexity
2. **Test Tools First**: Always test your `@tool` functions independently before adding AI
3. **Use Skills**: Group related tools into skills for better organization
4. **Monitor Performance**: Use `@monitor` decorator for important functions
5. **Handle Errors**: Wrap AI calls in try/catch blocks during development
6. **Enable Observability**: Use the unified observability system for production applications

### 🆘 Troubleshooting

**"Import Error"**: Make sure you installed with `pip install -e .`
**"No AI Response"**: Check your API keys and network connection
**"Tool Not Found"**: Ensure tool names match exactly in agent definitions
**"Async Issues"**: Remember to use `await` when calling agent methods

## 🎉 Congratulations!

You've just learned how to:
- ✅ Create tools with `@tool`
- ✅ Build agents with `@agent`
- ✅ Organize tools into skills with `@skill`
- ✅ Coordinate agents with `@master_agent`
- ✅ Add web interfaces (optional)

**You're ready to build powerful AI agent systems!** 🚀

---

**Need help?** Check the [examples](examples/) or open an issue on GitHub!
