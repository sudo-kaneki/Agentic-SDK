# EnergyAI SDK 🚀

**A Powerful SDK Wrapper for Semantic Kernel with Azure Cloud Integration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Semantic Kernel](https://img.shields.io/badge/semantic--kernel-1.0+-green.svg)](https://github.com/microsoft/semantic-kernel)
[![Azure Integration](https://img.shields.io/badge/azure-cosmos%20%7C%20monitor-blue.svg)](https://azure.microsoft.com/)

## 🎯 Mission

Transform complex Semantic Kernel agent creation into simple, declarative Python decorators with enterprise-grade Azure integration. Create powerful AI agents with just decorators and scale them with cloud-native architecture!

**🌟 Perfect for:** Energy analytics, multi-agent orchestration, Azure cloud deployment, enterprise AI applications, and production-ready AI platforms.

## ⚡ Quick Start

### Before (Raw Semantic Kernel)
```python
# 50+ lines of complex setup code...
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
# ... lots more imports and configuration
```

### After (EnergyAI SDK)
```python
from energyai_sdk import agent, tool, bootstrap_agents

@tool(name="calculate_energy")
def calculate_energy(power: float, time: float) -> dict:
    return {"energy_kwh": power * time}

@agent(name="EnergyExpert", tools=["calculate_energy"])
class EnergyExpert:
    system_prompt = "You are an expert energy analyst."

# Create all agents instantly!
agents = bootstrap_agents(azure_openai_config={
    "deployment_name": "gpt-4o",
    "endpoint": "your-endpoint",
    "api_key": "your-key"
})
```

## 🏗️ Cloud-Native Architecture

**Enterprise-Ready Design with Azure Integration:**
```
energyai_sdk/
├── core.py              # Registry, Base Classes, Kernel Factory
├── decorators.py        # @agent, @tool, @skill, @master_agent
├── agents.py            # Semantic Kernel Integration & bootstrap_agents()
├── application.py       # FastAPI Integration with Azure clients
├── middleware.py        # Request/Response Pipeline
├── config.py            # Configuration Management
├── exceptions.py        # Comprehensive Error Handling
└── clients/             # 🆕 Azure Integration
    ├── registry_client.py    # Cosmos DB agent/tool registry
    ├── context_store_client.py  # Session persistence
    └── monitoring.py         # OpenTelemetry + Azure Monitor
```

**✨ Key Benefits:**
- 🎯 **Decorator-Only**: No builder patterns, no complex APIs
- 🔧 **70% Code Reduction**: Clean, maintainable, and powerful
- ☁️ **Azure Native**: Full Cosmos DB + Azure Monitor integration
- 📊 **Enterprise Observability**: OpenTelemetry distributed tracing
- 🔄 **Dynamic Loading**: Runtime agent/tool loading from registry
- 💾 **Session Persistence**: Stateful conversations across requests
- 🚀 **Production Ready**: Comprehensive testing & error handling

## 📦 Installation

```bash
# Core dependency
pip install semantic-kernel

# Clone and install SDK
git clone <repository-url>
cd energyai_sdk_project
pip install -e .

# Azure integration dependencies
pip install azure-cosmos azure-monitor-opentelemetry-exporter

# Optional dependencies for web platform
pip install fastapi uvicorn  # For web API
pip install aiohttp          # For dynamic tool loading
pip install langfuse         # For monitoring
```

## 🎨 Features

### 🤖 **Agent Creation**
- **@agent**: Create AI agents with decorators
- **@master_agent**: Create coordinator agents for multi-agent workflows
- **Automatic Registration**: Agents auto-register in global registry

### 🛠️ **Tools & Skills**
- **@tool**: Register functions as agent tools
- **@skill**: Group related tools into reusable skill collections
- **Type Safety**: Full type hints and validation

### 📝 **Prompts & Planning**
- **@prompt**: Template-based prompt management
- **@planner**: Multi-step workflow coordination

### 📊 **Enterprise Observability** 🆕
- **@monitor**: Automatic performance tracking
- **Telemetry**: Azure Monitor & Langfuse integration
- **OpenTelemetry**: Full distributed tracing support
- **Session Persistence**: Stateful conversation management
- **Dynamic Registry**: Runtime agent/tool loading from Cosmos DB
- **Error Handling**: Comprehensive exception system

### 🌐 **Web Platform** (Optional)
- **FastAPI Integration**: REST API endpoints
- **Streaming Support**: Real-time responses
- **Middleware Pipeline**: Request/response processing

## 🌩️ Azure Integration Features

### 📋 **Externalized Agentic Registry (Cosmos DB)**
Dynamically load agents and tools from Azure Cosmos DB at runtime:

```python
from energyai_sdk.clients import RegistryClient

# Connect to Cosmos DB registry
registry_client = RegistryClient(
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_key="your_cosmos_key"
)

# List available tools and agents
tools = await registry_client.list_tools()
agents = await registry_client.list_agents()

# Get specific definitions
tool_def = await registry_client.get_tool_definition("energy_calculator")
agent_def = await registry_client.get_agent_definition("energy_analyst")
```

### 💾 **Externalized Context Store (Cosmos DB)**
Persistent conversation sessions across requests:

```python
from energyai_sdk.clients import ContextStoreClient

# Session persistence
context_store = ContextStoreClient(cosmos_endpoint, cosmos_key)

# Create session with initial context
session = await context_store.create_session(
    session_id="user_consultation",
    subject_id="energy_developer_123",
    initial_context={"project_type": "solar", "capacity_mw": 100}
)

# Chat messages automatically stored in session context
# Conversation context persists across server restarts
```

### 📊 **Integrated Observability (OpenTelemetry + Azure Monitor)**
Enterprise-grade monitoring and tracing:

```python
from energyai_sdk.clients import MonitoringClient, MonitoringConfig

# Configure monitoring
config = MonitoringConfig(
    service_name="energyai-production",
    environment="production",
    azure_monitor_connection_string="InstrumentationKey=your_key"
)

monitoring_client = MonitoringClient(config)

# Automatic tracing and metrics
with monitoring_client.start_span("energy_analysis") as span:
    # Agent operations are automatically traced
    # Metrics collected: execution time, success rate, error types
```

### 🔄 **Dynamic Tool Loading**
Load tools from registry into Semantic Kernel at runtime:

```python
from energyai_sdk.core import KernelFactory

# Create kernel and load tools from registry
kernel = KernelFactory.create_kernel()
loaded_count = await KernelFactory.load_tools_from_registry(kernel, registry_client)

print(f"Loaded {loaded_count} tools from registry")
# Tools with HTTP endpoints are automatically callable by agents
```

### 🏭 **Production-Ready Application**
Full Azure integration in one line:

```python
from energyai_sdk.application import create_production_application

# Complete production setup with Azure services
app = create_production_application(
    api_keys=["your_secure_api_key"],
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_key="your_cosmos_key",
    azure_monitor_connection_string="InstrumentationKey=your_key"
)

# Includes:
# - Session management endpoints (/sessions/{id})
# - Registry reload endpoint (/registry/reload)
# - Enhanced health checks with external service status
# - Automatic request tracing and metrics
```

## 🚀 Core Examples

### Basic Agent
```python
from energyai_sdk import agent, tool, bootstrap_agents

@tool(name="add_numbers")
def add_numbers(a: float, b: float) -> float:
    return a + b

@agent(name="MathBot", tools=["add_numbers"])
class MathBot:
    system_prompt = "You are a helpful math assistant."
    temperature = 0.1

agents = bootstrap_agents(azure_openai_config=config)
math_bot = agents["MathBot"]
```

### Master Agent (Multi-Agent Coordination)
```python
@agent(name="DataAnalyst")
class DataAnalyst:
    system_prompt = "You analyze data and create reports."

@agent(name="Visualizer")
class Visualizer:
    system_prompt = "You create charts and visualizations."

@master_agent(name="ReportMaster", subordinates=["DataAnalyst", "Visualizer"])
class ReportMaster:
    system_prompt = "You coordinate report generation."
    max_iterations = 3
```

### Skills (Tool Collections)
```python
@skill(name="EnergyCalculations")
class EnergyCalculations:
    @tool(name="lcoe")
    def calculate_lcoe(self, cost: float, generation: float) -> dict:
        return {"lcoe": cost / generation}

    @tool(name="capacity_factor")
    def capacity_factor(self, actual: float, theoretical: float) -> dict:
        return {"cf": actual / theoretical}

@agent(name="EnergyAnalyst")
class EnergyAnalyst:
    system_prompt = "You are an energy analysis expert with calculation skills."
    # Skills are automatically available to all agents
```

## ⚙️ Configuration

### Azure OpenAI
```python
azure_config = {
    "deployment_name": "gpt-4o",
    "endpoint": "https://your-endpoint.openai.azure.com/",
    "api_key": "your-api-key"
}

agents = bootstrap_agents(azure_openai_config=azure_config)
```

### OpenAI
```python
openai_config = {
    "api_key": "your-openai-key",
    "model": "gpt-4"
}

agents = bootstrap_agents(openai_config=openai_config)
```

## 🔬 Advanced Features

### Monitoring & Telemetry
```python
from energyai_sdk import monitor, initialize_sdk

@monitor("energy_calculation")
@tool(name="complex_calc")
def complex_calculation(data: dict) -> dict:
    # Automatically tracked performance
    return process_data(data)

# Initialize with telemetry
initialize_sdk(
    azure_monitor_connection_string="InstrumentationKey=...",
    langfuse_public_key="pk_...",
    langfuse_secret_key="sk_..."
)
```

### Web Platform
```python
from energyai_sdk.application import create_application, run_development_server

# Create web application
app = create_application(title="Energy AI Platform")

# Add agents to web API
for agent in agents.values():
    app.add_agent(agent)

# Run development server
run_development_server(
    agents=list(agents.values()),
    host="0.0.0.0",
    port=8000
)
```

### Energy-Specific Example
```python
from energyai_sdk import agent, tool, master_agent, bootstrap_agents

@tool(name="calculate_lcoe", description="Calculate Levelized Cost of Energy")
def calculate_lcoe(capex: float, opex: float, generation: float) -> dict:
    lcoe = (capex + opex) / generation if generation > 0 else 0
    return {"lcoe_per_mwh": lcoe, "analysis": "favorable" if lcoe < 50 else "expensive"}

@tool(name="capacity_analysis", description="Analyze capacity factor")
def capacity_analysis(actual_mwh: float, rated_capacity_mw: float) -> dict:
    theoretical_max = rated_capacity_mw * 8760  # 24/7 for year
    cf = actual_mwh / theoretical_max
    return {"capacity_factor": cf, "rating": "excellent" if cf > 0.4 else "good" if cf > 0.3 else "poor"}

@agent(name="FinancialAnalyst", tools=["calculate_lcoe"])
class FinancialAnalyst:
    system_prompt = "You are a financial analyst specializing in energy projects."
    temperature = 0.3

@agent(name="TechnicalAnalyst", tools=["capacity_analysis"])
class TechnicalAnalyst:
    system_prompt = "You are a technical expert in renewable energy systems."
    temperature = 0.3

@master_agent(name="EnergyProjectMaster", subordinates=["FinancialAnalyst", "TechnicalAnalyst"])
class EnergyProjectMaster:
    system_prompt = "You coordinate comprehensive energy project analysis."
    max_iterations = 3
```

## 📚 Documentation

- **[Getting Started](Getting_started.md)** - Step-by-step beginner tutorial
- **[Azure Integration Guide](docs/azure_integration.md)** - Complete Azure setup and deployment guide 🆕
- **[Examples](examples/)** - Complete working examples for various scenarios
  - `basic_agent.py` - Simple agent with Azure integration demo
  - `production_azure_platform.py` - Full production Azure platform 🆕
  - `azure_integration_example.py` - Comprehensive Azure features demo 🆕
- **[API Reference](docs/api_reference.md)** - Detailed API documentation

## 🛠️ Development

### Setup
```bash
# Install development dependencies
python scripts/setup_dev.py

# Run tests
python scripts/run_tests.py

# Build and deploy
python scripts/deploy.py
```

### Testing
```bash
# Run specific tests
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=energyai_sdk --cov-report=term-missing

# Test decorator functionality specifically
pytest tests/test_agents.py::TestDecoratorBasedAgents -v

# Test Azure integration features
pytest tests/test_azure_integration.py -v
pytest tests/test_application_azure.py -v
```

## 🧪 Example Usage Patterns

### 1. Simple Calculator Agent
```python
from energyai_sdk import agent, tool, bootstrap_agents

@tool(name="calculate")
def calculate(expression: str) -> dict:
    try:
        result = eval(expression)  # Note: Use safe_eval in production
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}

@agent(name="Calculator", tools=["calculate"])
class Calculator:
    system_prompt = "You are a helpful calculator. Use the calculate tool for math operations."
    temperature = 0.1
```

### 2. Multi-Agent Research Team
```python
@agent(name="Researcher")
class Researcher:
    system_prompt = "You research topics and gather information."

@agent(name="Writer")
class Writer:
    system_prompt = "You write well-structured reports and articles."

@agent(name="Reviewer")
class Reviewer:
    system_prompt = "You review content for quality and accuracy."

@master_agent(name="ResearchTeam", subordinates=["Researcher", "Writer", "Reviewer"])
class ResearchTeam:
    system_prompt = "You coordinate a research team to produce high-quality reports."
    max_iterations = 5
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

**Development Guidelines:**
- Use decorator-only patterns (no builder patterns)
- Add comprehensive tests for new features
- Update documentation for any API changes
- Follow type hints throughout

## 📄 License


## 🙏 Acknowledgments

- **Microsoft Semantic Kernel** - The powerful AI orchestration platform
- **FastAPI** - Modern web framework for APIs
- **Langfuse** - LLM engineering platform for observability
- **Pydantic** - Data validation and settings management

## 💬 Support

- **Issues**: Report bugs and request features in [GitHub Issues]
- **Discussions**: Ask questions in [GitHub Discussions]
- **Documentation**: Check our [comprehensive docs](docs/)

## 🎬 Quick Demo

Want to see it in action? Check out these examples:

```bash
# Try the basic examples
python examples/basic_agent.py --mode server
python examples/energy_tools.py
python examples/master_agent.py

# Try the new Azure integration examples 🆕
python examples/azure_integration_example.py
python examples/production_azure_platform.py --demo
python examples/production_azure_platform.py --mode dev --port 8000

# Run the complete production platform (requires Azure setup)
python examples/production_azure_platform.py --mode prod
```

---

**🚀 Ready to build powerful AI agents with simple decorators and enterprise Azure integration?**

🌟 **Get started with the [Getting Started Guide](Getting_started.md)!**

☁️ **For production Azure deployment, see the [Azure Integration Guide](docs/azure_integration.md)!**
