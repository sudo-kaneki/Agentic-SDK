"""
Production Azure Platform Example

This example demonstrates how to set up a complete production-ready
EnergyAI platform with full Azure integration including:

- Externalized Agentic Registry (Cosmos DB)
- Externalized Context Store (Cosmos DB)
- Integrated Observability (OpenTelemetry + Azure Monitor)
- Session persistence and management
- Dynamic tool loading from registry
- Full monitoring and telemetry

Usage:
    # Development mode (with mock clients)
    python examples/production_azure_platform.py --mode dev

    # Production mode (requires Azure credentials)
    python examples/production_azure_platform.py --mode prod
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk import (
    agent,
    bootstrap_agents,
    create_application,
    create_production_application,
    run_development_server,
    tool,
)
from energyai_sdk.clients import (
    ContextStoreClient,
    MockContextStoreClient,
    MockMonitoringClient,
    MockRegistryClient,
    MonitoringClient,
    MonitoringConfig,
    RegistryClient,
)
from energyai_sdk.core import KernelFactory


# Define energy-specific tools
@tool(
    name="calculate_lcoe",
    description="Calculate Levelized Cost of Energy for renewable energy projects",
)
def calculate_lcoe(
    capex: float,
    opex_annual: float,
    generation_annual_mwh: float,
    project_life_years: int = 25,
    discount_rate: float = 0.07,
) -> dict:
    """
    Calculate LCOE using standard financial formula.

    Args:
        capex: Capital expenditure (initial investment)
        opex_annual: Annual operating expenditure
        generation_annual_mwh: Annual energy generation in MWh
        project_life_years: Project lifetime in years
        discount_rate: Discount rate for NPV calculation

    Returns:
        Dictionary with LCOE calculation results
    """
    # Calculate present value of OPEX over project life
    opex_pv = sum(
        opex_annual / (1 + discount_rate) ** year for year in range(1, project_life_years + 1)
    )

    # Calculate present value of generation
    generation_pv = sum(
        generation_annual_mwh / (1 + discount_rate) ** year
        for year in range(1, project_life_years + 1)
    )

    # LCOE = (CAPEX + PV of OPEX) / PV of Generation
    lcoe = (capex + opex_pv) / generation_pv if generation_pv > 0 else float("inf")

    # Determine competitiveness
    competitive_threshold = 50  # $/MWh
    competitiveness = (
        "Highly Competitive"
        if lcoe < 30
        else "Competitive" if lcoe < competitive_threshold else "Less Competitive"
    )

    return {
        "lcoe_per_mwh": round(lcoe, 2),
        "capex": capex,
        "opex_pv": round(opex_pv, 2),
        "generation_pv": round(generation_pv, 2),
        "project_life_years": project_life_years,
        "discount_rate": discount_rate,
        "competitiveness": competitiveness,
        "analysis": f"LCOE of ${lcoe:.2f}/MWh is {competitiveness.lower()}",
    }


@tool(
    name="capacity_factor_analysis",
    description="Analyze capacity factor and performance for renewable energy systems",
)
def capacity_factor_analysis(
    actual_generation_mwh: float, nameplate_capacity_mw: float, period_hours: int = 8760
) -> dict:
    """
    Calculate and analyze capacity factor for renewable energy systems.

    Args:
        actual_generation_mwh: Actual energy generation in MWh
        nameplate_capacity_mw: Nameplate capacity in MW
        period_hours: Period in hours (default: 8760 for full year)

    Returns:
        Dictionary with capacity factor analysis
    """
    theoretical_max = nameplate_capacity_mw * period_hours
    capacity_factor = (actual_generation_mwh / theoretical_max) * 100 if theoretical_max > 0 else 0

    # Performance rating based on technology benchmarks
    if capacity_factor > 50:
        rating = "Excellent"
        comment = "Outstanding performance for renewable energy"
    elif capacity_factor > 35:
        rating = "Very Good"
        comment = "Above average performance"
    elif capacity_factor > 25:
        rating = "Good"
        comment = "Typical performance range"
    elif capacity_factor > 15:
        rating = "Fair"
        comment = "Below average, investigate issues"
    else:
        rating = "Poor"
        comment = "Significant performance issues"

    return {
        "capacity_factor_percent": round(capacity_factor, 2),
        "actual_generation_mwh": actual_generation_mwh,
        "theoretical_max_mwh": round(theoretical_max, 2),
        "nameplate_capacity_mw": nameplate_capacity_mw,
        "period_hours": period_hours,
        "performance_rating": rating,
        "analysis": comment,
        "efficiency": f"{capacity_factor:.1f}% of theoretical maximum",
    }


@tool(
    name="carbon_impact_calculator",
    description="Calculate carbon emission reductions and environmental impact",
)
def carbon_impact_calculator(
    generation_mwh: float, grid_emission_factor: float = 0.4, project_life_years: int = 25
) -> dict:
    """
    Calculate carbon emission reductions from renewable energy generation.

    Args:
        generation_mwh: Annual energy generation in MWh
        grid_emission_factor: Grid emission factor in tonnes CO2/MWh (default: 0.4)
        project_life_years: Project lifetime in years

    Returns:
        Dictionary with carbon impact analysis
    """
    annual_co2_reduction = generation_mwh * grid_emission_factor
    lifetime_co2_reduction = annual_co2_reduction * project_life_years

    # Environmental equivalents for context
    trees_equivalent = lifetime_co2_reduction * 16  # ~1 tree absorbs ~62 lbs CO2/year
    cars_equivalent = lifetime_co2_reduction / 4.6  # Average car emits ~4.6 tonnes CO2/year

    return {
        "annual_co2_reduction_tonnes": round(annual_co2_reduction, 2),
        "lifetime_co2_reduction_tonnes": round(lifetime_co2_reduction, 2),
        "grid_emission_factor": grid_emission_factor,
        "project_life_years": project_life_years,
        "environmental_equivalents": {
            "trees_planted_equivalent": round(trees_equivalent, 0),
            "cars_removed_equivalent": round(cars_equivalent, 0),
        },
        "impact_summary": f"Prevents {lifetime_co2_reduction:,.0f} tonnes CO2 over {project_life_years} years",
        "annual_impact": f"Reduces {annual_co2_reduction:,.0f} tonnes CO2 annually",
    }


# Define specialized energy agents
@agent(
    name="EnergyFinancialAnalyst",
    description="Expert in energy project financial analysis and LCOE calculations",
    system_prompt="""You are an expert energy financial analyst specializing in renewable energy project economics.

    You help users understand:
    - Levelized Cost of Energy (LCOE) calculations
    - Project financial viability
    - Investment analysis and returns
    - Cost comparisons with conventional energy

    Always provide clear explanations of financial metrics and their implications for energy projects.""",
    tools=["calculate_lcoe"],
)
class EnergyFinancialAnalyst:
    temperature = 0.3  # More deterministic for financial analysis
    max_tokens = 2000


@agent(
    name="EnergyPerformanceAnalyst",
    description="Expert in renewable energy system performance and capacity analysis",
    system_prompt="""You are an expert energy performance analyst specializing in renewable energy system optimization.

    You help users understand:
    - Capacity factor analysis and benchmarks
    - System performance optimization
    - Generation forecasting and analysis
    - Technology-specific performance insights

    Provide actionable insights on system performance and improvement opportunities.""",
    tools=["capacity_factor_analysis"],
)
class EnergyPerformanceAnalyst:
    temperature = 0.2
    max_tokens = 2000


@agent(
    name="EnvironmentalImpactAnalyst",
    description="Expert in environmental impact and carbon footprint analysis for energy projects",
    system_prompt="""You are an environmental impact analyst specializing in renewable energy sustainability analysis.

    You help users understand:
    - Carbon emission reduction calculations
    - Environmental impact assessment
    - Sustainability metrics and reporting
    - Climate impact of energy projects

    Provide comprehensive environmental impact analysis with clear, actionable insights.""",
    tools=["carbon_impact_calculator"],
)
class EnvironmentalImpactAnalyst:
    temperature = 0.2
    max_tokens = 2000


@agent(
    name="EnergyProjectConsultant",
    description="Comprehensive energy project consultant with access to all analysis tools",
    system_prompt="""You are a comprehensive energy project consultant with expertise across financial, technical, and environmental aspects of renewable energy projects.

    You can help with:
    - Complete project feasibility analysis
    - Financial and economic evaluation
    - Technical performance assessment
    - Environmental impact analysis
    - Strategic recommendations

    Use all available tools to provide thorough, multi-dimensional analysis of energy projects. Always consider financial viability, technical performance, and environmental impact in your recommendations.""",
    tools=["calculate_lcoe", "capacity_factor_analysis", "carbon_impact_calculator"],
)
class EnergyProjectConsultant:
    temperature = 0.4  # Balanced for comprehensive analysis
    max_tokens = 3000


async def setup_azure_clients(mode: str):
    """Setup Azure service clients based on mode."""
    if mode == "prod":
        # Production mode - use real Azure clients
        cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        cosmos_key = os.getenv("COSMOS_KEY")
        azure_monitor_conn = os.getenv("AZURE_MONITOR_CONNECTION_STRING")

        if not cosmos_endpoint or not cosmos_key:
            print(
                "❌ Production mode requires COSMOS_ENDPOINT and COSMOS_KEY environment variables"
            )
            print("Falling back to mock clients for demo...")
            return await setup_mock_clients()

        print("🔧 Setting up production Azure clients...")
        print(f"   Cosmos endpoint: {cosmos_endpoint}")

        # Real Azure clients
        registry_client = RegistryClient(cosmos_endpoint, cosmos_key)
        context_store_client = ContextStoreClient(cosmos_endpoint, cosmos_key)

        # Monitoring client
        monitoring_config = MonitoringConfig(
            service_name="energyai-production",
            environment="production",
            azure_monitor_connection_string=azure_monitor_conn,
        )
        monitoring_client = (
            MonitoringClient(monitoring_config) if azure_monitor_conn else MockMonitoringClient()
        )

        return registry_client, context_store_client, monitoring_client

    else:
        # Development mode - use mock clients
        return await setup_mock_clients()


async def setup_mock_clients():
    """Setup mock clients for development."""
    print("🔧 Setting up mock Azure clients for development...")

    registry_client = MockRegistryClient()
    context_store_client = MockContextStoreClient()
    monitoring_client = MockMonitoringClient()

    return registry_client, context_store_client, monitoring_client


async def demonstrate_azure_features(registry_client, context_store_client, monitoring_client):
    """Demonstrate Azure integration features."""
    print("\n🚀 Demonstrating Azure Integration Features")
    print("=" * 60)

    # 1. Registry Client Demo
    print("\n📋 Registry Client - Available Tools and Agents:")
    tools = await registry_client.list_tools()
    agents = await registry_client.list_agents()

    print(f"   Available tools: {len(tools)}")
    for tool in tools[:3]:  # Show first 3
        print(f"     - {tool.name}: {tool.description}")

    print(f"   Available agents: {len(agents)}")
    for agent in agents[:3]:  # Show first 3
        print(f"     - {agent.name}: {agent.description}")

    # 2. Context Store Demo
    print("\n💾 Context Store - Session Management:")
    session_id = "demo_energy_consultation"
    subject_id = "energy_developer_123"

    # Create session
    session = await context_store_client.create_session(
        session_id=session_id,
        subject_id=subject_id,
        initial_context={
            "project_type": "solar_farm",
            "capacity_mw": 100,
            "location": "Arizona, USA",
        },
    )
    print(f"   Created session: {session_id}")
    print(f"   Subject: {subject_id}")

    # Add conversation context
    await context_store_client.append_to_context(
        session_id=session_id,
        key="consultation_history",
        value={
            "timestamp": session.created_at.isoformat(),
            "query": "Need LCOE analysis for 100MW solar project",
            "response": "I'll help you analyze the financial viability...",
        },
    )
    print("   Added consultation context")

    # 3. Monitoring Demo
    print("\n📊 Monitoring - Recording Metrics and Traces:")

    # Record some demo metrics
    monitoring_client.record_metric("energy_consultations_total", 1.0, {"project_type": "solar"})
    monitoring_client.record_metric("project_capacity_mw", 100.0, {"technology": "solar"})

    # Create a traced operation
    with monitoring_client.start_span("energy_analysis_workflow") as span:
        with monitoring_client.start_span("lcoe_calculation"):
            # Simulate LCOE calculation
            import time

            time.sleep(0.1)

        with monitoring_client.start_span("performance_analysis"):
            # Simulate performance analysis
            time.sleep(0.05)

    if hasattr(monitoring_client, "get_recorded_metrics"):
        metrics = monitoring_client.get_recorded_metrics()
        traces = monitoring_client.get_recorded_traces()
        print(f"   Recorded {len(metrics)} metrics and {len(traces)} trace spans")


async def demonstrate_dynamic_tool_loading():
    """Demonstrate dynamic tool loading from registry."""
    print("\n⚙️ Dynamic Tool Loading from Registry")
    print("=" * 50)

    # Create kernel (requires Semantic Kernel)
    kernel = KernelFactory.create_kernel()
    if not kernel:
        print("   ⚠️  Semantic Kernel not available - skipping dynamic tool loading demo")
        return

    # Load tools from registry
    registry_client = MockRegistryClient()
    loaded_count = await KernelFactory.load_tools_from_registry(kernel, registry_client)

    print(f"   ✅ Loaded {loaded_count} tools from registry into kernel")
    print("   Tools are now available for agent execution")


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("energyai_platform.log")],
    )


async def run_production_platform(mode: str, port: int = 8000):
    """Run the production platform."""
    print(f"\n🚀 Starting EnergyAI Production Platform ({mode} mode)")
    print("=" * 60)

    # Setup Azure clients
    registry_client, context_store_client, monitoring_client = await setup_azure_clients(mode)

    # Demonstrate Azure features
    await demonstrate_azure_features(registry_client, context_store_client, monitoring_client)

    # Demonstrate dynamic tool loading
    await demonstrate_dynamic_tool_loading()

    # Configure Azure OpenAI (if available)
    azure_config = {
        "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
        "api_key": os.getenv("AZURE_OPENAI_KEY", "your-api-key"),
    }

    print("\n🤖 Bootstrapping Energy Agents...")
    try:
        # Bootstrap agents
        agents = bootstrap_agents(azure_openai_config=azure_config)
        print(f"   Created {len(agents)} specialized energy agents:")
        for agent_name in agents.keys():
            print(f"     - {agent_name}")
    except Exception as e:
        print(f"   ⚠️  Agent bootstrap error (likely missing Azure OpenAI config): {e}")
        print("   Continuing with application setup...")
        agents = {}

    # Create application with Azure integration
    if mode == "prod":
        print("\n🏭 Creating production application...")
        app = create_production_application(
            api_keys=[os.getenv("API_KEY", "demo_api_key")],
            cosmos_endpoint=os.getenv("COSMOS_ENDPOINT"),
            cosmos_key=os.getenv("COSMOS_KEY"),
            azure_monitor_connection_string=os.getenv("AZURE_MONITOR_CONNECTION_STRING"),
            max_requests_per_minute=100,
        )
    else:
        print("\n🔧 Creating development application...")
        app = create_application(
            title="EnergyAI Development Platform",
            registry_client=registry_client,
            context_store_client=context_store_client,
            monitoring_client=monitoring_client,
            debug=True,
        )

    # Add agents to application
    for agent in agents.values():
        app.add_agent(agent)

    print(f"\n🌐 Platform ready with {len(agents)} agents and Azure integration")
    print("\n📚 Available API endpoints:")
    print("   GET  /health              - Health check")
    print("   GET  /agents              - List all agents")
    print("   POST /chat                - Chat with any agent")
    print("   POST /agents/{id}/chat    - Chat with specific agent")
    print("   POST /sessions/{id}       - Create session")
    print("   GET  /sessions/{id}       - Get session")
    print("   POST /registry/reload     - Reload from registry")

    # Start development server
    if agents:
        print(f"\n🚀 Starting development server on port {port}...")
        print("   Access the platform at: http://localhost:{port}")
        print("   API documentation: http://localhost:{port}/docs")
        print("\n   Press Ctrl+C to stop the server")

        try:
            run_development_server(
                agents=list(agents.values()), host="0.0.0.0", port=port, reload=(mode == "dev")
            )
        except KeyboardInterrupt:
            print("\n✅ Server stopped gracefully")
        except Exception as e:
            print(f"\n❌ Server error: {e}")
    else:
        print("\n⚠️  No agents available - server not started")
        print("   Configure Azure OpenAI credentials to enable agents")


async def run_consultation_demo():
    """Run an interactive consultation demo."""
    print("\n💬 Energy Project Consultation Demo")
    print("=" * 50)
    print("This demo simulates a consultation session with persistent context")

    # Setup mock clients
    registry_client, context_store_client, monitoring_client = await setup_mock_clients()

    # Create session
    session_id = "energy_consultation_demo"
    subject_id = "demo_developer"

    session = await context_store_client.create_session(
        session_id=session_id,
        subject_id=subject_id,
        initial_context={
            "consultation_type": "solar_project_analysis",
            "project_details": {
                "technology": "solar_pv",
                "capacity_mw": 50,
                "location": "California, USA",
                "capex_per_mw": 1200000,  # $1.2M per MW
                "opex_annual": 50000,  # $50k annual O&M
            },
        },
    )

    print(f"📋 Created consultation session: {session_id}")

    # Simulate consultation queries
    queries = [
        {
            "question": "What would be the LCOE for this 50MW solar project?",
            "tool_used": "calculate_lcoe",
            "parameters": {
                "capex": 60000000,  # 50MW * $1.2M/MW
                "opex_annual": 50000,
                "generation_annual_mwh": 125000,  # ~25% capacity factor
                "project_life_years": 25,
            },
        },
        {
            "question": "How does the capacity factor look for this system?",
            "tool_used": "capacity_factor_analysis",
            "parameters": {"actual_generation_mwh": 125000, "nameplate_capacity_mw": 50},
        },
        {
            "question": "What's the environmental impact of this project?",
            "tool_used": "carbon_impact_calculator",
            "parameters": {
                "generation_mwh": 125000,
                "grid_emission_factor": 0.35,  # California grid factor
                "project_life_years": 25,
            },
        },
    ]

    consultation_results = []

    for i, query in enumerate(queries, 1):
        print(f"\n❓ Query {i}: {query['question']}")

        # Record in monitoring
        with monitoring_client.start_span(f"consultation_query_{i}") as span:
            # Execute the relevant tool
            if query["tool_used"] == "calculate_lcoe":
                result = calculate_lcoe(**query["parameters"])
            elif query["tool_used"] == "capacity_factor_analysis":
                result = capacity_factor_analysis(**query["parameters"])
            elif query["tool_used"] == "carbon_impact_calculator":
                result = carbon_impact_calculator(**query["parameters"])

            consultation_results.append(
                {"query": query["question"], "tool": query["tool_used"], "result": result}
            )

        # Add to session context
        await context_store_client.append_to_context(
            session_id=session_id,
            key="consultation_results",
            value={
                "query_id": i,
                "question": query["question"],
                "tool_used": query["tool_used"],
                "result": result,
                "timestamp": session.created_at.isoformat(),
            },
        )

        # Display key results
        if query["tool_used"] == "calculate_lcoe":
            print(f"   💰 LCOE: ${result['lcoe_per_mwh']}/MWh ({result['competitiveness']})")
        elif query["tool_used"] == "capacity_factor_analysis":
            print(
                f"   ⚡ Capacity Factor: {result['capacity_factor_percent']}% ({result['performance_rating']})"
            )
        elif query["tool_used"] == "carbon_impact_calculator":
            print(
                f"   🌱 Carbon Impact: {result['lifetime_co2_reduction_tonnes']:,.0f} tonnes CO2 saved"
            )

    # Final consultation summary
    print("\n📊 Consultation Summary for 50MW Solar Project:")
    print(f"   💰 Economics: {consultation_results[0]['result']['competitiveness']}")
    print(f"   ⚡ Performance: {consultation_results[1]['result']['performance_rating']}")
    print(
        f"   🌱 Environmental: {consultation_results[2]['result']['lifetime_co2_reduction_tonnes']:,.0f} tonnes CO2 prevented"
    )

    # Show session persistence
    retrieved_session = await context_store_client.get_session(session_id)
    consultation_count = len(retrieved_session.context.get("consultation_results", []))
    print(f"\n💾 Session Context: {consultation_count} consultation results stored")

    # Show monitoring data
    if hasattr(monitoring_client, "get_recorded_metrics"):
        metrics = monitoring_client.get_recorded_metrics()
        traces = monitoring_client.get_recorded_traces()
        print(f"📊 Monitoring: {len(traces)} operations traced, {len(metrics)} metrics recorded")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EnergyAI Production Platform with Azure Integration"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Run in development or production mode",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the development server on"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run consultation demo instead of server"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    print("🌟 EnergyAI SDK - Production Azure Platform")
    print("=" * 60)
    print("This example demonstrates the complete Azure integration including:")
    print("✅ Externalized Agentic Registry (Cosmos DB)")
    print("✅ Externalized Context Store (Cosmos DB)")
    print("✅ Integrated Observability (OpenTelemetry + Azure Monitor)")
    print("✅ Session Management and Persistence")
    print("✅ Dynamic Tool Loading from Registry")
    print("✅ Specialized Energy Analysis Agents")

    if args.mode == "prod":
        print("\n🏭 Production Mode Requirements:")
        print("   Set these environment variables:")
        print("   - COSMOS_ENDPOINT=https://your-account.documents.azure.com:443/")
        print("   - COSMOS_KEY=your_cosmos_key")
        print("   - AZURE_MONITOR_CONNECTION_STRING=InstrumentationKey=your_key")
        print("   - AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/")
        print("   - AZURE_OPENAI_KEY=your_api_key")
        print("   - AZURE_OPENAI_DEPLOYMENT=gpt-4")
        print("   - API_KEY=your_secure_api_key")

    try:
        if args.demo:
            asyncio.run(run_consultation_demo())
        else:
            asyncio.run(run_production_platform(args.mode, args.port))
    except KeyboardInterrupt:
        print("\n✅ Platform stopped gracefully")
    except Exception as e:
        print(f"\n❌ Platform error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
