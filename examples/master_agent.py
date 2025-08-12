# examples/master_agent.py
"""
Master Agent Example

This example demonstrates how to create a master agent that orchestrates
multiple specialized agents using the decorator approach.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk import AgentRequest, agent, initialize_sdk, master_agent, tool
from energyai_sdk.agents import bootstrap_agents

# Try to import optional components
try:
    from energyai_sdk.application import DevelopmentServer, create_application

    APPLICATION_AVAILABLE = True
except ImportError:
    APPLICATION_AVAILABLE = False
    print("Note: Application module not available.")


# ==============================================================================
# SPECIALIZED TOOLS FOR ENERGY ANALYSIS
# ==============================================================================


@tool(name="calculate_lcoe", description="Calculate Levelized Cost of Energy")
def calculate_lcoe(
    capital_cost: float,
    annual_generation: float,
    annual_operating_cost: float = 0.0,
    discount_rate: float = 0.08,
    lifetime_years: int = 25,
    degradation_rate: float = 0.005,
) -> dict:
    """Calculate LCOE with degradation modeling."""

    # Calculate present value of costs
    pv_capex = capital_cost
    pv_opex = 0
    pv_generation = 0

    for year in range(1, lifetime_years + 1):
        discount_factor = (1 + discount_rate) ** -year
        degradation_factor = (1 - degradation_rate) ** (year - 1)

        pv_opex += annual_operating_cost * discount_factor
        pv_generation += annual_generation * degradation_factor * discount_factor

    total_pv_costs = pv_capex + pv_opex
    lcoe = total_pv_costs / pv_generation

    return {
        "lcoe_per_mwh": round(lcoe, 2),
        "total_pv_costs": round(total_pv_costs, 0),
        "pv_generation": round(pv_generation, 0),
        "capex_share_percent": round(pv_capex / total_pv_costs * 100, 1),
        "opex_share_percent": round(pv_opex / total_pv_costs * 100, 1),
        "assumptions": {
            "discount_rate": discount_rate,
            "lifetime_years": lifetime_years,
            "degradation_rate": degradation_rate,
        },
    }


@tool(name="capacity_factor_analysis", description="Analyze capacity factor and performance")
def capacity_factor_analysis(
    technology: str,
    actual_generation_mwh: float,
    rated_capacity_mw: float,
    location: str = "Unknown",
) -> dict:
    """Analyze capacity factor with technology benchmarks."""

    max_generation = rated_capacity_mw * 8760  # 24/7 for year
    capacity_factor = actual_generation_mwh / max_generation

    # Technology benchmarks by region
    benchmarks = {
        "solar": {"excellent": 0.25, "good": 0.20, "average": 0.15, "poor": 0.10},
        "wind_onshore": {"excellent": 0.45, "good": 0.35, "average": 0.25, "poor": 0.15},
        "wind_offshore": {"excellent": 0.55, "good": 0.45, "average": 0.35, "poor": 0.25},
        "hydro": {"excellent": 0.60, "good": 0.45, "average": 0.35, "poor": 0.20},
    }

    tech_key = technology.lower().replace(" ", "_")
    benchmark = benchmarks.get(tech_key, benchmarks["solar"])

    if capacity_factor >= benchmark["excellent"]:
        performance_rating = "Excellent"
    elif capacity_factor >= benchmark["good"]:
        performance_rating = "Good"
    elif capacity_factor >= benchmark["average"]:
        performance_rating = "Average"
    else:
        performance_rating = "Below Average"

    return {
        "capacity_factor": round(capacity_factor, 3),
        "capacity_factor_percent": round(capacity_factor * 100, 1),
        "performance_rating": performance_rating,
        "technology": technology,
        "location": location,
        "benchmarks": benchmark,
        "analysis": f"{performance_rating} performance for {technology} technology",
        "recommendations": _get_performance_recommendations(capacity_factor, benchmark),
    }


def _get_performance_recommendations(cf: float, benchmarks: dict) -> str:
    """Get performance improvement recommendations."""
    if cf >= benchmarks["excellent"]:
        return "Exceptional performance. Consider expanding similar projects."
    elif cf >= benchmarks["good"]:
        return "Good performance. Monitor for optimization opportunities."
    elif cf >= benchmarks["average"]:
        return "Average performance. Consider operational improvements."
    else:
        return "Below average performance. Investigate optimization strategies."


@tool(name="financial_metrics", description="Calculate comprehensive financial metrics")
def financial_metrics(
    initial_investment: float, annual_cash_flows: list, discount_rate: float = 0.08
) -> dict:
    """Calculate NPV, IRR, and payback period."""

    # NPV calculation
    npv = -initial_investment
    for i, cash_flow in enumerate(annual_cash_flows):
        npv += cash_flow / ((1 + discount_rate) ** (i + 1))

    # Simple payback period
    cumulative = 0
    payback_years = None
    for i, cash_flow in enumerate(annual_cash_flows):
        cumulative += cash_flow
        if cumulative >= initial_investment and payback_years is None:
            payback_years = i + 1 + (initial_investment - (cumulative - cash_flow)) / cash_flow

    # IRR estimation (simplified)
    total_returns = sum(annual_cash_flows)
    years = len(annual_cash_flows)
    irr = None
    if years > 0 and total_returns > initial_investment:
        irr = ((total_returns / initial_investment) ** (1 / years)) - 1

    return {
        "npv": round(npv, 0),
        "irr": round(irr * 100, 2) if irr else None,
        "payback_years": round(payback_years, 1) if payback_years else None,
        "total_cash_flows": sum(annual_cash_flows),
        "roi_percent": round((total_returns - initial_investment) / initial_investment * 100, 1),
        "investment_recommendation": "Proceed" if npv > 0 else "Reconsider",
        "risk_level": "Low" if npv > initial_investment * 0.2 else "Medium" if npv > 0 else "High",
    }


# ==============================================================================
# SPECIALIZED AGENTS USING DECORATORS
# ==============================================================================


@agent(
    name="FinancialAnalyst",
    description="Expert financial analyst specializing in energy project economics",
    system_prompt="""You are a senior financial analyst with deep expertise in energy project finance.

    Your core competencies include:
    - LCOE (Levelized Cost of Energy) calculations and analysis
    - NPV, IRR, and payback period analysis
    - Investment risk assessment and financial modeling
    - Capital structure optimization
    - Revenue forecasting and cash flow modeling

    Always provide detailed financial analysis with clear recommendations and risk assessments.
    Use your financial tools to support your analysis with concrete numbers.""",
    tools=["calculate_lcoe", "financial_metrics"],
)
class FinancialAnalyst:
    """Expert financial analyst for energy projects."""

    temperature = 0.3  # Lower temperature for consistent financial analysis
    max_tokens = 1000


@agent(
    name="TechnicalAnalyst",
    description="Technical performance specialist for renewable energy systems",
    system_prompt="""You are a technical expert in renewable energy systems and performance analysis.

    Your expertise includes:
    - Capacity factor analysis and performance benchmarking
    - Technology-specific performance optimization
    - Technical risk assessment
    - Equipment and system performance evaluation
    - Operational efficiency analysis

    Provide detailed technical analysis with actionable recommendations for performance improvement.
    Use your analytical tools to support your assessments with data.""",
    tools=["capacity_factor_analysis"],
)
class TechnicalAnalyst:
    """Technical performance specialist for renewable energy."""

    temperature = 0.3
    max_tokens = 1000


@agent(
    name="MarketAnalyst",
    description="Energy market analysis and competitive intelligence specialist",
    system_prompt="""You are a market analyst specializing in renewable energy markets and competitive analysis.

    Your expertise includes:
    - Market trend analysis and forecasting
    - Competitive landscape assessment
    - Regulatory and policy impact analysis
    - Price forecasting and market timing
    - Investment opportunity evaluation

    Provide comprehensive market insights with strategic recommendations.
    Focus on market dynamics, competitive positioning, and growth opportunities.""",
)
class MarketAnalyst:
    """Energy market analysis specialist."""

    temperature = 0.4
    max_tokens = 1000


# ==============================================================================
# MASTER AGENT FOR COORDINATION
# ==============================================================================


@master_agent(
    name="EnergyProjectMaster",
    description="Master coordinator for comprehensive energy project analysis",
    system_prompt="""You are the Master Energy Project Analyst, coordinating comprehensive analysis of renewable energy projects.

    Your role is to:
    1. Orchestrate analysis across financial, technical, and market dimensions
    2. Synthesize insights from specialist agents
    3. Provide integrated recommendations considering all aspects
    4. Identify synergies and trade-offs between different analysis areas
    5. Present executive-level summaries with clear action items

    You coordinate these specialist agents:
    - FinancialAnalyst: For economic and investment analysis
    - TechnicalAnalyst: For performance and technical assessment
    - MarketAnalyst: For market dynamics and competitive analysis

    Always provide a comprehensive, well-structured analysis that integrates multiple perspectives.""",
    subordinates=["FinancialAnalyst", "TechnicalAnalyst", "MarketAnalyst"],
)
class EnergyProjectMaster:
    """Master coordinator for energy project analysis."""

    temperature = 0.4
    max_iterations = 3
    selection_strategy = "prompt"  # Let AI choose which subordinate to use


# ==============================================================================
# CONFIGURATION AND TESTING
# ==============================================================================


def create_energy_analysis_config():
    """Create configuration for energy analysis agents."""
    return {
        "deployment_name": "gpt-4o",
        "endpoint": "https://your-endpoint.openai.azure.com/",
        "api_key": "your-api-key-here",
        "api_version": "2024-02-01",
    }


async def test_specialist_agents():
    """Test individual specialist agents."""

    initialize_sdk(log_level="INFO")

    print("🧪 Testing Specialist Agent Tools")
    print("=" * 60)

    # Test LCOE calculation
    print("\n💰 Testing LCOE Analysis:")
    lcoe_result = calculate_lcoe(
        capital_cost=150_000_000,  # $150M solar project
        annual_generation=250_000,  # 250,000 MWh/year
        annual_operating_cost=2_000_000,  # $2M/year O&M
        discount_rate=0.08,
        lifetime_years=25,
        degradation_rate=0.005,
    )
    print(f"LCOE Analysis: {lcoe_result}")

    # Test capacity factor analysis
    print("\n⚡ Testing Technical Analysis:")
    cf_result = capacity_factor_analysis(
        technology="solar",
        actual_generation_mwh=250_000,
        rated_capacity_mw=100,
        location="California",
    )
    print(f"Capacity Factor Analysis: {cf_result}")

    # Test financial metrics
    print("\n📊 Testing Financial Metrics:")
    cash_flows = [10_000_000] * 25  # $10M annually for 25 years
    fm_result = financial_metrics(
        initial_investment=150_000_000, annual_cash_flows=cash_flows, discount_rate=0.08
    )
    print(f"Financial Metrics: {fm_result}")

    print("\n✅ All tools working correctly!")
    print("🤖 For AI-powered analysis, configure API keys and run with --mode ai-test")


async def test_with_ai():
    """Test the master agent system with AI."""

    initialize_sdk(log_level="INFO")

    config = create_energy_analysis_config()

    print("🤖 Testing Master Agent System with AI")
    print("=" * 60)
    print("⚠️  Note: This requires valid Azure OpenAI credentials")

    try:
        # Bootstrap all agents
        agents = bootstrap_agents(azure_openai_config=config)

        if "EnergyProjectMaster" not in agents:
            print("❌ Master agent not created. Check configuration.")
            return

        master_agent = agents["EnergyProjectMaster"]
        print("✅ Master agent system created successfully!")
        print(f"📋 Available agents: {list(agents.keys())}")

        # Test comprehensive analysis request
        analysis_request = AgentRequest(
            message="""
            Please provide a comprehensive analysis of a 100MW solar project with the following specifications:

            - Capital cost: $150 million
            - Expected annual generation: 250,000 MWh
            - Annual O&M cost: $2 million
            - Location: California
            - Project lifetime: 25 years

            I need your full analysis covering financial viability, technical performance, and market positioning.
            """,
            agent_id="EnergyProjectMaster",
            session_id="comprehensive_analysis",
        )

        print(f"\n📝 Analysis Request: {analysis_request.message[:100]}...")

        response = await master_agent.process_request(analysis_request)

        print("\n🎯 Master Agent Analysis:")
        print(f"Response: {response.content}")
        print(f"Execution time: {response.execution_time_ms}ms")

        if response.error:
            print(f"❌ Error: {response.error}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to configure your Azure OpenAI credentials")


def run_development_server():
    """Run the master agent system in development server."""

    if not APPLICATION_AVAILABLE:
        print("❌ Web server not available. Install FastAPI and uvicorn")
        return

    initialize_sdk(log_level="INFO")
    config = create_energy_analysis_config()

    try:
        # Bootstrap all agents
        agents = bootstrap_agents(azure_openai_config=config)

        print("🌐 Starting Master Agent Development Server...")
        print("📍 Available at: http://localhost:8000")
        print(f"🤖 Agents available: {list(agents.keys())}")

        # Create application with all agents
        app = create_application(
            title="Energy Analysis Platform",
            description="Master agent system for comprehensive energy project analysis",
        )

        # Add all agents to the application
        for agent in agents.values():
            app.add_agent(agent)

        # Start development server
        server = DevelopmentServer(app, port=8000)
        server.run()

    except Exception as e:
        print(f"❌ Error starting server: {e}")


def main():
    """Main function to run the master agent example."""
    import argparse

    parser = argparse.ArgumentParser(description="Master Agent Example")
    parser.add_argument(
        "--mode",
        choices=["test", "ai-test", "server"],
        default="test",
        help="Run mode: 'test' for tool testing, 'ai-test' for AI testing, 'server' for development server",
    )

    args = parser.parse_args()

    if args.mode == "test":
        # Test tools without AI
        asyncio.run(test_specialist_agents())
    elif args.mode == "ai-test":
        # Test with AI capabilities
        asyncio.run(test_with_ai())
    else:
        # Run development server
        try:
            run_development_server()
        except KeyboardInterrupt:
            print("\n👋 Shutting down server...")
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            print("\n🔧 Make sure you have:")
            print("1. Configured Azure OpenAI credentials in create_energy_analysis_config()")
            print("2. Installed required dependencies: pip install fastapi uvicorn")


if __name__ == "__main__":
    main()
