# examples/declarative_agents.py
"""
Declarative Agent Creation Example

This example demonstrates the decorator-based approach to creating agents,
which is the core vision of the EnergyAI SDK - making agent creation as simple
as decorating a class.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk import (
    AgentRequest,
    bootstrap_agents,
    enhanced_agent,
    get_registered_agent_classes,
    initialize_sdk,
    master_agent,
    skill,
    tool,
)

# ==============================================================================
# STEP 1: Define Tools and Skills (as before)
# ==============================================================================


@tool(name="calculate_lcoe", description="Calculate Levelized Cost of Energy")
def calculate_lcoe(
    capital_cost: float,
    annual_generation: float,
    annual_operating_cost: float = 0.0,
    discount_rate: float = 0.08,
    lifetime_years: int = 25,
) -> dict[str, float]:
    """Calculate LCOE for renewable energy projects."""
    pv_factor = sum([(1 + discount_rate) ** -i for i in range(1, lifetime_years + 1)])
    pv_operation_costs = annual_operating_cost * pv_factor
    total_pv_costs = capital_cost + pv_operation_costs
    pv_generation = annual_generation * pv_factor
    lcoe = total_pv_costs / pv_generation

    return {
        "lcoe_per_mwh": lcoe,
        "total_pv_costs": total_pv_costs,
        "pv_generation": pv_generation,
        "capital_cost_share": capital_cost / total_pv_costs,
    }


@tool(name="capacity_factor", description="Calculate capacity factor")
def calculate_capacity_factor(
    technology: str, rated_capacity: float, actual_generation: float
) -> dict[str, Any]:
    """Calculate capacity factor for renewable systems."""
    max_theoretical = rated_capacity * 8760
    capacity_factor = actual_generation / max_theoretical

    benchmarks = {
        "solar": {"excellent": 0.25, "good": 0.20},
        "wind": {"excellent": 0.45, "good": 0.35},
        "hydro": {"excellent": 0.60, "good": 0.45},
    }

    benchmark = benchmarks.get(technology.lower(), benchmarks["solar"])
    performance = (
        "Excellent"
        if capacity_factor >= benchmark["excellent"]
        else "Good" if capacity_factor >= benchmark["good"] else "Average"
    )

    return {
        "capacity_factor": capacity_factor,
        "performance_rating": performance,
        "annual_generation_mwh": actual_generation,
    }


@skill(name="EnergyFinance", description="Financial analysis tools for energy projects")
class EnergyFinance:
    """Collection of financial analysis tools."""

    @tool(name="npv_calculator")
    def calculate_npv(
        self, initial_investment: float, annual_cash_flows: list, discount_rate: float = 0.08
    ) -> dict[str, float]:
        """Calculate Net Present Value."""
        npv = -initial_investment
        for i, cash_flow in enumerate(annual_cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** (i + 1))

        return {
            "npv": npv,
            "initial_investment": initial_investment,
            "discount_rate": discount_rate,
            "payback_positive": npv > 0,
        }


# ==============================================================================
# STEP 2: Define Agents Using Decorators (The Magic!)
# ==============================================================================


@enhanced_agent(
    name="EnergyAnalyst",
    description="Expert in energy financial analysis and LCOE calculations",
    tools=["calculate_lcoe"],
    skills=["EnergyFinance"],
)
class EnergyAnalyst:
    """
    Financial analysis specialist for energy projects.

    This agent is created automatically by the @enhanced_agent decorator.
    No need to manually instantiate or configure Semantic Kernel!
    """

    system_prompt = """
    You are an expert energy financial analyst with deep knowledge of:
    - LCOE (Levelized Cost of Energy) calculations
    - NPV analysis and financial modeling
    - Energy project economics and risk assessment
    - Renewable energy investment analysis

    Use the available tools to provide accurate calculations and data-driven insights.
    Always explain your methodology and key assumptions.
    """

    temperature = 0.3  # More deterministic for financial calculations
    max_tokens = 2000

    # Model configuration - will be provided at bootstrap time
    # azure_openai_config = {...} # Optional: can be defined here


@enhanced_agent(
    name="TechnicalAnalyst",
    description="Expert in renewable energy technologies and performance analysis",
    tools=["capacity_factor"],
)
class TechnicalAnalyst:
    """Technical specialist for renewable energy systems."""

    system_prompt = """
    You are a technical expert in renewable energy systems with expertise in:
    - Solar, wind, and hydroelectric technologies
    - Capacity factor analysis and performance optimization
    - Technical feasibility studies
    - System design and efficiency analysis

    Provide detailed technical insights and use calculations to support your analysis.
    """

    temperature = 0.4
    max_tokens = 1500


@enhanced_agent(
    name="MarketAnalyst", description="Specialist in energy markets and pricing analysis"
)
class MarketAnalyst:
    """Market analysis specialist for energy sector."""

    system_prompt = """
    You are an energy market analyst with expertise in:
    - Energy market dynamics and pricing trends
    - Policy and regulatory impact analysis
    - Competitive analysis and market positioning
    - Energy trading and risk management

    Provide market intelligence and strategic insights for energy investments.
    """

    temperature = 0.5  # More creative for market insights
    max_tokens = 1800


# ==============================================================================
# STEP 3: Define Master Agent (Coordinates Other Agents)
# ==============================================================================


@master_agent(
    name="EnergyMaster",
    description="Master coordinator for comprehensive energy analysis",
    subordinates=["EnergyAnalyst", "TechnicalAnalyst", "MarketAnalyst"],
    selection_strategy="prompt",
    max_iterations=3,
)
class EnergyMasterAgent:
    """
    Master agent that coordinates energy analysis tasks.

    This agent automatically routes requests to the most appropriate
    subordinate agent based on the query type.
    """

    system_prompt = """
    You are a master coordinator for energy analysis tasks. Your role is to:

    1. Analyze incoming requests and determine the best approach
    2. Select the most appropriate subordinate agent:
       - EnergyAnalyst: For financial analysis, LCOE, NPV calculations
       - TechnicalAnalyst: For technical performance, capacity factors
       - MarketAnalyst: For market trends, pricing, policy analysis
    3. Coordinate responses and provide comprehensive insights

    Always explain why you selected a particular agent and provide context.
    """

    temperature = 0.6  # Balanced for coordination decisions
    max_tokens = 2500


# ==============================================================================
# STEP 4: Bootstrap and Use the Agents
# ==============================================================================


async def main():
    """Main demonstration of declarative agent creation."""

    # Initialize the SDK
    initialize_sdk(log_level="INFO")

    print("🚀 EnergyAI SDK - Declarative Agent Creation Demo")
    print("=" * 60)

    # Show registered agent classes
    agent_classes = get_registered_agent_classes()
    print(f"📋 Registered Agent Classes: {agent_classes}")

    # Bootstrap agents with configuration
    # In a real application, these would come from environment variables or config files
    model_config = {
        "deployment_name": "gpt-4o",  # Replace with your model
        "endpoint": "https://your-endpoint.openai.azure.com/",  # Replace with your endpoint
        "api_key": "your-api-key-here",  # Replace with your API key
        "api_version": "2024-02-01",
    }

    print("\n🔧 Bootstrapping agents...")
    try:
        # This single call creates all agents defined with decorators!
        agents = bootstrap_agents(azure_openai_config=model_config)

        print(f"✅ Created {len(agents)} agents: {list(agents.keys())}")

        # Test individual agents
        print("\n💰 Testing EnergyAnalyst...")
        financial_request = AgentRequest(
            message="Calculate the LCOE for a 100MW solar farm with $150M capital cost, 250,000 MWh annual generation, and $2M annual operating costs.",
            agent_id="EnergyAnalyst",
            user_id="demo_user",
        )

        energy_analyst = agents["EnergyAnalyst"]
        response = await energy_analyst.process_request(financial_request)
        print(f"Response: {response.content}")
        print(f"Execution time: {response.execution_time_ms}ms")

        # Test technical analyst
        print("\n⚡ Testing TechnicalAnalyst...")
        technical_request = AgentRequest(
            message="Analyze the capacity factor for a wind farm with 200MW rated capacity generating 700,000 MWh annually.",
            agent_id="TechnicalAnalyst",
            user_id="demo_user",
        )

        technical_analyst = agents["TechnicalAnalyst"]
        response = await technical_analyst.process_request(technical_request)
        print(f"Response: {response.content}")

        # Test master agent coordination
        print("\n🎯 Testing EnergyMaster coordination...")
        master_request = AgentRequest(
            message="I'm considering investing in a 50MW solar project. Can you provide a comprehensive analysis covering financial viability, technical performance, and market conditions?",
            agent_id="EnergyMaster",
            user_id="demo_user",
        )

        energy_master = agents["EnergyMaster"]
        response = await energy_master.process_request(master_request)
        print(f"Master Response: {response.content}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Note: This demo requires valid Azure OpenAI credentials.")
        print("   Update the model_config with your actual endpoint and API key.")
        print("   For testing without credentials, use the mock examples in other files.")


if __name__ == "__main__":
    print("🎯 Declarative Agent Creation - The EnergyAI SDK Way!")
    print("\nThis example shows how to create agents using just decorators:")
    print("1. @enhanced_agent - Creates individual specialized agents")
    print("2. @master_agent - Creates coordinator agents")
    print("3. bootstrap_agents() - Instantiates all agents with one call")
    print("\nNo manual Semantic Kernel configuration needed! 🎉")

    asyncio.run(main())
