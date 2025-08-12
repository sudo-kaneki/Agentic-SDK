# examples/decorator_concept_demo.py
"""
Decorator Concept Demo

This demonstrates how the EnergyAI SDK decorator-based approach works
conceptually, even without Semantic Kernel installed.

This shows the power of your vision - users define agents declaratively,
and the SDK handles all the complexity behind the scenes.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk import agent, agent_registry, initialize_sdk, planner, prompt, skill, tool
from energyai_sdk.core import get_available_features

# ==============================================================================
# STEP 1: Create Tools with Decorators
# ==============================================================================


@tool(name="calculate_lcoe", description="Calculate Levelized Cost of Energy")
def calculate_lcoe(
    capital_cost: float,
    annual_generation: float,
    annual_operating_cost: float = 0.0,
    discount_rate: float = 0.08,
    lifetime_years: int = 25,
) -> dict:
    """Calculate LCOE for renewable energy projects."""
    # Present value calculations
    pv_factor = sum([(1 + discount_rate) ** -i for i in range(1, lifetime_years + 1)])
    pv_operation_costs = annual_operating_cost * pv_factor
    total_pv_costs = capital_cost + pv_operation_costs
    pv_generation = annual_generation * pv_factor
    lcoe = total_pv_costs / pv_generation

    return {
        "lcoe_per_mwh": round(lcoe, 2),
        "total_pv_costs": total_pv_costs,
        "pv_generation": pv_generation,
        "capital_cost_share": round(capital_cost / total_pv_costs * 100, 1),
        "operation_cost_share": round(pv_operation_costs / total_pv_costs * 100, 1),
    }


@tool(name="capacity_factor", description="Calculate renewable energy capacity factor")
def calculate_capacity_factor(
    technology: str, rated_capacity_mw: float, actual_generation_mwh: float
) -> dict:
    """Calculate and evaluate capacity factor."""
    max_theoretical = rated_capacity_mw * 8760  # 24/7 for a year
    capacity_factor = actual_generation_mwh / max_theoretical

    # Technology benchmarks
    benchmarks = {
        "solar": {"excellent": 0.25, "good": 0.20, "average": 0.15},
        "wind_onshore": {"excellent": 0.45, "good": 0.35, "average": 0.25},
        "wind_offshore": {"excellent": 0.55, "good": 0.45, "average": 0.35},
        "hydro": {"excellent": 0.60, "good": 0.45, "average": 0.35},
    }

    tech_key = technology.lower().replace(" ", "_")
    benchmark = benchmarks.get(tech_key, benchmarks["solar"])

    if capacity_factor >= benchmark["excellent"]:
        performance = "Excellent"
    elif capacity_factor >= benchmark["good"]:
        performance = "Good"
    elif capacity_factor >= benchmark["average"]:
        performance = "Average"
    else:
        performance = "Below Average"

    return {
        "capacity_factor": round(capacity_factor, 3),
        "capacity_factor_percent": round(capacity_factor * 100, 1),
        "performance_rating": performance,
        "benchmark_excellent": benchmark["excellent"],
        "annual_generation_mwh": actual_generation_mwh,
        "theoretical_max_mwh": max_theoretical,
        "technology": technology,
    }


# ==============================================================================
# STEP 2: Create Skills (Collections of Tools)
# ==============================================================================


@skill(name="EnergyFinance", description="Financial analysis tools for energy projects")
class EnergyFinance:
    """Collection of financial analysis tools for energy investments."""

    @tool(name="npv_calculator")
    def calculate_npv(
        self, initial_investment: float, annual_cash_flows: list, discount_rate: float = 0.08
    ) -> dict:
        """Calculate Net Present Value of energy project."""
        npv = -initial_investment
        for i, cash_flow in enumerate(annual_cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** (i + 1))

        irr = self._calculate_irr(initial_investment, annual_cash_flows)

        return {
            "npv": round(npv, 2),
            "initial_investment": initial_investment,
            "discount_rate": discount_rate,
            "total_cash_flows": sum(annual_cash_flows),
            "years": len(annual_cash_flows),
            "irr": round(irr, 3) if irr else None,
            "profitable": npv > 0,
        }

    def _calculate_irr(self, initial_investment: float, cash_flows: list) -> float:
        """Simple IRR calculation using Newton's method."""
        # Simplified IRR calculation for demo
        total_returns = sum(cash_flows)
        years = len(cash_flows)
        if years > 0 and total_returns > initial_investment:
            return ((total_returns / initial_investment) ** (1 / years)) - 1
        return 0.0

    @tool(name="payback_period")
    def calculate_payback_period(self, initial_investment: float, annual_cash_flows: list) -> dict:
        """Calculate simple and discounted payback periods."""
        cumulative = 0
        simple_payback = None

        for i, cash_flow in enumerate(annual_cash_flows):
            cumulative += cash_flow
            if cumulative >= initial_investment and simple_payback is None:
                # Linear interpolation for exact payback time
                previous_cumulative = cumulative - cash_flow
                simple_payback = i + (initial_investment - previous_cumulative) / cash_flow

        return {
            "simple_payback_years": round(simple_payback, 2) if simple_payback else None,
            "cumulative_cash_flow": cumulative,
            "initial_investment": initial_investment,
            "payback_achieved": cumulative >= initial_investment,
        }


# ==============================================================================
# STEP 3: Create Prompts
# ==============================================================================


@prompt(name="energy_analysis", variables=["project_type", "capacity", "location"])
def energy_analysis_prompt():
    """
    Analyze the {project_type} project with {capacity} MW capacity in {location}.

    Please provide a comprehensive analysis covering:
    1. Technical feasibility and performance expectations
    2. Financial viability including LCOE and NPV analysis
    3. Market conditions and competitive positioning
    4. Risk assessment and mitigation strategies
    5. Regulatory and policy considerations

    Use available tools to calculate specific metrics and support your analysis.
    """


@prompt(name="investment_recommendation", variables=["project_name", "npv", "irr", "lcoe"])
def investment_recommendation_prompt():
    """
    Based on the financial analysis for {project_name}:
    - NPV: ${npv}
    - IRR: {irr}%
    - LCOE: ${lcoe}/MWh

    Provide a clear investment recommendation with:
    1. Go/No-Go decision with rationale
    2. Key risk factors and mitigation strategies
    3. Sensitivity analysis recommendations
    4. Market timing considerations
    """


# ==============================================================================
# STEP 4: Create Planners
# ==============================================================================


@planner(name="energy_project_planner", planner_type="sequential")
class EnergyProjectPlanner:
    """Sequential planner for energy project analysis."""

    def plan(self, project_request: str) -> list:
        """Create analysis plan based on project request."""
        return [
            "gather_project_specifications",
            "calculate_technical_metrics",
            "perform_financial_analysis",
            "assess_market_conditions",
            "evaluate_risks_and_opportunities",
            "generate_investment_recommendation",
        ]


# ==============================================================================
# STEP 5: Define Agent Classes (Conceptual - Full Implementation Requires SK)
# ==============================================================================


@agent(
    name="EnergyAnalyst",
    description="Expert in energy project financial analysis and LCOE calculations",
    system_prompt="You are an expert energy financial analyst...",
    tools=["calculate_lcoe"],
    skills=["EnergyFinance"],
)
class EnergyAnalyst:
    """
    Energy financial analysis specialist.

    This is a conceptual definition. With Semantic Kernel installed,
    this would automatically become a fully functional AI agent.
    """

    pass


@agent(
    name="TechnicalAnalyst",
    description="Expert in renewable energy technologies and performance",
    tools=["capacity_factor"],
)
class TechnicalAnalyst:
    """Technical specialist for renewable energy systems."""

    pass


@agent(
    name="InvestmentAdvisor",
    description="Investment analysis and recommendation specialist",
    skills=["EnergyFinance"],
)
class InvestmentAdvisor:
    """Investment advisory specialist for energy projects."""

    pass


# ==============================================================================
# DEMO FUNCTION
# ==============================================================================


def main():
    """Demonstrate the decorator-based SDK concept."""

    print("🚀 EnergyAI SDK - Decorator Concept Demo")
    print("=" * 60)

    # Initialize SDK
    initialize_sdk(log_level="INFO")

    # Show available features
    features = get_available_features()
    print(f"📋 Available Features: {features}")

    # Show registered components
    tools = agent_registry.tools
    skills = agent_registry.skills

    print(f"\n🔧 Registered Tools: {list(tools.keys())}")
    print(f"🎯 Registered Skills: {list(skills.keys())}")
    print(f"📝 Registered Prompts: {list(agent_registry.prompts.keys())}")
    print(
        f"🤖 Agent Classes Defined: {len([cls for cls in [EnergyAnalyst, TechnicalAnalyst, InvestmentAdvisor] if hasattr(cls, '_is_energyai_agent')])}"
    )

    print("\n" + "=" * 60)
    print("🧪 TESTING TOOLS DIRECTLY")
    print("=" * 60)

    # Test LCOE calculation
    print("\n💰 Testing LCOE calculation...")
    lcoe_result = calculate_lcoe(
        capital_cost=150_000_000,  # $150M
        annual_generation=250_000,  # 250,000 MWh/year
        annual_operating_cost=2_000_000,  # $2M/year
        discount_rate=0.08,
        lifetime_years=25,
    )

    print(f"LCOE Result: {lcoe_result}")

    # Test capacity factor
    print("\n⚡ Testing capacity factor calculation...")
    cf_result = calculate_capacity_factor(
        technology="solar", rated_capacity_mw=100, actual_generation_mwh=250_000
    )

    print(f"Capacity Factor Result: {cf_result}")

    # Test NPV calculation through skill
    print("\n📊 Testing NPV calculation (via skill)...")
    finance_skill = EnergyFinance()
    cash_flows = [8_000_000] * 25  # $8M annually for 25 years
    npv_result = finance_skill.calculate_npv(
        initial_investment=150_000_000, annual_cash_flows=cash_flows, discount_rate=0.08
    )

    print(f"NPV Result: {npv_result}")

    # Test payback period
    print("\n⏰ Testing payback period...")
    payback_result = finance_skill.calculate_payback_period(
        initial_investment=150_000_000, annual_cash_flows=cash_flows
    )

    print(f"Payback Result: {payback_result}")

    print("\n" + "=" * 60)
    print("🎯 WHAT HAPPENS WITH SEMANTIC KERNEL INSTALLED:")
    print("=" * 60)
    print(
        """
    1. Agent classes become REAL AI agents automatically
    2. Tools are integrated into agent capabilities
    3. Skills provide collections of related tools
    4. Prompts become executable templates
    5. Planners coordinate multi-step workflows

    🎉 Users just add decorators - SDK handles everything else!
    """
    )

    if not features["agents"]:
        print("💡 To see full agent functionality, install semantic-kernel:")
        print("   pip install semantic-kernel")


if __name__ == "__main__":
    main()
