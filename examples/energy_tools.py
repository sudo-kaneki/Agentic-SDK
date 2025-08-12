# examples/energy_tools.py
"""
Energy-Specific Tools Example

This example demonstrates specialized tools for energy analysis,
including LCOE calculations, capacity factor analysis, and carbon emissions.
"""

import sys
from pathlib import Path
from typing import Any

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk import initialize_sdk, tool


@tool(
    name="calculate_lcoe", description="Calculate Levelized Cost of Energy for renewable projects"
)
def calculate_lcoe(
    capital_cost: float,
    annual_generation: float,
    annual_operating_cost: float = 0.0,
    discount_rate: float = 0.08,
    lifetime_years: int = 25,
    degradation_rate: float = 0.005,
) -> dict[str, float]:
    """
    Calculate LCOE for renewable energy projects with advanced parameters.

    Args:
        capital_cost: Initial capital investment ($)
        annual_generation: First year energy generation (MWh)
        annual_operating_cost: Annual operation and maintenance cost ($)
        discount_rate: Discount rate for NPV calculation (default 8%)
        lifetime_years: Project lifetime in years (default 25)
        degradation_rate: Annual degradation rate (default 0.5% for solar)

    Returns:
        Dictionary with comprehensive LCOE analysis
    """
    total_generation = 0
    total_costs = capital_cost

    # Calculate year-by-year generation and costs
    for year in range(1, lifetime_years + 1):
        # Apply degradation to generation
        yearly_generation = annual_generation * ((1 - degradation_rate) ** (year - 1))

        # Discount generation and costs to present value
        pv_generation = yearly_generation / ((1 + discount_rate) ** year)
        pv_operating_cost = annual_operating_cost / ((1 + discount_rate) ** year)

        total_generation += pv_generation
        total_costs += pv_operating_cost

    # Calculate LCOE
    lcoe = total_costs / total_generation if total_generation > 0 else 0

    # Additional metrics
    capacity_utilization = annual_generation / (365.25 * 24)  # Assuming 1 MW capacity
    annual_revenue_required = lcoe * annual_generation

    return {
        "lcoe_per_mwh": round(lcoe, 2),
        "total_present_value_costs": round(total_costs, 2),
        "total_present_value_generation": round(total_generation, 2),
        "capital_cost_share": round(capital_cost / total_costs * 100, 2),
        "operating_cost_share": round((total_costs - capital_cost) / total_costs * 100, 2),
        "annual_revenue_required": round(annual_revenue_required, 2),
        "capacity_utilization_hours": round(capacity_utilization, 0),
        "lifetime_years": lifetime_years,
        "discount_rate_percent": discount_rate * 100,
        "degradation_rate_percent": degradation_rate * 100,
    }


@tool(
    name="capacity_factor_analysis",
    description="Analyze capacity factor performance for renewable energy systems",
)
def capacity_factor_analysis(
    technology: str,
    location_type: str,
    actual_capacity_factor: float,
    rated_capacity_mw: float,
    annual_generation_mwh: float,
) -> dict[str, Any]:
    """
    Comprehensive capacity factor analysis with benchmarking.

    Args:
        technology: Technology type (solar, wind_onshore, wind_offshore, hydro)
        location_type: Location characteristics (desert, coastal, mountain, etc.)
        actual_capacity_factor: Measured capacity factor (0-1)
        rated_capacity_mw: Rated capacity in MW
        annual_generation_mwh: Actual annual generation in MWh

    Returns:
        Detailed capacity factor analysis with benchmarks
    """
    # Technology and location benchmarks
    benchmarks = {
        "solar": {
            "desert": {"excellent": 0.28, "good": 0.23, "average": 0.18},
            "coastal": {"excellent": 0.22, "good": 0.18, "average": 0.14},
            "mountain": {"excellent": 0.25, "good": 0.20, "average": 0.15},
            "default": {"excellent": 0.25, "good": 0.20, "average": 0.15},
        },
        "wind_onshore": {
            "coastal": {"excellent": 0.50, "good": 0.40, "average": 0.30},
            "mountain": {"excellent": 0.45, "good": 0.35, "average": 0.25},
            "plains": {"excellent": 0.45, "good": 0.35, "average": 0.25},
            "default": {"excellent": 0.40, "good": 0.30, "average": 0.20},
        },
        "wind_offshore": {
            "shallow": {"excellent": 0.55, "good": 0.45, "average": 0.35},
            "deep": {"excellent": 0.60, "good": 0.50, "average": 0.40},
            "default": {"excellent": 0.50, "good": 0.40, "average": 0.30},
        },
        "hydro": {
            "run_of_river": {"excellent": 0.50, "good": 0.35, "average": 0.25},
            "reservoir": {"excellent": 0.60, "good": 0.45, "average": 0.30},
            "pumped_storage": {"excellent": 0.40, "good": 0.30, "average": 0.20},
            "default": {"excellent": 0.50, "good": 0.35, "average": 0.25},
        },
    }

    # Get appropriate benchmark
    tech_benchmarks = benchmarks.get(technology.lower(), benchmarks["solar"])
    location_benchmark = tech_benchmarks.get(location_type.lower(), tech_benchmarks["default"])

    # Calculate theoretical maximum generation
    theoretical_max_mwh = rated_capacity_mw * 8760  # 24/7 for a year
    calculated_cf = annual_generation_mwh / theoretical_max_mwh

    # Performance rating
    if actual_capacity_factor >= location_benchmark["excellent"]:
        performance_rating = "Excellent"
        percentile = 90
    elif actual_capacity_factor >= location_benchmark["good"]:
        performance_rating = "Good"
        percentile = 70
    elif actual_capacity_factor >= location_benchmark["average"]:
        performance_rating = "Average"
        percentile = 50
    else:
        performance_rating = "Below Average"
        percentile = 25

    # Efficiency metrics
    availability_factor = (
        calculated_cf / actual_capacity_factor if actual_capacity_factor > 0 else 1.0
    )

    return {
        "capacity_factor_actual": round(actual_capacity_factor, 4),
        "capacity_factor_calculated": round(calculated_cf, 4),
        "capacity_factor_percent": round(actual_capacity_factor * 100, 2),
        "performance_rating": performance_rating,
        "performance_percentile": percentile,
        "annual_generation_mwh": annual_generation_mwh,
        "theoretical_max_mwh": theoretical_max_mwh,
        "generation_efficiency_percent": round(
            calculated_cf / location_benchmark["excellent"] * 100, 1
        ),
        "benchmark_excellent": location_benchmark["excellent"],
        "benchmark_good": location_benchmark["good"],
        "benchmark_average": location_benchmark["average"],
        "availability_factor": round(availability_factor, 4),
        "technology": technology,
        "location_type": location_type,
        "rated_capacity_mw": rated_capacity_mw,
    }


@tool(
    name="carbon_footprint_calculator",
    description="Calculate carbon emissions and offset potential for energy projects",
)
def carbon_footprint_calculator(
    energy_source: str,
    annual_generation_mwh: float,
    displaced_source: str = "grid_average",
    project_lifetime_years: int = 25,
    include_lifecycle: bool = True,
) -> dict[str, Any]:
    """
    Comprehensive carbon footprint analysis for energy projects.

    Args:
        energy_source: Primary energy source
        annual_generation_mwh: Annual energy generation
        displaced_source: Energy source being displaced
        project_lifetime_years: Project lifetime for total calculations
        include_lifecycle: Include manufacturing and decommissioning emissions

    Returns:
        Detailed carbon footprint analysis
    """
    # Emission factors in kg CO2/MWh (operational)
    operational_emissions = {
        "coal": 820,
        "natural_gas": 490,
        "oil": 650,
        "grid_average_us": 400,
        "grid_average_eu": 300,
        "grid_average": 350,
        "solar_pv": 40,
        "wind_onshore": 11,
        "wind_offshore": 12,
        "hydro": 24,
        "nuclear": 12,
        "biomass": 230,
        "geothermal": 38,
    }

    # Lifecycle emissions (including manufacturing, installation, decommissioning)
    lifecycle_emissions = {
        "coal": 1050,
        "natural_gas": 550,
        "oil": 750,
        "grid_average_us": 500,
        "grid_average_eu": 400,
        "grid_average": 450,
        "solar_pv": 85,
        "wind_onshore": 26,
        "wind_offshore": 28,
        "hydro": 48,
        "nuclear": 66,
        "biomass": 280,
        "geothermal": 56,
    }

    # Choose emission factors based on lifecycle inclusion
    emission_factors = lifecycle_emissions if include_lifecycle else operational_emissions

    source_emissions_factor = emission_factors.get(energy_source.lower(), 0)
    displaced_emissions_factor = emission_factors.get(displaced_source.lower(), 400)

    # Annual calculations
    annual_source_emissions = source_emissions_factor * annual_generation_mwh
    annual_displaced_emissions = displaced_emissions_factor * annual_generation_mwh
    annual_net_avoided = annual_displaced_emissions - annual_source_emissions

    # Lifetime calculations
    lifetime_source_emissions = annual_source_emissions * project_lifetime_years
    lifetime_displaced_emissions = annual_displaced_emissions * project_lifetime_years
    lifetime_net_avoided = annual_net_avoided * project_lifetime_years

    # Convert to tons and calculate equivalent metrics
    annual_net_avoided_tons = annual_net_avoided / 1000
    lifetime_net_avoided_tons = lifetime_net_avoided / 1000

    # Equivalent metrics (approximate)
    cars_off_road_equivalent = lifetime_net_avoided_tons / 4.6  # Average car emissions per year
    trees_planted_equivalent = lifetime_net_avoided_tons / 0.022  # CO2 absorbed by tree per year

    return {
        "annual_analysis": {
            "source_emissions_kg_co2": round(annual_source_emissions, 2),
            "displaced_emissions_kg_co2": round(annual_displaced_emissions, 2),
            "net_emissions_avoided_kg_co2": round(annual_net_avoided, 2),
            "net_emissions_avoided_tons_co2": round(annual_net_avoided_tons, 2),
        },
        "lifetime_analysis": {
            "source_emissions_kg_co2": round(lifetime_source_emissions, 2),
            "displaced_emissions_kg_co2": round(lifetime_displaced_emissions, 2),
            "net_emissions_avoided_kg_co2": round(lifetime_net_avoided, 2),
            "net_emissions_avoided_tons_co2": round(lifetime_net_avoided_tons, 2),
        },
        "emission_factors": {
            "source_kg_co2_per_mwh": source_emissions_factor,
            "displaced_kg_co2_per_mwh": displaced_emissions_factor,
            "lifecycle_included": include_lifecycle,
        },
        "equivalents": {
            "cars_off_road_equivalent": round(cars_off_road_equivalent, 0),
            "trees_planted_equivalent": round(trees_planted_equivalent, 0),
        },
        "project_parameters": {
            "energy_source": energy_source,
            "displaced_source": displaced_source,
            "annual_generation_mwh": annual_generation_mwh,
            "project_lifetime_years": project_lifetime_years,
        },
    }


@tool(
    name="financial_metrics_calculator",
    description="Calculate comprehensive financial metrics for energy projects",
)
def financial_metrics_calculator(
    capital_cost: float,
    annual_cash_flows: list[float],
    discount_rate: float = 0.08,
    electricity_price_per_mwh: float = 50.0,
    annual_generation_mwh: float = 0,
) -> dict[str, Any]:
    """
    Calculate NPV, IRR, payback period, and other financial metrics.

    Args:
        capital_cost: Initial investment
        annual_cash_flows: List of annual cash flows
        discount_rate: Discount rate for NPV
        electricity_price_per_mwh: Electricity selling price
        annual_generation_mwh: Annual generation for revenue calculation

    Returns:
        Comprehensive financial analysis
    """
    # NPV calculation
    npv = -capital_cost
    for i, cash_flow in enumerate(annual_cash_flows):
        npv += cash_flow / ((1 + discount_rate) ** (i + 1))

    # Payback period calculation
    cumulative_cash_flow = -capital_cost
    simple_payback = None
    discounted_payback = None

    for i, cash_flow in enumerate(annual_cash_flows):
        year = i + 1
        cumulative_cash_flow += cash_flow

        # Simple payback
        if simple_payback is None and cumulative_cash_flow >= 0:
            simple_payback = year - (cumulative_cash_flow - cash_flow) / cash_flow

        # Discounted payback
        cash_flow / ((1 + discount_rate) ** year)
        if discounted_payback is None:
            # This is simplified - would need cumulative discounted CF tracking
            pass

    # IRR calculation (simplified Newton-Raphson method)
    def npv_at_rate(rate):
        result = -capital_cost
        for i, cf in enumerate(annual_cash_flows):
            result += cf / ((1 + rate) ** (i + 1))
        return result

    # Simple IRR approximation
    irr = None
    for rate in [i * 0.01 for i in range(1, 50)]:  # Test rates from 1% to 49%
        if abs(npv_at_rate(rate)) < 100:  # Close to zero
            irr = rate
            break

    # Additional metrics
    total_cash_flows = sum(annual_cash_flows)
    profit_index = (npv + capital_cost) / capital_cost if capital_cost > 0 else 0

    # Revenue metrics if generation data provided
    annual_revenue = (
        annual_generation_mwh * electricity_price_per_mwh if annual_generation_mwh > 0 else 0
    )

    return {
        "npv": round(npv, 2),
        "irr_percent": round(irr * 100, 2) if irr else None,
        "simple_payback_years": round(simple_payback, 2) if simple_payback else None,
        "profit_index": round(profit_index, 3),
        "total_cash_flows": round(total_cash_flows, 2),
        "capital_cost": capital_cost,
        "discount_rate_percent": discount_rate * 100,
        "project_lifetime_years": len(annual_cash_flows),
        "revenue_analysis": (
            {
                "annual_revenue": round(annual_revenue, 2),
                "electricity_price_per_mwh": electricity_price_per_mwh,
                "annual_generation_mwh": annual_generation_mwh,
                "lifetime_revenue": round(annual_revenue * len(annual_cash_flows), 2),
            }
            if annual_generation_mwh > 0
            else None
        ),
        "investment_metrics": {
            "npv_positive": npv > 0,
            "irr_exceeds_discount_rate": (irr > discount_rate) if irr else None,
            "payback_reasonable": (simple_payback < 10) if simple_payback else None,
        },
    }


def demonstrate_tools():
    """Demonstrate all energy tools with sample calculations."""

    print("Energy Tools Demonstration")
    print("=" * 50)

    # Initialize SDK
    initialize_sdk(log_level="INFO")

    # LCOE Example
    print("\n1. LCOE Calculation for 100MW Solar Farm")
    print("-" * 40)
    lcoe_result = calculate_lcoe(
        capital_cost=150_000_000,  # $150M
        annual_generation=250_000,  # 250 GWh
        annual_operating_cost=2_000_000,  # $2M per year
        discount_rate=0.08,
        lifetime_years=25,
        degradation_rate=0.005,
    )

    for key, value in lcoe_result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Capacity Factor Analysis
    print("\n2. Capacity Factor Analysis for Offshore Wind")
    print("-" * 40)
    cf_result = capacity_factor_analysis(
        technology="wind_offshore",
        location_type="shallow",
        actual_capacity_factor=0.45,
        rated_capacity_mw=500,
        annual_generation_mwh=1_971_000,  # 500 MW * 0.45 CF * 8760 hours
    )

    for key, value in cf_result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Carbon Footprint
    print("\n3. Carbon Footprint Analysis")
    print("-" * 40)
    carbon_result = carbon_footprint_calculator(
        energy_source="wind_offshore",
        annual_generation_mwh=1_971_000,
        displaced_source="grid_average",
        project_lifetime_years=25,
        include_lifecycle=True,
    )

    print("Annual Analysis:")
    for key, value in carbon_result["annual_analysis"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\nLifetime Analysis:")
    for key, value in carbon_result["lifetime_analysis"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\nEquivalents:")
    for key, value in carbon_result["equivalents"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # Financial Metrics
    print("\n4. Financial Metrics Analysis")
    print("-" * 40)

    # Generate sample cash flows (revenue - operating costs)
    annual_revenue = 1_971_000 * 60  # 250 GWh * $60/MWh
    annual_operating_cost = 2_000_000
    annual_cash_flow = annual_revenue - annual_operating_cost
    cash_flows = [annual_cash_flow] * 25  # 25 years

    financial_result = financial_metrics_calculator(
        capital_cost=500_000_000,  # $500M for offshore wind
        annual_cash_flows=cash_flows,
        discount_rate=0.08,
        electricity_price_per_mwh=60.0,
        annual_generation_mwh=1_971_000,
    )

    for key, value in financial_result.items():
        if key != "revenue_analysis":
            print(f"{key.replace('_', ' ').title()}: {value}")

    if financial_result["revenue_analysis"]:
        print("\nRevenue Analysis:")
        for key, value in financial_result["revenue_analysis"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    demonstrate_tools()
