# examples/energy_skills.py
"""
Energy Skills Example

This example demonstrates how to create skills that group related tools
and functionality for energy analysis.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energyai_sdk import initialize_sdk, skill, tool


@skill(
    name="EnergyEconomics", description="Comprehensive financial analysis tools for energy projects"
)
class EnergyEconomics:
    """
    Collection of financial analysis tools for energy projects.

    This skill provides tools for:
    - Net Present Value (NPV) calculations
    - Internal Rate of Return (IRR) analysis
    - Payback period calculations
    - Return on Investment (ROI) analysis
    - Sensitivity analysis
    """

    def __init__(self):
        self.default_discount_rate = 0.08
        self.default_tax_rate = 0.25

    @tool(name="calculate_npv")
    def calculate_npv(
        self,
        initial_investment: float,
        cash_flows: list[float],
        discount_rate: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Calculate Net Present Value of an energy project.

        Args:
            initial_investment: Initial capital investment
            cash_flows: List of annual cash flows
            discount_rate: Discount rate (defaults to class default)

        Returns:
            NPV analysis results
        """
        rate = discount_rate or self.default_discount_rate

        npv = -initial_investment
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + rate) ** (i + 1))

        # Additional metrics
        total_cash_flows = sum(cash_flows)
        profit_index = (
            (npv + initial_investment) / initial_investment if initial_investment > 0 else 0
        )

        return {
            "npv": round(npv, 2),
            "initial_investment": initial_investment,
            "total_cash_flows": round(total_cash_flows, 2),
            "profit_index": round(profit_index, 3),
            "discount_rate": rate,
            "project_years": len(cash_flows),
            "npv_positive": npv > 0,
        }

    @tool(name="calculate_irr")
    def calculate_irr(
        self, initial_investment: float, cash_flows: list[float], precision: float = 0.001
    ) -> dict[str, Any]:
        """
        Calculate Internal Rate of Return using iterative method.

        Args:
            initial_investment: Initial capital investment
            cash_flows: List of annual cash flows
            precision: Calculation precision

        Returns:
            IRR analysis results
        """

        def npv_at_rate(rate):
            result = -initial_investment
            for i, cf in enumerate(cash_flows):
                if rate == -1:  # Avoid division by zero
                    return float("inf")
                result += cf / ((1 + rate) ** (i + 1))
            return result

        # Use binary search to find IRR
        low_rate = -0.99
        high_rate = 5.0  # 500% max

        # Check if solution exists
        if npv_at_rate(low_rate) * npv_at_rate(high_rate) > 0:
            return {
                "irr": None,
                "irr_percent": None,
                "error": "No IRR solution found in range",
                "initial_investment": initial_investment,
                "cash_flows_sum": sum(cash_flows),
            }

        # Binary search for IRR
        for _ in range(1000):  # Max iterations
            mid_rate = (low_rate + high_rate) / 2
            npv_mid = npv_at_rate(mid_rate)

            if abs(npv_mid) < precision:
                return {
                    "irr": round(mid_rate, 6),
                    "irr_percent": round(mid_rate * 100, 2),
                    "initial_investment": initial_investment,
                    "cash_flows_sum": sum(cash_flows),
                    "iterations_used": _ + 1,
                    "irr_exceeds_10_percent": mid_rate > 0.10,
                }

            if npv_at_rate(low_rate) * npv_mid < 0:
                high_rate = mid_rate
            else:
                low_rate = mid_rate

        return {
            "irr": None,
            "irr_percent": None,
            "error": "IRR calculation did not converge",
            "initial_investment": initial_investment,
        }

    @tool(name="payback_analysis")
    def payback_analysis(
        self,
        initial_investment: float,
        annual_cash_flows: list[float],
        discount_rate: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Calculate simple and discounted payback periods.

        Args:
            initial_investment: Initial capital investment
            annual_cash_flows: List of annual cash flows
            discount_rate: Discount rate for discounted payback

        Returns:
            Payback analysis results
        """
        rate = discount_rate or self.default_discount_rate

        # Simple payback
        cumulative_simple = 0
        simple_payback = None

        # Discounted payback
        cumulative_discounted = 0
        discounted_payback = None

        for i, cash_flow in enumerate(annual_cash_flows):
            year = i + 1

            # Simple payback calculation
            cumulative_simple += cash_flow
            if simple_payback is None and cumulative_simple >= initial_investment:
                # Interpolate for fractional year
                excess = cumulative_simple - initial_investment
                simple_payback = year - (excess / cash_flow)

            # Discounted payback calculation
            discounted_cf = cash_flow / ((1 + rate) ** year)
            cumulative_discounted += discounted_cf
            if discounted_payback is None and cumulative_discounted >= initial_investment:
                # Interpolate for fractional year
                excess = cumulative_discounted - initial_investment
                discounted_payback = year - (excess / discounted_cf)

        return {
            "simple_payback_years": round(simple_payback, 2) if simple_payback else None,
            "discounted_payback_years": (
                round(discounted_payback, 2) if discounted_payback else None
            ),
            "initial_investment": initial_investment,
            "total_undiscounted_cash_flows": sum(annual_cash_flows),
            "discount_rate": rate,
            "payback_within_10_years": (simple_payback or float("inf")) <= 10,
            "discounted_payback_reasonable": (discounted_payback or float("inf")) <= 15,
        }


@skill(
    name="TechnicalPerformance",
    description="Technical performance analysis tools for renewable energy systems",
)
class TechnicalPerformance:
    """
    Collection of technical performance analysis tools.

    Provides tools for:
    - Capacity factor optimization
    - Performance ratio calculations
    - Degradation analysis
    - Availability analysis
    """

    def __init__(self):
        self.standard_test_conditions = {
            "solar_irradiance": 1000,  # W/m²
            "cell_temperature": 25,  # °C
            "air_mass": 1.5,
        }

    @tool(name="performance_ratio_calculator")
    def performance_ratio_calculator(
        self,
        actual_energy_output: float,
        expected_energy_output: float,
        nameplate_capacity: float,
        irradiance_data: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Calculate performance ratio and system efficiency metrics.

        Args:
            actual_energy_output: Measured energy output (kWh)
            expected_energy_output: Expected output under standard conditions (kWh)
            nameplate_capacity: Nameplate capacity (kW)
            irradiance_data: Optional irradiance measurements

        Returns:
            Performance ratio analysis
        """
        performance_ratio = (
            actual_energy_output / expected_energy_output if expected_energy_output > 0 else 0
        )

        # Capacity factor
        theoretical_max = nameplate_capacity * 24 * 365  # kWh/year if running 24/7
        capacity_factor = actual_energy_output / theoretical_max if theoretical_max > 0 else 0

        # System efficiency
        system_efficiency = performance_ratio * 0.20  # Assuming 20% module efficiency baseline

        # Performance categories
        if performance_ratio >= 0.85:
            performance_category = "Excellent"
        elif performance_ratio >= 0.75:
            performance_category = "Good"
        elif performance_ratio >= 0.65:
            performance_category = "Average"
        else:
            performance_category = "Poor"

        result = {
            "performance_ratio": round(performance_ratio, 4),
            "performance_ratio_percent": round(performance_ratio * 100, 2),
            "capacity_factor": round(capacity_factor, 4),
            "capacity_factor_percent": round(capacity_factor * 100, 2),
            "system_efficiency_percent": round(system_efficiency * 100, 2),
            "performance_category": performance_category,
            "actual_energy_output_kwh": actual_energy_output,
            "expected_energy_output_kwh": expected_energy_output,
            "nameplate_capacity_kw": nameplate_capacity,
        }

        # Add irradiance analysis if data provided
        if irradiance_data:
            avg_irradiance = sum(irradiance_data) / len(irradiance_data)
            peak_irradiance = max(irradiance_data)
            result.update(
                {
                    "average_irradiance_w_per_m2": round(avg_irradiance, 2),
                    "peak_irradiance_w_per_m2": round(peak_irradiance, 2),
                    "irradiance_variability": round(
                        (peak_irradiance - min(irradiance_data)) / avg_irradiance, 2
                    ),
                }
            )

        return result

    @tool(name="degradation_analysis")
    def degradation_analysis(
        self,
        initial_capacity: float,
        current_capacity: float,
        years_in_operation: float,
        technology_type: str = "silicon",
    ) -> dict[str, Any]:
        """
        Analyze system degradation and predict future performance.

        Args:
            initial_capacity: Initial system capacity (kW)
            current_capacity: Current measured capacity (kW)
            years_in_operation: Years since installation
            technology_type: Technology type for benchmark comparison

        Returns:
            Degradation analysis results
        """
        # Calculate degradation
        total_degradation = (initial_capacity - current_capacity) / initial_capacity
        annual_degradation_rate = (
            total_degradation / years_in_operation if years_in_operation > 0 else 0
        )

        # Technology benchmarks (annual degradation rates)
        benchmarks = {
            "silicon": {"typical": 0.005, "good": 0.003, "excellent": 0.002},
            "thin_film": {"typical": 0.008, "good": 0.005, "excellent": 0.003},
            "perovskite": {"typical": 0.015, "good": 0.010, "excellent": 0.005},
            "bifacial": {"typical": 0.004, "good": 0.003, "excellent": 0.002},
        }

        benchmark = benchmarks.get(technology_type.lower(), benchmarks["silicon"])

        # Performance assessment
        if annual_degradation_rate <= benchmark["excellent"]:
            degradation_assessment = "Excellent - Below expected degradation"
        elif annual_degradation_rate <= benchmark["good"]:
            degradation_assessment = "Good - Within normal range"
        elif annual_degradation_rate <= benchmark["typical"]:
            degradation_assessment = "Average - Typical degradation"
        else:
            degradation_assessment = "Concerning - Above typical degradation"

        # Future projections
        remaining_life = 25 - years_in_operation  # Assuming 25-year design life
        projected_capacity_eol = current_capacity * (
            (1 - annual_degradation_rate) ** remaining_life
        )

        return {
            "annual_degradation_rate": round(annual_degradation_rate, 6),
            "annual_degradation_percent": round(annual_degradation_rate * 100, 3),
            "total_degradation_percent": round(total_degradation * 100, 2),
            "current_capacity_percent": round((current_capacity / initial_capacity) * 100, 2),
            "degradation_assessment": degradation_assessment,
            "technology_type": technology_type,
            "years_in_operation": years_in_operation,
            "benchmark_typical_percent": benchmark["typical"] * 100,
            "projected_capacity_end_of_life": round(projected_capacity_eol, 2),
            "projected_capacity_retention_eol": round(
                (projected_capacity_eol / initial_capacity) * 100, 2
            ),
            "remaining_useful_life_years": max(0, remaining_life),
        }


@skill(name="MarketAnalysis", description="Energy market analysis and forecasting tools")
class MarketAnalysis:
    """
    Collection of market analysis tools for energy trading and forecasting.

    Provides tools for:
    - Price trend analysis
    - Market volatility assessment
    - Demand forecasting
    - Competitive analysis
    """

    def __init__(self):
        self.regional_price_benchmarks = {
            "pjm": {"low": 30, "avg": 45, "high": 70},
            "caiso": {"low": 40, "avg": 60, "high": 90},
            "ercot": {"low": 25, "avg": 40, "high": 65},
            "neiso": {"low": 45, "avg": 65, "high": 95},
        }

    @tool(name="price_trend_analyzer")
    def price_trend_analyzer(
        self, historical_prices: list[float], time_periods: list[str], market_region: str = "pjm"
    ) -> dict[str, Any]:
        """
        Analyze energy price trends and volatility.

        Args:
            historical_prices: List of historical prices ($/MWh)
            time_periods: List of time period labels
            market_region: Market region for benchmark comparison

        Returns:
            Price trend analysis results
        """
        if len(historical_prices) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Basic statistics
        avg_price = sum(historical_prices) / len(historical_prices)
        min_price = min(historical_prices)
        max_price = max(historical_prices)
        price_range = max_price - min_price

        # Volatility calculation (standard deviation)
        variance = sum((p - avg_price) ** 2 for p in historical_prices) / len(historical_prices)
        volatility = variance**0.5
        coefficient_of_variation = volatility / avg_price if avg_price > 0 else 0

        # Trend calculation (linear regression slope)
        n = len(historical_prices)
        sum_x = sum(range(n))
        sum_y = sum(historical_prices)
        sum_xy = sum(i * price for i, price in enumerate(historical_prices))
        sum_x2 = sum(i * i for i in range(n))

        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            # Trend direction and strength
            if abs(slope) < 0.1:
                trend_direction = "Stable"
            elif slope > 0:
                trend_direction = "Increasing"
            else:
                trend_direction = "Decreasing"

            trend_strength = min(abs(slope) / avg_price * 100, 100)
        else:
            slope = 0
            intercept = avg_price
            trend_direction = "Stable"
            trend_strength = 0

        # Market comparison
        benchmarks = self.regional_price_benchmarks.get(
            market_region.lower(), self.regional_price_benchmarks["pjm"]
        )

        if avg_price <= benchmarks["low"]:
            market_position = "Below Market"
        elif avg_price <= benchmarks["avg"]:
            market_position = "Average Market"
        elif avg_price <= benchmarks["high"]:
            market_position = "Above Market"
        else:
            market_position = "Premium Market"

        # Forecast next period (simple linear projection)
        forecast_next = intercept + slope * n

        return {
            "price_statistics": {
                "average_price": round(avg_price, 2),
                "minimum_price": round(min_price, 2),
                "maximum_price": round(max_price, 2),
                "price_range": round(price_range, 2),
                "volatility": round(volatility, 2),
                "coefficient_of_variation": round(coefficient_of_variation, 3),
            },
            "trend_analysis": {
                "trend_direction": trend_direction,
                "trend_slope": round(slope, 4),
                "trend_strength_percent": round(trend_strength, 2),
                "forecast_next_period": round(forecast_next, 2),
            },
            "market_analysis": {
                "market_region": market_region,
                "market_position": market_position,
                "benchmark_low": benchmarks["low"],
                "benchmark_avg": benchmarks["avg"],
                "benchmark_high": benchmarks["high"],
            },
            "data_quality": {
                "data_points": n,
                "time_span": (
                    f"{time_periods[0]} to {time_periods[-1]}" if time_periods else "Unknown"
                ),
            },
        }

    @tool(name="demand_forecast")
    def demand_forecast(
        self,
        historical_demand: list[float],
        weather_factors: Optional[dict[str, float]] = None,
        economic_indicators: Optional[dict[str, float]] = None,
        forecast_periods: int = 12,
    ) -> dict[str, Any]:
        """
        Forecast energy demand based on historical data and external factors.

        Args:
            historical_demand: Historical demand data (MWh)
            weather_factors: Weather impact factors
            economic_indicators: Economic indicators (GDP growth, etc.)
            forecast_periods: Number of periods to forecast

        Returns:
            Demand forecast analysis
        """
        if len(historical_demand) < 3:
            return {"error": "Insufficient historical data for forecasting"}

        # Basic trend analysis
        n = len(historical_demand)

        # Simple moving average
        window_size = min(3, n)
        recent_avg = sum(historical_demand[-window_size:]) / window_size

        # Seasonal adjustment (simplified)
        if n >= 12:
            seasonal_pattern = []
            for i in range(12):
                seasonal_values = [historical_demand[j] for j in range(i, n, 12)]
                seasonal_avg = sum(seasonal_values) / len(seasonal_values)
                overall_avg = sum(historical_demand) / n
                seasonal_pattern.append(seasonal_avg / overall_avg)
        else:
            seasonal_pattern = [1.0] * 12  # No seasonal adjustment

        # Growth rate calculation
        if n >= 2:
            growth_rate = (historical_demand[-1] / historical_demand[0]) ** (1 / (n - 1)) - 1
        else:
            growth_rate = 0

        # Apply external factors
        weather_adjustment = 1.0
        if weather_factors:
            # Simplified weather impact (cooling/heating degree days)
            temperature_factor = weather_factors.get("temperature_deviation", 0)
            weather_adjustment = 1 + (temperature_factor * 0.02)  # 2% per degree deviation

        economic_adjustment = 1.0
        if economic_indicators:
            gdp_growth = economic_indicators.get("gdp_growth", 0)
            economic_adjustment = 1 + (gdp_growth * 0.5)  # 0.5x GDP growth impact

        # Generate forecast
        forecast = []
        base_value = recent_avg

        for period in range(forecast_periods):
            # Apply growth trend
            trend_value = base_value * ((1 + growth_rate) ** (period + 1))

            # Apply seasonal adjustment
            seasonal_index = period % 12
            seasonal_value = trend_value * seasonal_pattern[seasonal_index]

            # Apply external adjustments
            final_value = seasonal_value * weather_adjustment * economic_adjustment

            forecast.append(round(final_value, 2))

        # Confidence intervals (simplified)
        avg_historical = sum(historical_demand) / n
        volatility = (sum((d - avg_historical) ** 2 for d in historical_demand) / n) ** 0.5
        confidence_band = volatility * 1.96  # 95% confidence interval

        forecast_with_bands = [
            {
                "period": i + 1,
                "forecast": forecast[i],
                "lower_bound": round(max(0, forecast[i] - confidence_band), 2),
                "upper_bound": round(forecast[i] + confidence_band, 2),
            }
            for i in range(forecast_periods)
        ]

        return {
            "forecast_summary": {
                "historical_average": round(avg_historical, 2),
                "recent_average": round(recent_avg, 2),
                "growth_rate_percent": round(growth_rate * 100, 2),
                "forecast_periods": forecast_periods,
            },
            "adjustments_applied": {
                "weather_adjustment": round(weather_adjustment, 3),
                "economic_adjustment": round(economic_adjustment, 3),
                "seasonal_adjustment": "Applied" if n >= 12 else "Not Applied",
            },
            "forecast_data": forecast_with_bands,
            "model_quality": {
                "historical_volatility": round(volatility, 2),
                "confidence_interval_width": round(confidence_band * 2, 2),
                "data_points_used": n,
            },
        }


def demonstrate_skills():
    """Demonstrate all energy skills with comprehensive examples."""

    print("Energy Skills Demonstration")
    print("=" * 60)

    # Initialize SDK
    initialize_sdk(log_level="INFO")

    # Create skill instances
    economics = EnergyEconomics()
    performance = TechnicalPerformance()
    market = MarketAnalysis()

    # 1. Energy Economics Skill
    print("\n1. ENERGY ECONOMICS SKILL")
    print("-" * 40)

    # Sample project: 100MW Solar Farm
    initial_investment = 120_000_000  # $120M
    annual_cash_flows = [8_000_000] * 25  # $8M per year for 25 years

    print("NPV Analysis:")
    npv_result = economics.calculate_npv(initial_investment, annual_cash_flows, 0.08)
    for key, value in npv_result.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\nIRR Analysis:")
    irr_result = economics.calculate_irr(initial_investment, annual_cash_flows)
    for key, value in irr_result.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\nPayback Analysis:")
    payback_result = economics.payback_analysis(initial_investment, annual_cash_flows, 0.08)
    for key, value in payback_result.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # 2. Technical Performance Skill
    print("\n\n2. TECHNICAL PERFORMANCE SKILL")
    print("-" * 40)

    print("Performance Ratio Analysis:")
    pr_result = performance.performance_ratio_calculator(
        actual_energy_output=180_000_000,  # 180 GWh
        expected_energy_output=200_000_000,  # 200 GWh expected
        nameplate_capacity=100_000,  # 100 MW
        irradiance_data=[850, 920, 780, 1100, 950, 880, 1020, 950],
    )
    for key, value in pr_result.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\nDegradation Analysis:")
    degradation_result = performance.degradation_analysis(
        initial_capacity=100_000,  # 100 MW initial
        current_capacity=97_500,  # 97.5 MW current (after 5 years)
        years_in_operation=5,
        technology_type="silicon",
    )
    for key, value in degradation_result.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # 3. Market Analysis Skill
    print("\n\n3. MARKET ANALYSIS SKILL")
    print("-" * 40)

    # Sample price data
    historical_prices = [35, 42, 38, 45, 52, 48, 55, 50, 58, 62, 59, 65]
    time_periods = [f"2024-{i:02d}" for i in range(1, 13)]

    print("Price Trend Analysis:")
    price_result = market.price_trend_analyzer(
        historical_prices=historical_prices, time_periods=time_periods, market_region="pjm"
    )

    print("  Price Statistics:")
    for key, value in price_result["price_statistics"].items():
        print(f"    {key.replace('_', ' ').title()}: {value}")

    print("  Trend Analysis:")
    for key, value in price_result["trend_analysis"].items():
        print(f"    {key.replace('_', ' ').title()}: {value}")

    # Sample demand data
    historical_demand = [1200, 1150, 1300, 1250, 1400, 1350, 1500, 1450, 1380, 1420, 1380, 1450]

    print("\nDemand Forecast:")
    demand_result = market.demand_forecast(
        historical_demand=historical_demand,
        weather_factors={"temperature_deviation": 2.5},
        economic_indicators={"gdp_growth": 0.03},
        forecast_periods=6,
    )

    print("  Forecast Summary:")
    for key, value in demand_result["forecast_summary"].items():
        print(f"    {key.replace('_', ' ').title()}: {value}")

    print("  Next 3 Periods Forecast:")
    for _i, period_data in enumerate(demand_result["forecast_data"][:3]):
        print(
            f"    Period {period_data['period']}: {period_data['forecast']} MWh "
            f"({period_data['lower_bound']}-{period_data['upper_bound']})"
        )

    print("\n" + "=" * 60)
    print("Skills demonstration completed!")
    print("These skills can be used individually or combined in agents.")


if __name__ == "__main__":
    demonstrate_skills()
