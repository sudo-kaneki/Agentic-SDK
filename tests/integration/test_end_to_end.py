# tests/integration/test_end_to_end.py
"""
End-to-end integration tests for the EnergyAI SDK.
Tests complete workflows using decorator-based agents.
"""

from unittest.mock import Mock, patch

import pytest

from energyai_sdk import agent, agent_registry, initialize_sdk, master_agent, tool
from energyai_sdk.agents import bootstrap_agents
from energyai_sdk.application import create_application
from energyai_sdk.config import ConfigurationManager
from energyai_sdk.middleware import create_default_pipeline


@pytest.mark.integration
class TestCompleteSDKWorkflow:
    """Test complete SDK workflows using decorator-based agents."""

    def setup_method(self):
        """Setup for each test."""
        agent_registry.clear()
        initialize_sdk(log_level="DEBUG")

    def test_decorator_based_workflow(self):
        """Test complete workflow using only decorators."""

        # 1. Define tools using decorators
        @tool(
            name="wind_power_calculator",
            description="Calculate wind power output based on wind speed and turbine specs",
        )
        def wind_power_calculator(wind_speed: float, turbine_capacity: float = 2.5) -> dict:
            """Calculate power output from wind speed using simplified power curve."""
            if wind_speed < 3:  # Cut-in wind speed
                power_output = 0
            elif wind_speed > 25:  # Cut-out wind speed
                power_output = 0
            elif wind_speed > 15:  # Rated wind speed
                power_output = turbine_capacity
            else:
                # Simplified power curve: P = 0.5 * ρ * A * v³ * Cp (normalized)
                power_output = min(turbine_capacity, (wind_speed**3) / (15**3) * turbine_capacity)

            return {
                "power_output_mw": round(power_output, 3),
                "wind_speed_ms": wind_speed,
                "turbine_capacity_mw": turbine_capacity,
                "capacity_factor": power_output / turbine_capacity if turbine_capacity > 0 else 0,
            }

        # 2. Create agent with tools using decorator
        @agent(
            name="WindAnalyst",
            description="Wind energy analysis specialist",
            system_prompt="""You are a wind energy specialist. You can calculate wind power output
                based on wind speeds and turbine specifications. Use the available tools
            to provide accurate calculations and insights.""",
            tools=["wind_power_calculator"],
        )
        class WindAnalyst:
            temperature = 0.3
            max_tokens = 1000

        # 3. Verify registrations
        assert "WindAnalyst" in agent_registry.list_agents()
        assert "wind_power_calculator" in agent_registry.list_tools()

        # 4. Test tool functionality directly
        result = wind_power_calculator(10.0, 2.5)
        assert result["wind_speed_ms"] == 10.0
        assert result["turbine_capacity_mw"] == 2.5
        assert result["power_output_mw"] > 0

    def test_multi_agent_energy_platform(self):
        """Test multi-agent energy platform using decorators."""

        # Define energy analysis tools
        @tool(name="lcoe_calculator", description="Calculate LCOE")
        def lcoe_calculator(capex: float, opex: float, generation: float) -> dict:
            lcoe = (capex + opex) / generation if generation > 0 else 0
            return {
                "lcoe_per_mwh": lcoe,
                "capex": capex,
                "opex": opex,
                "generation_mwh": generation,
            }

        @tool(name="carbon_calculator", description="Calculate carbon emissions")
        def carbon_calculator(generation_mwh: float, emission_factor: float = 0.5) -> dict:
            emissions = generation_mwh * emission_factor
            return {
                "emissions_tons_co2": emissions,
                "generation_mwh": generation_mwh,
                "factor": emission_factor,
            }

        # Define specialized agents
        @agent(
            name="SolarAnalyst",
            description="Solar energy analysis specialist",
            system_prompt="You analyze solar energy projects and provide technical insights.",
            tools=["lcoe_calculator"],
        )
        class SolarAnalyst:
            temperature = 0.3

        @agent(
            name="FinancialAnalyst",
            description="Financial analysis specialist for energy projects",
            system_prompt="You analyze financial metrics for energy projects.",
            tools=["lcoe_calculator"],
        )
        class FinancialAnalyst:
            temperature = 0.2

        # Master agent to coordinate
        @master_agent(
            name="EnergyPlatformManager",
            description="Master coordinator for energy analysis platform",
            system_prompt="You coordinate analysis from multiple energy specialists.",
            subordinates=["SolarAnalyst", "FinancialAnalyst"],
        )
        class EnergyPlatformManager:
            temperature = 0.4
            max_iterations = 3

        # Verify all agents are registered
        assert "SolarAnalyst" in agent_registry.list_agents()
        assert "FinancialAnalyst" in agent_registry.list_agents()
        assert "EnergyPlatformManager" in agent_registry.list_agents()
        assert "lcoe_calculator" in agent_registry.list_tools()
        assert "carbon_calculator" in agent_registry.list_tools()

    @pytest.mark.asyncio
    async def test_bootstrap_and_process_requests(self):
        """Test bootstrap agents and process requests."""

        # Define test agent with decorator
        @tool(name="simple_calc", description="Simple calculation")
        def simple_calc(x: float, y: float) -> dict:
            return {"sum": x + y, "product": x * y}

        @agent(
            name="TestAgent",
            description="Agent for request processing test",
            system_prompt="You can perform simple calculations.",
            tools=["simple_calc"],
        )
        class TestAgent:
            temperature = 0.5

        # Mock configuration
        mock_config = {
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com/",
            "api_key": "test-key-123",
            "service_type": "azure_openai",
        }

        # Mock Semantic Kernel components
        with patch("energyai_sdk.agents.SEMANTIC_KERNEL_AVAILABLE", True):
            with (
                patch("energyai_sdk.agents.Kernel") as mock_kernel,
                patch("energyai_sdk.agents.AzureAIInferenceChatCompletion") as mock_service,
                patch("energyai_sdk.agents.ChatHistory") as mock_history,
            ):

                # Setup mocks
                mock_kernel_instance = Mock()
                mock_kernel.return_value = mock_kernel_instance

                mock_service_instance = Mock()
                mock_service.return_value = mock_service_instance

                mock_history_instance = Mock()
                mock_history.return_value = mock_history_instance

                # Bootstrap agents
                agents = bootstrap_agents(azure_openai_config=mock_config)

                # Verify agent was created
                assert "TestAgent" in agents
                test_agent = agents["TestAgent"]
                assert test_agent is not None
                assert hasattr(test_agent, "agent_name")
                assert test_agent.agent_name == "TestAgent"

    def test_application_integration(self):
        """Test integration with FastAPI application."""

        # Define a simple energy agent
        @tool(name="efficiency_calc", description="Calculate efficiency")
        def efficiency_calc(output: float, input: float) -> dict:
            efficiency = (output / input * 100) if input > 0 else 0
            return {"efficiency_percent": efficiency, "output": output, "input": input}

        @agent(
            name="EfficiencyAgent",
            description="Energy efficiency specialist",
            system_prompt="You calculate and analyze energy efficiency metrics.",
            tools=["efficiency_calc"],
        )
        class EfficiencyAgent:
            temperature = 0.3

        # Verify agent registration
        assert "EfficiencyAgent" in agent_registry.list_agents()

        # Test application creation (without actually starting server)
        try:
            app = create_application(title="Test Energy Platform", description="Test platform")
            assert app is not None
        except ImportError:
            # FastAPI not available, skip this test
            pytest.skip("FastAPI not available for application testing")

    def test_middleware_integration(self):
        """Test middleware integration with decorator-based agents."""

        @tool(name="power_calc", description="Calculate power")
        def power_calc(voltage: float, current: float) -> dict:
            power = voltage * current
            return {"power_watts": power, "voltage": voltage, "current": current}

        @agent(
            name="PowerAgent",
            description="Electrical power specialist",
            system_prompt="You calculate electrical power metrics.",
            tools=["power_calc"],
        )
        class PowerAgent:
            temperature = 0.3

        # Verify agent registration
        assert "PowerAgent" in agent_registry.list_agents()

        # Test middleware pipeline creation
        try:
            pipeline = create_default_pipeline()
            assert pipeline is not None
        except ImportError:
            # Middleware dependencies not available
            pytest.skip("Middleware dependencies not available")

    def test_configuration_management(self):
        """Test configuration management with decorator agents."""

        @tool(name="storage_calc", description="Calculate energy storage")
        def storage_calc(capacity_kwh: float, efficiency: float = 0.95) -> dict:
            usable_capacity = capacity_kwh * efficiency
            return {
                "usable_capacity_kwh": usable_capacity,
                "total_capacity_kwh": capacity_kwh,
                "efficiency": efficiency,
            }

        @agent(
            name="StorageAgent",
            description="Energy storage specialist",
            system_prompt="You analyze energy storage systems and their performance.",
            tools=["storage_calc"],
        )
        class StorageAgent:
            temperature = 0.3

        # Test basic configuration
        config_manager = ConfigurationManager()
        assert config_manager is not None

        # Verify agent exists
        assert "StorageAgent" in agent_registry.list_agents()

    def test_error_handling_and_resilience(self):
        """Test error handling with decorator-based agents."""

        @tool(name="failing_calc", description="Calculator that sometimes fails")
        def failing_calc(value: float, should_fail: bool = False) -> dict:
            if should_fail:
                raise ValueError("Intentional failure for testing")
            return {"result": value * 2, "status": "success"}

        @agent(
            name="ResilientAgent",
            description="Agent that handles errors gracefully",
            system_prompt="You handle calculations even when tools might fail.",
            tools=["failing_calc"],
        )
        class ResilientAgent:
            temperature = 0.4

        # Test successful tool execution
        result = failing_calc(5.0, False)
        assert result["result"] == 10.0
        assert result["status"] == "success"

        # Test error handling
        with pytest.raises(ValueError, match="Intentional failure"):
            failing_calc(5.0, True)

        # Verify agent registration
        assert "ResilientAgent" in agent_registry.list_agents()

    def teardown_method(self):
        """Cleanup after each test."""
        agent_registry.clear()


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world energy analysis scenarios."""

    def setup_method(self):
        """Setup for scenario tests."""
        agent_registry.clear()
        initialize_sdk(log_level="INFO")

    def test_solar_project_analysis(self):
        """Test complete solar project analysis workflow."""

        # Define solar-specific tools
        @tool(name="solar_generation", description="Calculate solar generation")
        def solar_generation(capacity_mw: float, solar_hours: float, pr: float = 0.8) -> dict:
            generation = capacity_mw * solar_hours * pr * 365
            return {
                "annual_generation_mwh": generation,
                "capacity_mw": capacity_mw,
                "solar_hours": solar_hours,
                "pr": pr,
            }

        @tool(name="financial_analysis", description="Analyze project finances")
        def financial_analysis(
            capex: float, generation_mwh: float, electricity_price: float = 50
        ) -> dict:
            revenue = generation_mwh * electricity_price
            payback = capex / revenue if revenue > 0 else float("inf")
            return {
                "annual_revenue": revenue,
                "simple_payback_years": payback,
                "electricity_price": electricity_price,
            }

        # Define solar analyst
        @agent(
            name="SolarProjectAnalyst",
            description="Complete solar project analysis specialist",
            system_prompt="""You are a solar project analyst. You can calculate generation potential
            and financial metrics for solar projects. Provide comprehensive analysis.""",
            tools=["solar_generation", "financial_analysis"],
        )
        class SolarProjectAnalyst:
            temperature = 0.2
            max_tokens = 1500

        # Test the analysis tools
        gen_result = solar_generation(100, 5.5)  # 100 MW, 5.5 hours avg
        assert gen_result["annual_generation_mwh"] > 0

        fin_result = financial_analysis(150_000_000, gen_result["annual_generation_mwh"])
        assert fin_result["annual_revenue"] > 0
        assert fin_result["simple_payback_years"] > 0

        # Verify agent registration
        assert "SolarProjectAnalyst" in agent_registry.list_agents()

    def test_wind_farm_optimization(self):
        """Test wind farm optimization scenario."""

        @tool(name="wind_assessment", description="Assess wind resource")
        def wind_assessment(avg_wind_speed: float, weibull_k: float = 2.0) -> dict:
            # Simplified wind assessment
            if avg_wind_speed < 6:
                class_rating = "Poor"
            elif avg_wind_speed < 8:
                class_rating = "Marginal"
            elif avg_wind_speed < 10:
                class_rating = "Good"
            else:
                class_rating = "Excellent"

            return {
                "avg_wind_speed": avg_wind_speed,
                "wind_class": class_rating,
                "weibull_k": weibull_k,
            }

        @tool(name="turbine_selection", description="Select optimal turbine")
        def turbine_selection(wind_class: str, site_constraints: str = "none") -> dict:
            turbine_map = {
                "Poor": {"model": "Low wind turbine", "capacity": 2.0, "hub_height": 120},
                "Marginal": {"model": "Medium wind turbine", "capacity": 2.5, "hub_height": 100},
                "Good": {"model": "Standard turbine", "capacity": 3.0, "hub_height": 90},
                "Excellent": {"model": "High wind turbine", "capacity": 3.5, "hub_height": 80},
            }

            selected = turbine_map.get(wind_class, turbine_map["Good"])
            selected["site_constraints"] = site_constraints
            return selected

        @agent(
            name="WindFarmOptimizer",
            description="Wind farm optimization specialist",
            system_prompt="""You optimize wind farm designs based on wind resources and site constraints.
            Use wind assessment and turbine selection tools to provide optimal recommendations.""",
            tools=["wind_assessment", "turbine_selection"],
        )
        class WindFarmOptimizer:
            temperature = 0.3

        # Test optimization workflow
        wind_result = wind_assessment(8.5)
        assert wind_result["wind_class"] == "Good"

        turbine_result = turbine_selection(wind_result["wind_class"])
        assert turbine_result["capacity"] == 3.0

        # Verify agent registration
        assert "WindFarmOptimizer" in agent_registry.list_agents()

    def teardown_method(self):
        """Cleanup after scenario tests."""
        agent_registry.clear()


if __name__ == "__main__":
    pytest.main([__file__])
