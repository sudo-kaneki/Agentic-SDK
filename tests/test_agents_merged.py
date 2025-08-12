# tests/test_agents.py
"""
Comprehensive test suite for agent implementations using decorator-based approaches.
"""

from unittest.mock import patch

import pytest

from energyai_sdk import agent_registry
from energyai_sdk.agents import bootstrap_agents
from energyai_sdk.core import CoreAgent
from energyai_sdk.decorators import agent, master_agent, skill, tool

# Try to import optional agent classes if available
try:
    from energyai_sdk.agents import SimpleSemanticKernelAgent
except ImportError:
    SimpleSemanticKernelAgent = None


class TestCoreAgent:
    """Test CoreAgent abstract base class."""

    def test_core_agent_is_abstract(self):
        """Test that CoreAgent cannot be instantiated directly."""

        with pytest.raises(TypeError):
            CoreAgent(agent_name="test", agent_description="test", system_prompt="test")

    def test_core_agent_subclass_requirements(self):
        """Test that CoreAgent subclasses must implement required methods."""

        class IncompleteAgent(CoreAgent):
            pass

        with pytest.raises(TypeError):
            IncompleteAgent(agent_name="test", agent_description="test", system_prompt="test")


class TestDecoratorBasedAgents:
    """Test decorator-based agent creation and functionality."""

    def setup_method(self):
        """Reset registry before each test."""
        agent_registry.clear()

    def test_decorator_basic_configuration(self):
        """Test basic decorator configuration."""

        @agent(
            name="DecoratorTestAgent",
            description="Agent created with decorator",
            system_prompt="You are a decorator test agent.",
        )
        class DecoratorTestAgent:
            temperature = 0.5
            max_tokens = 1500

        # Check agent was registered
        assert "DecoratorTestAgent" in agent_registry.list_agents()

    def test_decorator_with_tools(self):
        """Test decorator with tool configuration."""

        @tool(name="test_calc", description="Test calculation")
        def test_calc(x: float) -> dict:
            return {"result": x * 2}

        @agent(
            name="ToolTestAgent",
            description="Agent with tools",
            system_prompt="I have tools.",
            tools=["test_calc"],
        )
        class ToolTestAgent:
            temperature = 0.3

        # Verify agent and tool were registered
        assert "ToolTestAgent" in agent_registry.list_agents()
        assert "test_calc" in agent_registry.list_tools()

    def test_decorator_with_skills(self):
        """Test decorator with skill configuration."""

        @skill(name="TestMath", description="Math operations")
        class TestMath:
            @tool(name="test_calculator")
            def calculate(self, x: float, y: float) -> dict:
                return {"sum": x + y}

        @agent(
            name="SkillTestAgent",
            description="Agent with skills",
            system_prompt="I have math skills.",
        )
        class SkillTestAgent:
            temperature = 0.2

        # Verify registrations
        assert "SkillTestAgent" in agent_registry.list_agents()
        assert "TestMath" in agent_registry.list_skills()
        assert "test_calculator" in agent_registry.list_tools()

    def test_master_agent_decorator(self):
        """Test master agent decorator functionality."""

        @agent(name="SubAgent1", description="Sub 1", system_prompt="I am sub 1")
        class SubAgent1:
            pass

        @agent(name="SubAgent2", description="Sub 2", system_prompt="I am sub 2")
        class SubAgent2:
            pass

        @master_agent(
            name="TestMaster",
            description="Test master agent",
            system_prompt="I coordinate other agents.",
            subordinates=["SubAgent1", "SubAgent2"],
        )
        class TestMaster:
            temperature = 0.4
            max_iterations = 3

        # Verify all agents were registered
        assert "TestMaster" in agent_registry.list_agents()
        assert "SubAgent1" in agent_registry.list_agents()
        assert "SubAgent2" in agent_registry.list_agents()

    def test_agent_configuration_attributes(self):
        """Test that agent class attributes are properly captured."""

        @agent(
            name="ConfiguredAgent",
            description="Agent with custom configuration",
            system_prompt="Custom prompt",
        )
        class ConfiguredAgent:
            temperature = 0.1
            max_tokens = 500
            custom_attribute = "test_value"

        # Check that decorator captured the configuration
        from energyai_sdk.decorators import get_pending_agents

        pending = get_pending_agents()

        assert "ConfiguredAgent" in pending
        agent_class = pending["ConfiguredAgent"]
        assert hasattr(agent_class, "temperature")
        assert agent_class.temperature == 0.1
        assert agent_class.max_tokens == 500
        assert agent_class.custom_attribute == "test_value"

    def teardown_method(self):
        """Clean up after each test."""
        agent_registry.clear()


class TestBootstrapIntegration:
    """Test bootstrap_agents functionality with decorator-defined agents."""

    def setup_method(self):
        """Setup for bootstrap tests."""
        agent_registry.clear()

    def test_bootstrap_functionality(self):
        """Test that bootstrap_agents works with decorator-defined agents."""

        @tool(name="simple_tool", description="Simple test tool")
        def simple_tool(x: float) -> dict:
            return {"result": x}

        @agent(
            name="SimpleTestAgent",
            description="Simple test agent",
            system_prompt="I am a simple test agent.",
            tools=["simple_tool"],
        )
        class SimpleTestAgent:
            temperature = 0.5

        # Verify the agent was registered through decorators
        assert "SimpleTestAgent" in agent_registry.list_agents()
        assert "simple_tool" in agent_registry.list_tools()

    @pytest.mark.asyncio
    async def test_bootstrap_with_mock_sk(self):
        """Test bootstrap_agents with mocked Semantic Kernel."""

        @tool(name="mock_tool", description="Mock tool")
        def mock_tool(value: str) -> dict:
            return {"processed": value}

        @agent(
            name="MockBootstrapAgent",
            description="Agent for mock bootstrap testing",
            system_prompt="I am a mock agent.",
            tools=["mock_tool"],
        )
        class MockBootstrapAgent:
            temperature = 0.7

        # Mock configuration
        mock_config = {
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com/",
            "api_key": "test-key",
            "service_type": "azure_openai",
        }

        # Mock Semantic Kernel components
        with patch("energyai_sdk.agents.SEMANTIC_KERNEL_AVAILABLE", True):
            with (
                patch("energyai_sdk.agents.Kernel"),
                patch("energyai_sdk.agents.AzureAIInferenceChatCompletion"),
                patch("energyai_sdk.agents.ChatHistory"),
            ):

                agents = bootstrap_agents(azure_openai_config=mock_config)

                # Verify agent was created
                assert "MockBootstrapAgent" in agents
                if agents["MockBootstrapAgent"]:
                    assert hasattr(agents["MockBootstrapAgent"], "agent_name")

    def teardown_method(self):
        """Clean up after tests."""
        agent_registry.clear()


class TestEnergySpecificAgents:
    """Test energy domain-specific agent patterns."""

    def setup_method(self):
        """Reset registry before each test."""
        agent_registry.clear()

    def test_financial_analyst_pattern(self):
        """Test financial analyst agent pattern."""

        @tool(name="npv_calculator", description="NPV calculation")
        def npv_calculator(cash_flows: list[float], discount_rate: float) -> dict:
            npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows, 1))
            return {"npv": npv}

        @agent(
            name="FinancialAnalyst",
            description="Expert in energy project finance",
            system_prompt="""You are a financial analyst specializing in energy projects.
            Use your tools to perform accurate financial calculations.""",
            tools=["npv_calculator"],
        )
        class FinancialAnalyst:
            temperature = 0.3  # Lower temperature for consistent financial analysis
            max_tokens = 1000

        assert "FinancialAnalyst" in agent_registry.list_agents()
        assert "npv_calculator" in agent_registry.list_tools()

    def test_technical_analyst_pattern(self):
        """Test technical analyst agent pattern."""

        @tool(name="capacity_factor", description="Calculate capacity factor")
        def capacity_factor(actual_output: float, theoretical_max: float) -> dict:
            cf = actual_output / theoretical_max
            return {"capacity_factor": cf, "percentage": cf * 100}

        @agent(
            name="TechnicalAnalyst",
            description="Renewable energy technical expert",
            system_prompt="""You are a technical expert in renewable energy systems.
            Analyze performance data and provide technical insights.""",
            tools=["capacity_factor"],
        )
        class TechnicalAnalyst:
            temperature = 0.4
            max_tokens = 1200

        assert "TechnicalAnalyst" in agent_registry.list_agents()
        assert "capacity_factor" in agent_registry.list_tools()

    def test_energy_platform_pattern(self):
        """Test complete energy platform with multiple agents."""

        # Define tools
        @tool(name="lcoe_calc", description="LCOE calculation")
        def lcoe_calc(capex: float, opex: float, generation: float) -> dict:
            return {"lcoe": (capex + opex) / generation}

        @tool(name="carbon_calc", description="Carbon emissions calculation")
        def carbon_calc(energy_mwh: float, emission_factor: float) -> dict:
            return {"emissions_tons": energy_mwh * emission_factor / 1000}

        # Define specialized agents
        @agent(
            name="EconomicsSpecialist",
            description="Energy economics specialist",
            system_prompt="You analyze energy economics and financial metrics.",
            tools=["lcoe_calc"],
        )
        class EconomicsSpecialist:
            temperature = 0.3

        @agent(
            name="SustainabilitySpecialist",
            description="Environmental impact specialist",
            system_prompt="You analyze environmental and sustainability metrics.",
            tools=["carbon_calc"],
        )
        class SustainabilitySpecialist:
            temperature = 0.4

        # Master coordinator
        @master_agent(
            name="EnergyPlatformMaster",
            description="Master coordinator for energy analysis platform",
            system_prompt="You coordinate comprehensive energy project analysis.",
            subordinates=["EconomicsSpecialist", "SustainabilitySpecialist"],
        )
        class EnergyPlatformMaster:
            temperature = 0.5
            max_iterations = 3

        # Verify all components are registered
        assert "EconomicsSpecialist" in agent_registry.list_agents()
        assert "SustainabilitySpecialist" in agent_registry.list_agents()
        assert "EnergyPlatformMaster" in agent_registry.list_agents()
        assert "lcoe_calc" in agent_registry.list_tools()
        assert "carbon_calc" in agent_registry.list_tools()

    def teardown_method(self):
        """Clean up after each test."""
        agent_registry.clear()


if __name__ == "__main__":
    pytest.main([__file__])
