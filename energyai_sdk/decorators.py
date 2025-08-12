# Decorators for agents, tools, and skills

import inspect
import logging
from typing import Any, Callable, Optional, Union

from .core import PlannerDefinition, PromptTemplate, SkillDefinition, ToolDefinition, agent_registry

# Tool decorator


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Decorator to register a function as a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        parameters: Parameter definitions for the tool
        metadata: Additional metadata for the tool

    Example:
        @tool(name="calculate_lcoe", description="Calculate Levelized Cost of Energy")
        def calculate_lcoe(capital_cost: float, annual_generation: float) -> dict:
            return {"lcoe": capital_cost / annual_generation}
    """

    def decorator(func: Callable) -> Callable:
        # Extract tool information
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"
        tool_parameters = parameters or _extract_function_parameters(func)
        tool_metadata = metadata or {}

        # Create tool definition
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            function=func,
            parameters=tool_parameters,
            metadata={
                **tool_metadata,
                "function_name": func.__name__,
                "module": func.__module__,
                "registered_via": "decorator",
            },
        )

        # Register the tool
        agent_registry.register_tool(tool_name, tool_def)

        # Add tool definition to function for later reference
        func._tool_definition = tool_def
        func._is_energyai_tool = True

        logger = logging.getLogger("energyai_sdk.decorators")
        logger.debug(f"Registered tool: {tool_name}")

        return func

    return decorator


# ==============================================================================
# SKILL DECORATOR
# ==============================================================================


def skill(
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Decorator to register a class as a skill containing multiple tools.

    Args:
        name: Skill name (defaults to class name)
        description: Skill description (defaults to class docstring)
        metadata: Additional metadata for the skill

    Example:
        @skill(name="EnergyCalculations", description="Energy analysis tools")
        class EnergyCalculations:
            @tool(name="lcoe")
            def calculate_lcoe(self, cost, generation):
                return cost / generation
    """

    def decorator(cls: type) -> type:
        # Extract skill information
        skill_name = name or cls.__name__
        skill_description = description or cls.__doc__ or f"Skill: {skill_name}"
        skill_metadata = metadata or {}

        # Find all tool methods in the class
        tool_functions = []
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, "_is_energyai_tool"):
                tool_functions.append(attr)

        # Create skill definition
        skill_def = SkillDefinition(
            name=skill_name,
            description=skill_description,
            functions=tool_functions,
            metadata={
                **skill_metadata,
                "class_name": cls.__name__,
                "module": cls.__module__,
                "tool_count": len(tool_functions),
                "registered_via": "decorator",
            },
        )

        # Register the skill
        agent_registry.register_skill(skill_name, skill_def)

        # Add skill definition to class
        cls._skill_definition = skill_def
        cls._is_energyai_skill = True

        logger = logging.getLogger("energyai_sdk.decorators")
        logger.debug(f"Registered skill: {skill_name} with {len(tool_functions)} tools")

        return cls

    return decorator


# ==============================================================================
# PROMPT DECORATOR
# ==============================================================================


def prompt(
    name: Optional[str] = None,
    template: Optional[str] = None,
    variables: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Decorator to register a function as a prompt template.

    Args:
        name: Prompt name (defaults to function name)
        template: Prompt template string (defaults to function docstring)
        variables: List of template variables
        metadata: Additional metadata for the prompt

    Example:
        @prompt(name="energy_analysis", variables=["project_type", "capacity"])
        def energy_analysis_prompt():
            '''
            Analyze the {project_type} project with {capacity} MW capacity.
            Consider technical, financial, and environmental aspects.
            '''
            pass
    """

    def decorator(func: Callable) -> Callable:
        # Extract prompt information
        prompt_name = name or func.__name__
        prompt_template = template or func.__doc__ or f"Template for {prompt_name}"
        prompt_variables = variables or _extract_template_variables(prompt_template)
        prompt_metadata = metadata or {}

        # Create prompt definition
        prompt_def = PromptTemplate(
            name=prompt_name,
            template=prompt_template,
            variables=prompt_variables,
            metadata={
                **prompt_metadata,
                "function_name": func.__name__,
                "module": func.__module__,
                "variable_count": len(prompt_variables),
                "registered_via": "decorator",
            },
        )

        # Register the prompt
        agent_registry.register_prompt(prompt_name, prompt_def)

        # Add prompt definition to function
        func._prompt_definition = prompt_def
        func._is_energyai_prompt = True

        logger = logging.getLogger("energyai_sdk.decorators")
        logger.debug(f"Registered prompt: {prompt_name}")

        return func

    return decorator


# ==============================================================================
# PLANNER DECORATOR
# ==============================================================================


def planner(
    name: Optional[str] = None,
    planner_type: str = "sequential",
    description: Optional[str] = None,
    configuration: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Decorator to register a class or function as a planner.

    Args:
        name: Planner name
        planner_type: Type of planner (sequential, parallel, conditional)
        description: Planner description
        configuration: Planner configuration
        metadata: Additional metadata

    Example:
        @planner(name="energy_project_planner", planner_type="sequential")
        class EnergyProjectPlanner:
            def plan(self, request):
                return ["analyze_feasibility", "calculate_economics", "assess_risk"]
    """

    def decorator(obj: Union[type, Callable]) -> Union[type, Callable]:
        # Extract planner information
        planner_name = name or getattr(obj, "__name__", "unknown_planner")
        planner_description = description or getattr(obj, "__doc__", f"Planner: {planner_name}")
        planner_config = configuration or {}
        planner_metadata = metadata or {}

        # Create planner definition
        planner_def = PlannerDefinition(
            name=planner_name,
            description=planner_description,
            planner_type=planner_type,
            configuration=planner_config,
            metadata={
                **planner_metadata,
                "object_name": getattr(obj, "__name__", "unknown"),
                "object_type": "class" if inspect.isclass(obj) else "function",
                "module": getattr(obj, "__module__", "unknown"),
                "registered_via": "decorator",
            },
        )

        # Register the planner
        agent_registry.register_planner(planner_name, planner_def)

        # Add planner definition to object
        obj._planner_definition = planner_def
        obj._is_energyai_planner = True

        logger = logging.getLogger("energyai_sdk.decorators")
        logger.debug(f"Registered planner: {planner_name} ({planner_type})")

        return obj

    return decorator


# ==============================================================================
# AGENT DECORATORS - ALL AGENT-RELATED DECORATORS IN ONE PLACE
# ==============================================================================

# Registry for pending agent classes
_pending_agents: dict[str, type] = {}


def agent(
    name: Optional[str] = None,
    description: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[list[str]] = None,
    skills: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Enhanced agent decorator that registers agents for automatic creation.

    This is THE main decorator for your SDK vision - simple agent creation!

    Args:
        name: Agent name (defaults to class name)
        description: Agent description
        system_prompt: System prompt for the agent
        tools: List of tool names to include
        skills: List of skill names to include
        metadata: Additional metadata

    Example:
        @agent(name="EnergyAnalyst", tools=["calculate_lcoe"])
        class EnergyAnalyst:
            system_prompt = "You are an expert energy analyst..."
            temperature = 0.3
    """

    def decorator(cls: type) -> type:
        agent_name = name or cls.__name__
        agent_description = description or cls.__doc__ or f"Agent: {agent_name}"
        agent_system_prompt = system_prompt or getattr(
            cls, "system_prompt", f"You are {agent_name}, a helpful AI assistant."
        )

        # Store configuration
        cls._agent_config = {
            "name": agent_name,
            "description": agent_description,
            "system_prompt": agent_system_prompt,
            "tools": tools or [],
            "skills": skills or [],
            "metadata": {
                **(metadata or {}),
                "class_name": cls.__name__,
                "module": cls.__module__,
                "registered_via": "decorator",
            },
        }

        cls._is_energyai_agent = True

        # Register for automatic creation
        _pending_agents[agent_name] = cls

        logger = logging.getLogger("energyai_sdk.decorators")
        logger.info(f"Registered agent class: {agent_name}")

        return cls

    return decorator


def master_agent(
    name: Optional[str] = None,
    description: Optional[str] = None,
    system_prompt: Optional[str] = None,
    subordinates: Optional[list[str]] = None,
    selection_strategy: str = "prompt",
    max_iterations: int = 5,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Decorator for creating master/coordinator agents.

    Args:
        name: Master agent name
        description: Master agent description
        system_prompt: System prompt for coordination
        subordinates: List of subordinate agent names
        selection_strategy: How to select subordinate agents
        max_iterations: Maximum orchestration iterations
        metadata: Additional metadata

    Example:
        @master_agent(
            name="EnergyMaster",
            subordinates=["EnergyAnalyst", "TechnicalAnalyst"],
            selection_strategy="prompt"
        )
        class EnergyMasterAgent:
            system_prompt = "You coordinate energy analysis tasks..."
            temperature = 0.5
    """

    def decorator(cls: type) -> type:
        # Apply base agent decorator first
        cls = agent(name, description, system_prompt, metadata=metadata)(cls)

        # Add master-specific configuration
        cls._agent_config.update(
            {
                "agent_type": "master",
                "subordinates": subordinates or [],
                "selection_strategy": selection_strategy,
                "max_iterations": max_iterations,
            }
        )

        return cls

    return decorator


def get_registered_agent_classes() -> list[str]:
    """Get list of registered agent classes."""
    return list(_pending_agents.keys())


def get_pending_agents() -> dict[str, type]:
    """Get pending agent classes for creation."""
    return _pending_agents.copy()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def _extract_function_parameters(func: Callable) -> dict[str, Any]:
    """Extract parameter information from a function signature."""
    try:
        sig = inspect.signature(func)
        parameters = {}

        for param_name, param in sig.parameters.items():
            param_info = {
                "name": param_name,
                "required": param.default == inspect.Parameter.empty,
                "type": _get_type_string(param.annotation),
            }

            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        return parameters

    except Exception as e:
        logger = logging.getLogger("energyai_sdk.decorators")
        logger.warning(f"Failed to extract parameters from {func.__name__}: {e}")
        return {}


def _get_type_string(type_annotation) -> str:
    """Convert type annotation to string."""
    if type_annotation == inspect.Parameter.empty:
        return "any"

    if hasattr(type_annotation, "__name__"):
        return type_annotation.__name__

    return str(type_annotation)


def _extract_template_variables(template: str) -> list[str]:
    """Extract variables from a template string."""
    import re

    # Look for {variable_name} patterns
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, template)

    # Remove duplicates and return
    return list(set(matches))


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def get_registered_tools() -> dict[str, ToolDefinition]:
    """Get all registered tools."""
    return agent_registry.tools.copy()


def get_registered_skills() -> dict[str, SkillDefinition]:
    """Get all registered skills."""
    return agent_registry.skills.copy()


def get_registered_prompts() -> dict[str, PromptTemplate]:
    """Get all registered prompts."""
    return agent_registry.prompts.copy()


def get_registered_planners() -> dict[str, PlannerDefinition]:
    """Get all registered planners."""
    return agent_registry.planners.copy()


def is_tool(func: Callable) -> bool:
    """Check if a function is a registered tool."""
    return hasattr(func, "_is_energyai_tool")


def is_skill(cls: type) -> bool:
    """Check if a class is a registered skill."""
    return hasattr(cls, "_is_energyai_skill")


def is_prompt(func: Callable) -> bool:
    """Check if a function is a registered prompt."""
    return hasattr(func, "_is_energyai_prompt")


def is_planner(obj: Union[type, Callable]) -> bool:
    """Check if an object is a registered planner."""
    return hasattr(obj, "_is_energyai_planner")


def is_agent(cls: type) -> bool:
    """Check if a class is marked as an agent."""
    return hasattr(cls, "_is_energyai_agent")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Example tool
    @tool(name="example_calculator", description="Simple calculator tool")
    def calculate(a: float, b: float, operation: str = "add") -> dict:
        """Calculate mathematical operations."""
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else None,
        }
        return {"result": operations.get(operation)}

    # Example skill
    @skill(name="ExampleMath", description="Mathematical operations skill")
    class MathSkill:
        """Collection of mathematical tools."""

        @tool(name="power")
        def power(self, base: float, exponent: float) -> dict:
            """Calculate power."""
            return {"result": base**exponent}

    # Example prompt
    @prompt(name="analysis_prompt", variables=["topic", "context"])
    def analysis_prompt():
        """
        Analyze the {topic} in the context of {context}.
        Provide detailed insights and recommendations.
        """
        pass

    # Example planner
    @planner(name="example_planner", planner_type="sequential")
    class SequentialPlanner:
        """Example sequential planner."""

        def plan(self, request: str) -> list[str]:
            return ["step1", "step2", "step3"]

    # Example agent class
    @agent(name="ExampleAgent", tools=["example_calculator"], skills=["ExampleMath"])
    class ExampleAgentClass:
        """Example agent class."""

        pass

    print("Decorators example completed")
    print(f"Registered tools: {list(agent_registry.tools.keys())}")
    print(f"Registered skills: {list(agent_registry.skills.keys())}")
    print(f"Registered prompts: {list(agent_registry.prompts.keys())}")
    print(f"Registered planners: {list(agent_registry.planners.keys())}")
