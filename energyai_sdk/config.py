# Configuration management

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

try:
    from pydantic import BaseSettings, Field, validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Configuration models


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelType(str, Enum):
    """Supported model types."""

    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    AZURE_AI_INFERENCE = "azure_ai_inference"


@dataclass
class ModelConfig:
    """Configuration for a model."""

    deployment_name: str
    model_type: ModelType
    api_key: str
    endpoint: Optional[str] = None
    api_version: str = "2024-02-01"
    service_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_name": self.deployment_name,
            "model_type": self.model_type.value,
            "api_key": self.api_key,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "service_id": self.service_id or self.deployment_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "is_default": self.is_default,
        }


# ObservabilityConfig is deprecated - use MonitoringConfig from clients.monitoring
# This is kept for backward compatibility
@dataclass
class ObservabilityConfig:
    """Deprecated: Use MonitoringConfig from clients.monitoring instead."""

    # Service metadata
    service_name: str = "energyai-sdk"
    service_version: str = "1.0.0"
    environment: str = "production"

    # Langfuse LLM Observability
    enable_langfuse: bool = True
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # OpenTelemetry APM
    enable_opentelemetry: bool = False
    otlp_endpoint: Optional[str] = None  # For custom collectors
    azure_monitor_connection_string: Optional[str] = None

    # Sampling and export settings
    trace_sample_rate: float = 1.0
    metrics_export_interval: int = 5000  # 5 seconds

    # Feature flags
    enable_traces: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True


@dataclass
class SecurityConfig:
    """Configuration for security settings."""

    api_keys: list[str] = field(default_factory=list)
    bearer_tokens: list[str] = field(default_factory=list)
    enable_auth: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000
    enable_cors: bool = True
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class ApplicationConfig:
    """Configuration for application settings."""

    title: str = "EnergyAI Agentic SDK"
    version: str = "1.0.0"
    description: str = "AI Agent Platform for Energy Analytics"
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000
    debug: bool = False
    enable_gzip: bool = True
    max_message_length: int = 10000
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


if PYDANTIC_AVAILABLE:

    class EnergyAISettings(BaseSettings):
        """
        Comprehensive settings using Pydantic for type validation and environment variable loading.
        """

        # Application Settings
        app_title: str = Field("EnergyAI Agentic SDK", env="ENERGYAI_APP_TITLE")
        app_version: str = Field("1.0.0", env="ENERGYAI_APP_VERSION")
        app_description: str = Field(
            "AI Agent Platform for Energy Analytics", env="ENERGYAI_APP_DESCRIPTION"
        )
        app_host: str = Field("0.0.0.0", env="ENERGYAI_APP_HOST")  # nosec B104
        app_port: int = Field(8000, env="ENERGYAI_APP_PORT")
        app_debug: bool = Field(False, env="ENERGYAI_APP_DEBUG")

        # Model Configurations
        azure_openai_endpoint: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
        azure_openai_api_key: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
        azure_openai_deployment_name: Optional[str] = Field(
            None, env="AZURE_OPENAI_DEPLOYMENT_NAME"
        )
        azure_openai_api_version: str = Field("2024-02-01", env="AZURE_OPENAI_API_VERSION")

        openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
        openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")

        # Telemetry Settings
        azure_monitor_connection_string: Optional[str] = Field(
            None, env="AZURE_MONITOR_CONNECTION_STRING"
        )
        otlp_endpoint: Optional[str] = Field(None, env="OTLP_ENDPOINT")
        langfuse_public_key: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")
        langfuse_host: str = Field("https://cloud.langfuse.com", env="LANGFUSE_HOST")
        langfuse_environment: str = Field("production", env="LANGFUSE_ENVIRONMENT")

        # Security Settings
        api_keys: str = Field("", env="ENERGYAI_API_KEYS")  # Comma-separated
        bearer_tokens: str = Field("", env="ENERGYAI_BEARER_TOKENS")  # Comma-separated
        enable_auth: bool = Field(True, env="ENERGYAI_ENABLE_AUTH")
        enable_rate_limiting: bool = Field(True, env="ENERGYAI_ENABLE_RATE_LIMITING")
        max_requests_per_minute: int = Field(60, env="ENERGYAI_MAX_REQUESTS_PER_MINUTE")
        max_requests_per_hour: int = Field(1000, env="ENERGYAI_MAX_REQUESTS_PER_HOUR")

        # Application Features
        enable_caching: bool = Field(True, env="ENERGYAI_ENABLE_CACHING")
        cache_ttl_seconds: int = Field(300, env="ENERGYAI_CACHE_TTL_SECONDS")
        max_message_length: int = Field(10000, env="ENERGYAI_MAX_MESSAGE_LENGTH")
        enable_cors: bool = Field(True, env="ENERGYAI_ENABLE_CORS")

        # Logging
        log_level: LogLevel = Field(LogLevel.INFO, env="ENERGYAI_LOG_LEVEL")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False

        @validator("api_keys")
        def parse_api_keys(cls, v):
            if isinstance(v, str):
                return [key.strip() for key in v.split(",") if key.strip()]
            return v

        @validator("bearer_tokens")
        def parse_bearer_tokens(cls, v):
            if isinstance(v, str):
                return [token.strip() for token in v.split(",") if token.strip()]
            return v

        def get_model_configs(self) -> list[ModelConfig]:
            """Get model configurations from settings."""
            configs = []

            # Azure OpenAI configuration
            if (
                self.azure_openai_endpoint
                and self.azure_openai_api_key
                and self.azure_openai_deployment_name
            ):
                configs.append(
                    ModelConfig(
                        deployment_name=self.azure_openai_deployment_name,
                        model_type=ModelType.AZURE_OPENAI,
                        api_key=self.azure_openai_api_key,
                        endpoint=self.azure_openai_endpoint,
                        api_version=self.azure_openai_api_version,
                        is_default=True,
                    )
                )

            # OpenAI configuration
            if self.openai_api_key:
                configs.append(
                    ModelConfig(
                        deployment_name="gpt-4o",
                        model_type=ModelType.OPENAI,
                        api_key=self.openai_api_key,
                        endpoint=self.openai_base_url,
                        is_default=len(configs) == 0,  # Default if no Azure OpenAI
                    )
                )

            return configs

        def get_observability_config(self) -> ObservabilityConfig:
            """Get observability configuration. Deprecated: use get_monitoring_config instead."""
            return ObservabilityConfig(
                environment=self.langfuse_environment,
                enable_langfuse=bool(self.langfuse_public_key and self.langfuse_secret_key),
                langfuse_public_key=self.langfuse_public_key,
                langfuse_secret_key=self.langfuse_secret_key,
                langfuse_host=self.langfuse_host,
                enable_opentelemetry=bool(
                    self.azure_monitor_connection_string or self.otlp_endpoint
                ),
                azure_monitor_connection_string=self.azure_monitor_connection_string,
                otlp_endpoint=self.otlp_endpoint,
            )

        def get_monitoring_config(self):
            """Get monitoring configuration using the new unified client."""
            try:
                from .clients.monitoring import MonitoringConfig

                return MonitoringConfig(
                    environment=self.langfuse_environment,
                    enable_langfuse=bool(self.langfuse_public_key and self.langfuse_secret_key),
                    langfuse_public_key=self.langfuse_public_key,
                    langfuse_secret_key=self.langfuse_secret_key,
                    langfuse_host=self.langfuse_host,
                    enable_opentelemetry=bool(
                        self.azure_monitor_connection_string or self.otlp_endpoint
                    ),
                    azure_monitor_connection_string=self.azure_monitor_connection_string,
                    otlp_trace_endpoint=self.otlp_endpoint,
                    otlp_metrics_endpoint=self.otlp_endpoint,
                )
            except ImportError:
                # Fallback to ObservabilityConfig for backward compatibility
                return self.get_observability_config()

        def get_security_config(self) -> SecurityConfig:
            """Get security configuration."""
            return SecurityConfig(
                api_keys=self.api_keys if isinstance(self.api_keys, list) else [],
                bearer_tokens=self.bearer_tokens if isinstance(self.bearer_tokens, list) else [],
                enable_auth=self.enable_auth,
                enable_rate_limiting=self.enable_rate_limiting,
                max_requests_per_minute=self.max_requests_per_minute,
                max_requests_per_hour=self.max_requests_per_hour,
                enable_cors=self.enable_cors,
            )

        def get_application_config(self) -> ApplicationConfig:
            """Get application configuration."""
            return ApplicationConfig(
                title=self.app_title,
                version=self.app_version,
                description=self.app_description,
                host=self.app_host,
                port=self.app_port,
                debug=self.app_debug,
                max_message_length=self.max_message_length,
                enable_caching=self.enable_caching,
                cache_ttl_seconds=self.cache_ttl_seconds,
            )


# Configuration manager


class ConfigurationManager:
    """Manager for loading and managing configurations."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.logger = logging.getLogger(__name__)
        self._config_cache = {}

    def load_from_file(self, config_path: Union[str, Path]) -> dict[str, Any]:
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    try:
                        import yaml

                        return yaml.safe_load(f)
                    except ImportError:
                        raise ImportError("PyYAML is required to load YAML configuration files")
                elif config_path.suffix.lower() == ".json":
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")
            raise

    def load_from_env(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        if PYDANTIC_AVAILABLE:
            settings = EnergyAISettings()
            return settings.dict()
        else:
            # Manual environment variable loading
            return {
                "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "azure_openai_deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
                "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
                "azure_monitor_connection_string": os.getenv("AZURE_MONITOR_CONNECTION_STRING"),
                "app_debug": os.getenv("ENERGYAI_APP_DEBUG", "false").lower() == "true",
                "app_port": int(os.getenv("ENERGYAI_APP_PORT", "8000")),
                "log_level": os.getenv("ENERGYAI_LOG_LEVEL", "INFO"),
            }

    def get_settings(self) -> Union["EnergyAISettings", dict[str, Any]]:
        """Get settings object or dictionary."""
        if PYDANTIC_AVAILABLE:
            return EnergyAISettings()
        else:
            return self.load_from_env()


# Global configuration manager
config_manager = ConfigurationManager()


def get_settings() -> Union["EnergyAISettings", dict[str, Any]]:
    """Get global settings."""
    return config_manager.get_settings()


# Example application


from .agents import SemanticKernelAgent, bootstrap_agents
from .application import create_application
from .core import agent_registry, initialize_sdk

# Import SDK components
from .decorators import agent, skill, tool

# ==============================================================================
# ENERGY-SPECIFIC TOOLS
# ==============================================================================


@tool(
    name="calculate_lcoe", description="Calculate Levelized Cost of Energy for renewable projects"
)
def calculate_lcoe(
    capital_cost: float,
    annual_generation: float,
    operation_cost: float = 0.0,
    discount_rate: float = 0.08,
    lifetime_years: int = 25,
) -> dict[str, float]:
    """
    Calculate LCOE for renewable energy projects.

    Args:
        capital_cost: Initial capital investment ($)
        annual_generation: Annual energy generation (MWh)
        operation_cost: Annual operation and maintenance cost ($)
        discount_rate: Discount rate for NPV calculation
        lifetime_years: Project lifetime in years

    Returns:
        Dictionary with LCOE calculation results
    """
    # Present value of costs
    pv_factor = sum([(1 + discount_rate) ** -i for i in range(1, lifetime_years + 1)])
    pv_operation_costs = operation_cost * pv_factor
    total_pv_costs = capital_cost + pv_operation_costs

    # Present value of generation
    pv_generation = annual_generation * pv_factor

    # LCOE calculation
    lcoe = total_pv_costs / pv_generation

    return {
        "lcoe_per_mwh": lcoe,
        "total_pv_costs": total_pv_costs,
        "pv_generation": pv_generation,
        "capital_cost_share": capital_cost / total_pv_costs,
        "operation_cost_share": pv_operation_costs / total_pv_costs,
    }


@tool(
    name="capacity_factor_calculator",
    description="Calculate capacity factor for renewable energy systems",
)
def capacity_factor_calculator(
    technology: str, location_type: str, rated_capacity: float, actual_generation: float
) -> dict[str, Any]:
    """
    Calculate capacity factor and efficiency metrics.

    Args:
        technology: Type of technology (solar, wind, hydro)
        location_type: Location characteristics (offshore, onshore, desert, etc.)
        rated_capacity: Rated capacity in MW
        actual_generation: Actual annual generation in MWh

    Returns:
        Capacity factor analysis
    """
    max_theoretical_generation = rated_capacity * 8760  # 24/7 for a year
    capacity_factor = actual_generation / max_theoretical_generation

    # Technology-specific benchmarks
    benchmarks = {
        "solar": {"excellent": 0.25, "good": 0.20, "average": 0.15},
        "wind_onshore": {"excellent": 0.45, "good": 0.35, "average": 0.25},
        "wind_offshore": {"excellent": 0.55, "good": 0.45, "average": 0.35},
        "hydro": {"excellent": 0.60, "good": 0.45, "average": 0.35},
    }

    tech_key = technology.lower()
    if "wind" in tech_key:
        tech_key = (
            f"wind_{location_type.lower()}"
            if location_type.lower() in ["onshore", "offshore"]
            else "wind_onshore"
        )

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
        "capacity_factor": capacity_factor,
        "capacity_factor_percent": capacity_factor * 100,
        "performance_rating": performance_rating,
        "benchmark_excellent": benchmark["excellent"],
        "annual_generation_mwh": actual_generation,
        "theoretical_max_mwh": max_theoretical_generation,
        "technology": technology,
        "location_type": location_type,
    }


@tool(
    name="carbon_emissions_calculator",
    description="Calculate carbon emissions and offsets for energy projects",
)
def carbon_emissions_calculator(
    energy_source: str, annual_generation_mwh: float, displaced_source: str = "grid_average"
) -> dict[str, float]:
    """
    Calculate carbon emissions and offset potential.

    Args:
        energy_source: Type of energy source (solar, wind, natural_gas, coal, etc.)
        annual_generation_mwh: Annual energy generation in MWh
        displaced_source: What energy source is being displaced

    Returns:
        Carbon emissions analysis
    """
    # Emission factors in kg CO2/MWh
    emission_factors = {
        "coal": 820,
        "natural_gas": 490,
        "grid_average": 400,  # Approximate grid average
        "solar": 40,
        "wind": 11,
        "hydro": 24,
        "nuclear": 12,
        "biomass": 230,
    }

    source_emissions = emission_factors.get(energy_source.lower(), 0)
    displaced_emissions = emission_factors.get(displaced_source.lower(), 400)

    total_source_emissions = source_emissions * annual_generation_mwh
    total_displaced_emissions = displaced_emissions * annual_generation_mwh
    net_emissions_avoided = total_displaced_emissions - total_source_emissions

    return {
        "source_emissions_kg_co2": total_source_emissions,
        "displaced_emissions_kg_co2": total_displaced_emissions,
        "net_emissions_avoided_kg_co2": net_emissions_avoided,
        "net_emissions_avoided_tons_co2": net_emissions_avoided / 1000,
        "emission_factor_source": source_emissions,
        "emission_factor_displaced": displaced_emissions,
        "annual_generation_mwh": annual_generation_mwh,
    }


# ==============================================================================
# ENERGY ANALYSIS SKILLS
# ==============================================================================


@skill(name="EnergyEconomics", description="Financial analysis tools for energy projects")
class EnergyEconomics:
    """Collection of financial analysis tools for energy projects."""

    @tool(name="npv_calculator")
    def calculate_npv(
        self, initial_investment: float, annual_cash_flows: list[float], discount_rate: float = 0.08
    ) -> dict[str, float]:
        """Calculate Net Present Value of energy project."""
        npv = -initial_investment
        for i, cash_flow in enumerate(annual_cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** (i + 1))

        return {
            "npv": npv,
            "initial_investment": initial_investment,
            "discount_rate": discount_rate,
            "years": len(annual_cash_flows),
            "total_cash_flows": sum(annual_cash_flows),
        }

    @tool(name="payback_period")
    def calculate_payback_period(
        self, initial_investment: float, annual_cash_flows: list[float]
    ) -> dict[str, Any]:
        """Calculate payback period for energy project."""
        cumulative_cash_flow = 0
        for i, cash_flow in enumerate(annual_cash_flows):
            cumulative_cash_flow += cash_flow
            if cumulative_cash_flow >= initial_investment:
                payback_period = i + 1 - (cumulative_cash_flow - initial_investment) / cash_flow
                return {
                    "payback_period_years": payback_period,
                    "payback_year": i + 1,
                    "cumulative_cash_flow": cumulative_cash_flow,
                    "initial_investment": initial_investment,
                }

        return {
            "payback_period_years": None,
            "payback_year": None,
            "cumulative_cash_flow": cumulative_cash_flow,
            "initial_investment": initial_investment,
            "note": "Payback period exceeds project lifetime",
        }


@skill(name="MarketAnalysis", description="Energy market analysis and forecasting tools")
class MarketAnalysis:
    """Tools for energy market analysis."""

    @tool(name="price_trend_analyzer")
    def analyze_price_trends(
        self, historical_prices: list[float], periods: int = 12
    ) -> dict[str, Any]:
        """Analyze energy price trends."""
        if len(historical_prices) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Simple trend analysis
        recent_prices = (
            historical_prices[-periods:] if len(historical_prices) >= periods else historical_prices
        )
        average_price = sum(recent_prices) / len(recent_prices)

        # Calculate trend
        n = len(recent_prices)
        sum_x = sum(range(n))
        sum_y = sum(recent_prices)
        sum_xy = sum(i * price for i, price in enumerate(recent_prices))
        sum_x2 = sum(i * i for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

        return {
            "average_price": average_price,
            "trend_slope": slope,
            "trend_direction": trend_direction,
            "trend_strength": abs(slope),
            "periods_analyzed": len(recent_prices),
            "price_volatility": max(recent_prices) - min(recent_prices),
            "forecast_next_period": intercept + slope * n,
        }


# ==============================================================================
# SPECIALIZED AGENTS USING DECORATORS
# ==============================================================================


# Define specialized agents using the decorator approach
@agent(
    name="ConfigFinancialAnalyst",
    description="Specialized in financial analysis and economic modeling of energy projects",
    system_prompt="""You are an expert financial analyst specializing in energy project economics.
    You excel at:
    - LCOE calculations and financial modeling
    - NPV and ROI analysis
    - Risk assessment and sensitivity analysis
    - Investment recommendations

    Use the available tools to provide accurate calculations and data-driven insights.
    Always explain your analysis methodology and key assumptions.""",
    tools=["calculate_lcoe", "financial_metrics_calculator"],
)
class ConfigFinancialAnalyst:
    """Financial analyst agent for configuration-based setup."""

    temperature = 0.3
    max_tokens = 1000


@agent(
    name="ConfigTechnicalAnalyst",
    description="Expert in renewable energy technologies and performance analysis",
    system_prompt="""You are a technical expert in renewable energy systems.
    Your expertise includes:
    - Solar, wind, and other renewable technologies
    - Performance analysis and optimization
    - Capacity factor analysis
    - Technical feasibility studies

    Provide detailed technical insights and use calculations to support your analysis.""",
    tools=["capacity_factor_analysis", "carbon_footprint_calculator"],
)
class ConfigTechnicalAnalyst:
    """Technical analyst agent for configuration-based setup."""

    temperature = 0.4
    max_tokens = 1000


@agent(
    name="ConfigMarketAnalyst",
    description="Specialist in energy markets, pricing, and policy analysis",
    system_prompt="""You are an energy market analyst with deep knowledge of:
    - Energy market dynamics and pricing
    - Policy and regulatory impacts
    - Market trends and forecasting
    - Competitive analysis

    Provide market intelligence and strategic insights for energy investments.""",
)
class ConfigMarketAnalyst:
    """Market analyst agent for configuration-based setup."""

    temperature = 0.5
    max_tokens = 1000


def create_energy_agents(settings) -> list[SemanticKernelAgent]:
    """Create specialized energy analysis agents using the decorator-defined agents."""

    # Get model configurations
    if PYDANTIC_AVAILABLE and hasattr(settings, "get_model_configs"):
        model_configs = [config.to_dict() for config in settings.get_model_configs()]
    else:
        # Fallback configuration
        model_configs = [
            {
                "deployment_name": "gpt-4o",
                "endpoint": settings.get("azure_openai_endpoint")
                or "https://your-endpoint.openai.azure.com/",
                "api_key": settings.get("azure_openai_api_key") or "your-api-key",
                "service_type": "azure_openai",
                "is_default": True,
            }
        ]

    if (
        not model_configs
        or not model_configs[0].get("api_key")
        or model_configs[0]["api_key"] == "your-api-key"
    ):
        logging.warning(
            "No valid model configuration found. Agents will be created but may not work without proper API keys."
        )
        return []

    # Use bootstrap_agents to create the decorator-defined agents
    azure_config = model_configs[0] if model_configs else None
    if azure_config:
        created_agents = bootstrap_agents(azure_openai_config=azure_config)

        # Filter to only the config-specific agents
        config_agent_names = [
            "ConfigFinancialAnalyst",
            "ConfigTechnicalAnalyst",
            "ConfigMarketAnalyst",
        ]
        config_agents = [
            created_agents[name] for name in config_agent_names if name in created_agents
        ]

        return config_agents

    return []


# ==============================================================================
# MAIN APPLICATION SETUP
# ==============================================================================


def create_energy_platform(config_path: Optional[str] = None) -> "EnergyAIApplication":
    """Create a complete energy analysis platform."""

    # Load configuration
    if config_path:
        config = config_manager.load_from_file(config_path)
    else:
        config = config_manager.get_settings()

    # Initialize SDK with telemetry
    if PYDANTIC_AVAILABLE and hasattr(config, "get_observability_config"):
        observability_config = config.get_observability_config()
        initialize_sdk(
            azure_monitor_connection_string=observability_config.azure_monitor_connection_string,
            langfuse_public_key=observability_config.langfuse_public_key,
            langfuse_secret_key=observability_config.langfuse_secret_key,
            log_level=config.log_level.value if hasattr(config, "log_level") else "INFO",
        )
    else:
        initialize_sdk(
            azure_monitor_connection_string=config.get("azure_monitor_connection_string"),
            langfuse_public_key=config.get("langfuse_public_key"),
            langfuse_secret_key=config.get("langfuse_secret_key"),
            log_level=config.get("log_level", "INFO"),
        )

    # Create specialized agents
    agents = create_energy_agents(config)

    # Create application
    if PYDANTIC_AVAILABLE and hasattr(config, "get_application_config"):
        app_config = config.get_application_config()
        app = create_application(
            title=app_config.title,
            version=app_config.version,
            description=app_config.description,
            debug=app_config.debug,
        )
    else:
        app = create_application(
            title=config.get("app_title", "EnergyAI Platform"),
            version=config.get("app_version", "1.0.0"),
            debug=config.get("app_debug", False),
        )

    # Add agents to application
    for agent in agents:
        app.add_agent(agent)

    return app


def main():
    """Main entry point for the energy analysis platform."""
    try:
        # Create the platform
        app = create_energy_platform()

        # Get configuration for server settings
        config = config_manager.get_settings()

        if PYDANTIC_AVAILABLE and hasattr(config, "app_host"):
            host = config.app_host
            port = config.app_port
            debug = config.app_debug
        else:
            host = config.get("app_host", "127.0.0.1")
            port = config.get("app_port", 8000)
            debug = config.get("app_debug", False)

        # Log platform status
        capabilities = agent_registry.get_capabilities()
        logging.info(f"EnergyAI Platform initialized with {len(capabilities['agents'])} agents")
        logging.info(f"Available agents: {capabilities['agents']}")
        logging.info(f"Available tools: {capabilities['tools']}")
        logging.info(f"Available skills: {capabilities['skills']}")

        # Run development server
        from .application import DevelopmentServer

        server = DevelopmentServer(app, host=host, port=port)
        server.run(reload=debug)

    except Exception as e:
        logging.error(f"Error starting EnergyAI Platform: {e}")
        raise


if __name__ == "__main__":
    # Example .env file content:
    """
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    AZURE_OPENAI_API_KEY=your_api_key_here
    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

    # Telemetry Configuration
    LANGFUSE_PUBLIC_KEY=pk_lf_your_key
    LANGFUSE_SECRET_KEY=sk_lf_your_secret
    AZURE_MONITOR_CONNECTION_STRING=InstrumentationKey=your_key

    # Application Configuration
    ENERGYAI_APP_DEBUG=true
    ENERGYAI_APP_PORT=8000
    ENERGYAI_LOG_LEVEL=DEBUG

    # Security Configuration
    ENERGYAI_API_KEYS=key1,key2,key3
    ENERGYAI_ENABLE_AUTH=false
    ENERGYAI_MAX_REQUESTS_PER_MINUTE=100
    """

    main()
