"""
Unified Monitoring and Observability Client for EnergyAI SDK.

This consolidated module provides comprehensive monitoring capabilities including:
- Langfuse for LLM-specific observability (traces, generations, etc.)
- OpenTelemetry for general application monitoring (metrics, traces, logs)
- Azure Monitor integration for cloud deployments

The MonitoringManager acts as a facade over these systems, providing
a single, consistent API for all monitoring needs.
"""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

# OpenTelemetry imports
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Azure Monitor imports
try:
    from azure.monitor.opentelemetry.exporter import (
        AzureMonitorMetricExporter,
        AzureMonitorTraceExporter,
    )

    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False

# Langfuse imports
try:
    from langfuse import Langfuse
    from langfuse.model import CreateGeneration, CreateSpan, CreateTrace

    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    CreateTrace = None
    CreateGeneration = None
    CreateSpan = None
    LANGFUSE_AVAILABLE = False



@dataclass
class MonitoringConfig:
    """Unified configuration for all monitoring systems."""

    # Service metadata
    service_name: str = "energyai-sdk"
    service_version: str = "1.0.0"
    environment: str = "development"

    # OpenTelemetry endpoints
    otlp_trace_endpoint: Optional[str] = None
    otlp_metrics_endpoint: Optional[str] = None

    # Azure Monitor
    azure_monitor_connection_string: Optional[str] = None

    # Langfuse LLM Observability
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Sampling and export settings
    trace_sample_rate: float = 1.0
    metrics_export_interval: int = 5000  # 5 seconds

    # Feature flags
    enable_traces: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_langfuse: bool = True
    enable_opentelemetry: bool = True

    # Debug settings
    debug: bool = False


class LangfuseClient:
    """
    Internal Langfuse client for LLM-specific observability.

    Handles traces, generations, and spans specifically for LLM interactions.
    """

    def __init__(self, config: MonitoringConfig):
        """Initialize the Langfuse client."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.langfuse")
        self.client = None
        self.enabled = False

        if not LANGFUSE_AVAILABLE:
            self.logger.warning("Langfuse not available. Install with: pip install langfuse")
            return

        if not config.langfuse_public_key or not config.langfuse_secret_key:
            self.logger.warning("Langfuse credentials not provided. LLM observability disabled.")
            return

        try:
            self.client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                host=config.langfuse_host,
                debug=config.debug,
            )
            self.enabled = True
            self.logger.info(f"Langfuse client initialized for environment: {config.environment}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Langfuse client: {e}")

    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled."""
        return self.enabled and self.client is not None

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Create a Langfuse trace."""
        if not self.is_enabled():
            return None

        try:
            trace_data = {
                "name": name,
                "timestamp": datetime.now(timezone.utc),
            }

            if user_id:
                trace_data["user_id"] = user_id
            if session_id:
                trace_data["session_id"] = session_id
            if input_data:
                trace_data["input"] = input_data

            combined_metadata = {
                "environment": self.config.environment,
                "sdk_version": self.config.service_version,
                "trace_id": str(uuid.uuid4()),
            }
            if metadata:
                combined_metadata.update(metadata)
            trace_data["metadata"] = combined_metadata

            if tags:
                trace_data["tags"] = tags

            trace = self.client.trace(**trace_data)

            if self.config.debug:
                self.logger.debug(f"Created Langfuse trace: {name}")

            return trace

        except Exception as e:
            self.logger.error(f"Failed to create Langfuse trace: {e}")
            return None

    def create_generation(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Create a generation within a trace."""
        if not self.is_enabled() or not trace:
            return None

        try:
            generation_data = {
                "name": name,
                "start_time": datetime.now(timezone.utc),
            }

            if input_data:
                generation_data["input"] = input_data
            if model:
                generation_data["model"] = model
            if model_parameters:
                generation_data["model_parameters"] = model_parameters

            combined_metadata = {"generation_id": str(uuid.uuid4())}
            if metadata:
                combined_metadata.update(metadata)
            generation_data["metadata"] = combined_metadata

            generation = trace.generation(**generation_data)

            if self.config.debug:
                self.logger.debug(f"Created Langfuse generation: {name}")

            return generation

        except Exception as e:
            self.logger.error(f"Failed to create Langfuse generation: {e}")
            return None

    def end_generation(
        self,
        generation,
        output: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """End a generation."""
        if not self.is_enabled() or not generation:
            return

        try:
            end_data = {"end_time": end_time or datetime.now(timezone.utc), "level": level}

            if output is not None:
                end_data["output"] = output
            if usage:
                end_data["usage"] = usage
            if status_message:
                end_data["status_message"] = status_message

            generation.end(**end_data)

            if self.config.debug:
                self.logger.debug(f"Ended Langfuse generation with level: {level}")

        except Exception as e:
            self.logger.error(f"Failed to end Langfuse generation: {e}")

    def update_trace(
        self,
        trace,
        output: Optional[Any] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a trace."""
        if not self.is_enabled() or not trace:
            return

        try:
            update_data = {"level": level}

            if output is not None:
                update_data["output"] = output
            if status_message:
                update_data["status_message"] = status_message
            if metadata:
                update_data["metadata"] = metadata

            trace.update(**update_data)

            if self.config.debug:
                self.logger.debug(f"Updated Langfuse trace with level: {level}")

        except Exception as e:
            self.logger.error(f"Failed to update Langfuse trace: {e}")

    def flush(self) -> None:
        """Flush Langfuse data."""
        if not self.is_enabled():
            return

        try:
            self.client.flush()
            if self.config.debug:
                self.logger.debug("Flushed Langfuse telemetry data")
        except Exception as e:
            self.logger.error(f"Failed to flush Langfuse data: {e}")

    def health_check(self) -> bool:
        """Check Langfuse health."""
        if not self.is_enabled():
            return False

        try:
            test_trace = self.create_trace(
                name="health-check", metadata={"test": True, "timestamp": time.time()}
            )
            if test_trace:
                self.update_trace(test_trace, output="health-check-success")
                self.flush()
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Langfuse health check failed: {e}")
            return False


class OpenTelemetryClient:
    """
    Internal OpenTelemetry client for application monitoring.

    Handles metrics, distributed tracing, and logging integration.
    """

    def __init__(self, config: MonitoringConfig):
        """Initialize the OpenTelemetry client."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.otel")

        self._tracer_provider: Optional[TracerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracer = None
        self._meter = None
        self._initialized = False
        self._lock = threading.Lock()

        if not OTEL_AVAILABLE:
            self.logger.warning(
                "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
            return

        try:
            self._setup_resource()

            if config.enable_traces:
                self._setup_tracing()

            if config.enable_metrics:
                self._setup_metrics()

            if config.enable_logging:
                self._setup_logging()

            self._initialized = True
            self.logger.info("OpenTelemetry client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")

    def _setup_resource(self) -> None:
        """Setup OpenTelemetry resource."""
        self._resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
                ResourceAttributes.SERVICE_VERSION: self.config.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            }
        )

    def _setup_tracing(self) -> None:
        """Setup distributed tracing."""
        self._tracer_provider = TracerProvider(
            resource=self._resource,
            sampler=trace.sampling.TraceIdRatioBased(self.config.trace_sample_rate),
        )

        exporters = []

        if self.config.otlp_trace_endpoint:
            exporters.append(OTLPSpanExporter(endpoint=self.config.otlp_trace_endpoint))

        if self.config.azure_monitor_connection_string and AZURE_MONITOR_AVAILABLE:
            exporters.append(
                AzureMonitorTraceExporter(
                    connection_string=self.config.azure_monitor_connection_string
                )
            )

        for exporter in exporters:
            processor = BatchSpanProcessor(exporter)
            self._tracer_provider.add_span_processor(processor)

        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(__name__)

    def _setup_metrics(self) -> None:
        """Setup metrics collection."""
        exporters = []

        if self.config.otlp_metrics_endpoint:
            exporters.append(OTLPMetricExporter(endpoint=self.config.otlp_metrics_endpoint))

        if self.config.azure_monitor_connection_string and AZURE_MONITOR_AVAILABLE:
            exporters.append(
                AzureMonitorMetricExporter(
                    connection_string=self.config.azure_monitor_connection_string
                )
            )

        readers = []
        for exporter in exporters:
            reader = PeriodicExportingMetricReader(
                exporter=exporter, export_interval_millis=self.config.metrics_export_interval
            )
            readers.append(reader)

        self._meter_provider = MeterProvider(resource=self._resource, metric_readers=readers)
        metrics.set_meter_provider(self._meter_provider)
        self._meter = metrics.get_meter(__name__)

    def _setup_logging(self) -> None:
        """Setup structured logging integration."""
        LoggingInstrumentor().instrument(set_logging_format=True)

    def is_enabled(self) -> bool:
        """Check if OpenTelemetry is enabled."""
        return self._initialized

    def get_tracer(self, name: str = None) -> Optional[Any]:
        """Get OpenTelemetry tracer."""
        if not self._initialized:
            return None
        if self._tracer_provider:
            return trace.get_tracer(name or __name__)
        return None

    def get_meter(self, name: str = None) -> Optional[Any]:
        """Get OpenTelemetry meter."""
        if not self._initialized:
            return None
        if self._meter_provider:
            return metrics.get_meter(name or __name__)
        return None

    @contextmanager
    def start_span(self, name: str, **attributes):
        """Start a new span with automatic lifecycle management."""
        tracer = self.get_tracer()
        if not tracer:
            yield None
            return

        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        meter = self.get_meter()
        if not meter:
            return

        try:
            if name.endswith("_count") or name.endswith("_total"):
                counter = meter.create_counter(name)
                counter.add(value, tags or {})
            else:
                histogram = meter.create_histogram(name)
                histogram.record(value, tags or {})
        except Exception as e:
            self.logger.warning(f"Failed to record metric {name}: {e}")

    def health_check(self) -> bool:
        """Check OpenTelemetry health."""
        try:
            if not self._initialized:
                return False

            if self.config.enable_traces and self._tracer_provider:
                with self.start_span("health_check"):
                    pass

            if self.config.enable_metrics and self._meter_provider:
                self.record_metric("health_check_count", 1.0)

            return True
        except Exception as e:
            self.logger.error(f"OpenTelemetry health check failed: {e}")
            return False

    def shutdown(self):
        """Shutdown OpenTelemetry providers."""
        try:
            if self._tracer_provider:
                for processor in self._tracer_provider._active_span_processor._span_processors:
                    processor.shutdown()

            if self._meter_provider:
                self._meter_provider.shutdown()

            self.logger.info("OpenTelemetry client shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during OpenTelemetry shutdown: {e}")


class MonitoringClient:
    """
    Unified monitoring client that provides a single interface for all monitoring needs.

    This client integrates:
    - Langfuse for LLM observability
    - OpenTelemetry for application monitoring
    - Azure Monitor for cloud deployments

    It acts as a facade over these systems, providing a consistent API regardless
    of which backends are available or configured.
    """

    def __init__(self, config: MonitoringConfig):
        """Initialize the unified monitoring client."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize sub-clients
        self.langfuse_client = None
        self.otel_client = None

        if config.enable_langfuse and LANGFUSE_AVAILABLE:
            self.langfuse_client = LangfuseClient(config)

        if config.enable_opentelemetry and OTEL_AVAILABLE:
            self.otel_client = OpenTelemetryClient(config)

        self._log_initialization_status()

    def _log_initialization_status(self) -> None:
        """Log initialization status."""
        status = {
            "langfuse": (
                "enabled"
                if self.langfuse_client and self.langfuse_client.is_enabled()
                else "disabled"
            ),
            "opentelemetry": (
                "enabled" if self.otel_client and self.otel_client.is_enabled() else "disabled"
            ),
        }
        self.logger.info(f"MonitoringClient initialized: {status}")

    # ==== Langfuse LLM Observability Methods ====

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_data: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """Create a Langfuse trace for LLM interactions."""
        if self.langfuse_client:
            return self.langfuse_client.create_trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                input_data=input_data,
                metadata=metadata,
                tags=tags,
            )
        return None

    def create_generation(
        self,
        trace,
        name: str,
        input_data: Optional[Dict] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Any:
        """Create a Langfuse generation for LLM calls."""
        if self.langfuse_client:
            return self.langfuse_client.create_generation(
                trace=trace,
                name=name,
                input_data=input_data,
                model=model,
                model_parameters=model_parameters,
                metadata=metadata,
            )
        return None

    def end_generation(
        self,
        generation,
        output: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
    ) -> None:
        """End a Langfuse generation."""
        if self.langfuse_client:
            self.langfuse_client.end_generation(
                generation=generation,
                output=output,
                usage=usage,
                level=level,
                status_message=status_message,
            )

    def update_trace(
        self,
        trace,
        output: Optional[Any] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Update a Langfuse trace."""
        if self.langfuse_client:
            self.langfuse_client.update_trace(
                trace=trace,
                output=output,
                level=level,
                status_message=status_message,
                metadata=metadata,
            )

    # ==== OpenTelemetry Application Monitoring Methods ====

    @contextmanager
    def start_span(self, name: str, **attributes):
        """Start an OpenTelemetry span."""
        if self.otel_client:
            with self.otel_client.start_span(name, **attributes) as span:
                yield span
        else:
            yield None

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric with OpenTelemetry."""
        if self.otel_client:
            self.otel_client.record_metric(name, value, tags)

    def trace_function(self, name: str = None, **default_attributes):
        """Decorator to automatically trace function execution."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = name or f"{func.__module__}.{func.__name__}"
                with self.start_span(span_name, **default_attributes) as span:
                    if span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                    return func(*args, **kwargs)

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_name = name or f"{func.__module__}.{func.__name__}"
                with self.start_span(span_name, **default_attributes) as span:
                    if span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                    return await func(*args, **kwargs)

            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def create_counter(self, name: str, description: str = ""):
        """Create a counter metric."""
        if self.otel_client:
            meter = self.otel_client.get_meter()
            if meter:
                return meter.create_counter(name, description=description)
        return None

    def create_histogram(self, name: str, description: str = ""):
        """Create a histogram metric."""
        if self.otel_client:
            meter = self.otel_client.get_meter()
            if meter:
                return meter.create_histogram(name, description=description)
        return None

    def create_gauge(self, name: str, description: str = ""):
        """Create a gauge metric."""
        if self.otel_client:
            meter = self.otel_client.get_meter()
            if meter:
                return meter.create_observable_gauge(name, description=description)
        return None

    # ==== Unified Management Methods ====

    def health_check(self) -> Dict[str, bool]:
        """Check health of all monitoring systems."""
        health = {"langfuse": False, "opentelemetry": False, "overall": False}

        if self.langfuse_client:
            health["langfuse"] = self.langfuse_client.health_check()

        if self.otel_client:
            health["opentelemetry"] = self.otel_client.health_check()

        health["overall"] = health["langfuse"] or health["opentelemetry"]
        return health

    def flush(self) -> None:
        """Flush all pending telemetry data."""
        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
            except Exception as e:
                self.logger.warning(f"Failed to flush Langfuse data: {e}")

    def shutdown(self):
        """Shutdown all monitoring systems."""
        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
                self.logger.info("Langfuse client shutdown complete")
            except Exception as e:
                self.logger.warning(f"Error during Langfuse shutdown: {e}")

        if self.otel_client:
            try:
                self.otel_client.shutdown()
                self.logger.info("OpenTelemetry client shutdown complete")
            except Exception as e:
                self.logger.warning(f"Error during OpenTelemetry shutdown: {e}")


# Convenience functions for agent and tool monitoring
def monitor_agent_execution(agent_name: str, monitoring_client: MonitoringClient):
    """Decorator for monitoring agent execution."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitoring_client.start_span(
                "agent.execute", agent_name=agent_name, agent_type="energyai_agent"
            ) as span:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    monitoring_client.record_metric(
                        "agent_execution_duration",
                        duration,
                        {"agent_name": agent_name, "status": "success"},
                    )
                    monitoring_client.record_metric(
                        "agent_execution_count",
                        1.0,
                        {"agent_name": agent_name, "status": "success"},
                    )

                    if span:
                        span.set_attribute("execution.duration", duration)
                        span.set_attribute("execution.status", "success")

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    monitoring_client.record_metric(
                        "agent_execution_duration",
                        duration,
                        {"agent_name": agent_name, "status": "error"},
                    )
                    monitoring_client.record_metric(
                        "agent_execution_count", 1.0, {"agent_name": agent_name, "status": "error"}
                    )

                    if span:
                        span.set_attribute("execution.duration", duration)
                        span.set_attribute("execution.status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))

                    raise

        return wrapper

    return decorator


def monitor_tool_execution(tool_name: str, monitoring_client: MonitoringClient):
    """Decorator for monitoring tool execution."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitoring_client.start_span(
                "tool.execute", tool_name=tool_name, tool_type="energyai_tool"
            ) as span:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    monitoring_client.record_metric(
                        "tool_execution_duration",
                        duration,
                        {"tool_name": tool_name, "status": "success"},
                    )
                    monitoring_client.record_metric(
                        "tool_execution_count", 1.0, {"tool_name": tool_name, "status": "success"}
                    )

                    if span:
                        span.set_attribute("execution.duration", duration)
                        span.set_attribute("execution.status", "success")

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    monitoring_client.record_metric(
                        "tool_execution_duration",
                        duration,
                        {"tool_name": tool_name, "status": "error"},
                    )
                    monitoring_client.record_metric(
                        "tool_execution_count", 1.0, {"tool_name": tool_name, "status": "error"}
                    )

                    if span:
                        span.set_attribute("execution.duration", duration)
                        span.set_attribute("execution.status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))

                    raise

        return wrapper

    return decorator


# Global monitoring client instance
_global_monitoring_client: Optional[MonitoringClient] = None


def get_monitoring_client() -> Optional[MonitoringClient]:
    """Get the global monitoring client instance."""
    return _global_monitoring_client


def initialize_monitoring(config: MonitoringConfig) -> MonitoringClient:
    """Initialize global monitoring client."""
    global _global_monitoring_client
    _global_monitoring_client = MonitoringClient(config)
    return _global_monitoring_client


def shutdown_monitoring():
    """Shutdown global monitoring client."""
    global _global_monitoring_client
    if _global_monitoring_client:
        _global_monitoring_client.shutdown()
        _global_monitoring_client = None


# Mock client for development/testing
class MockMonitoringClient(MonitoringClient):
    """Mock monitoring client for development and testing."""

    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        self._traces = []
        self._metrics = []
        self._langfuse_traces = []

        # Mock sub-clients
        self.langfuse_client = MagicMock()
        self.langfuse_client.is_enabled.return_value = True
        self.otel_client = MagicMock()
        self.otel_client.is_enabled.return_value = True

    def create_trace(self, name: str, **kwargs) -> Any:
        """Mock trace creation."""
        trace_data = {
            "name": name,
            "start_time": time.time(),
            "attributes": kwargs,
            "trace_id": str(uuid.uuid4()),
        }
        self._langfuse_traces.append(trace_data)
        return trace_data

    def create_generation(self, trace, name: str, **kwargs) -> Any:
        """Mock generation creation."""
        generation_data = {
            "trace": trace,
            "name": name,
            "start_time": time.time(),
            "attributes": kwargs,
            "generation_id": str(uuid.uuid4()),
        }
        return generation_data

    def end_generation(self, generation, **kwargs) -> None:
        """Mock end generation."""
        pass

    def update_trace(self, trace, **kwargs) -> None:
        """Mock update trace."""
        pass

    @contextmanager
    def start_span(self, name: str, **attributes):
        """Mock span creation."""

        class MockSpan:
            def __init__(self, name: str, **attributes):
                self.name = name
                self.start_time = time.time()
                self.attributes = attributes
                self.span_id = str(uuid.uuid4())

            def set_attribute(self, key: str, value: Any):
                self.attributes[key] = value

        span_data = MockSpan(name, **attributes)
        self._traces.append(
            {
                "name": span_data.name,
                "start_time": span_data.start_time,
                "attributes": span_data.attributes,
                "span_id": span_data.span_id,
            }
        )
        yield span_data

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Mock metric recording."""
        metric_data = {"name": name, "value": value, "tags": tags or {}, "timestamp": time.time()}
        self._metrics.append(metric_data)

    def get_recorded_traces(self) -> List[Dict[str, Any]]:
        """Get all recorded trace data."""
        return self._traces.copy()

    def get_recorded_langfuse_traces(self) -> List[Dict[str, Any]]:
        """Get all recorded Langfuse trace data."""
        return self._langfuse_traces.copy()

    def get_recorded_metrics(self) -> List[Dict[str, Any]]:
        """Get all recorded metric data."""
        return self._metrics.copy()

    def clear_data(self):
        """Clear all recorded data."""
        self._traces.clear()
        self._metrics.clear()
        self._langfuse_traces.clear()

    def health_check(self) -> Dict[str, bool]:
        """Mock health check."""
        return {"langfuse": True, "opentelemetry": True, "overall": True}

    def flush(self) -> None:
        """Mock flush."""
        pass

    def shutdown(self):
        """Mock shutdown."""
        self.clear_data()


# Convenience function for unified monitoring decorator
def monitor(name: str, **attributes) -> Callable:
    """
    Decorator to monitor function execution with the appropriate backend.

    Uses OpenTelemetry for general function monitoring.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = get_monitoring_client()
            if client is None:
                return func(*args, **kwargs)

            with client.start_span(name, **attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator
