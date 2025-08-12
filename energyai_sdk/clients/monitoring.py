"""
Monitoring and Observability Client for OpenTelemetry integration.

This client provides comprehensive monitoring capabilities using OpenTelemetry
for distributed tracing, metrics collection, and logging integration.
"""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

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

try:
    from azure.monitor.opentelemetry.exporter import (
        AzureMonitorMetricExporter,
        AzureMonitorTraceExporter,
    )

    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False

from ..exceptions import SDKError


@dataclass
class MonitoringConfig:
    """Configuration for monitoring setup."""

    service_name: str = "energyai-sdk"
    service_version: str = "1.0.0"
    environment: str = "development"

    # OpenTelemetry endpoints
    otlp_trace_endpoint: Optional[str] = None
    otlp_metrics_endpoint: Optional[str] = None

    # Azure Monitor
    azure_monitor_connection_string: Optional[str] = None

    # Sampling and export settings
    trace_sample_rate: float = 1.0
    metrics_export_interval: int = 5000  # 5 seconds

    # Feature flags
    enable_traces: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True


class MonitoringClient:
    """
    Client for integrated observability with OpenTelemetry.

    Provides distributed tracing, metrics collection, and structured logging
    for the EnergyAI SDK platform.
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize the Monitoring Client.

        Args:
            config: MonitoringConfig with monitoring settings
        """
        if not OTEL_AVAILABLE:
            raise SDKError(
                "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )

        self.config = config
        self.logger = logging.getLogger(__name__)

        self._tracer_provider: Optional[TracerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracer = None
        self._meter = None

        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize OpenTelemetry providers and instruments."""
        with self._lock:
            if self._initialized:
                return

            try:
                self._setup_resource()

                if self.config.enable_traces:
                    self._setup_tracing()

                if self.config.enable_metrics:
                    self._setup_metrics()

                if self.config.enable_logging:
                    self._setup_logging()

                self._initialized = True
                self.logger.info("Monitoring client initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize monitoring: {e}")
                raise SDKError(f"Monitoring initialization failed: {e}") from e

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
        # Create tracer provider
        self._tracer_provider = TracerProvider(
            resource=self._resource,
            sampler=trace.sampling.TraceIdRatioBased(self.config.trace_sample_rate),
        )

        # Setup exporters
        exporters = []

        # OTLP exporter
        if self.config.otlp_trace_endpoint:
            exporters.append(OTLPSpanExporter(endpoint=self.config.otlp_trace_endpoint))

        # Azure Monitor exporter
        if self.config.azure_monitor_connection_string and AZURE_MONITOR_AVAILABLE:
            exporters.append(
                AzureMonitorTraceExporter(
                    connection_string=self.config.azure_monitor_connection_string
                )
            )

        # Add span processors
        for exporter in exporters:
            processor = BatchSpanProcessor(exporter)
            self._tracer_provider.add_span_processor(processor)

        # Set global tracer provider
        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(__name__)

    def _setup_metrics(self) -> None:
        """Setup metrics collection."""
        # Setup exporters
        exporters = []

        # OTLP exporter
        if self.config.otlp_metrics_endpoint:
            exporters.append(OTLPMetricExporter(endpoint=self.config.otlp_metrics_endpoint))

        # Azure Monitor exporter
        if self.config.azure_monitor_connection_string and AZURE_MONITOR_AVAILABLE:
            exporters.append(
                AzureMonitorMetricExporter(
                    connection_string=self.config.azure_monitor_connection_string
                )
            )

        # Create metric readers
        readers = []
        for exporter in exporters:
            reader = PeriodicExportingMetricReader(
                exporter=exporter, export_interval_millis=self.config.metrics_export_interval
            )
            readers.append(reader)

        # Create meter provider
        self._meter_provider = MeterProvider(resource=self._resource, metric_readers=readers)

        # Set global meter provider
        metrics.set_meter_provider(self._meter_provider)
        self._meter = metrics.get_meter(__name__)

    def _setup_logging(self) -> None:
        """Setup structured logging integration."""
        LoggingInstrumentor().instrument(set_logging_format=True)

    def get_tracer(self, name: str = None) -> Optional[Any]:
        """Get OpenTelemetry tracer."""
        if not self._initialized:
            self.initialize()

        if self._tracer_provider:
            return trace.get_tracer(name or __name__)
        return None

    def get_meter(self, name: str = None) -> Optional[Any]:
        """Get OpenTelemetry meter."""
        if not self._initialized:
            self.initialize()

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
            # Add attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    def trace_function(self, name: str = None, **default_attributes):
        """Decorator to automatically trace function execution."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = name or f"{func.__module__}.{func.__name__}"

                with self.start_span(span_name, **default_attributes) as span:
                    if span:
                        # Add function metadata
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

            # Return appropriate wrapper based on function type
            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        meter = self.get_meter()
        if not meter:
            return

        try:
            # Create counter or histogram based on use case
            if name.endswith("_count") or name.endswith("_total"):
                counter = meter.create_counter(name)
                counter.add(value, tags or {})
            else:
                histogram = meter.create_histogram(name)
                histogram.record(value, tags or {})

        except Exception as e:
            self.logger.warning(f"Failed to record metric {name}: {e}")

    def create_counter(self, name: str, description: str = ""):
        """Create a counter metric."""
        meter = self.get_meter()
        if meter:
            return meter.create_counter(name, description=description)
        return None

    def create_histogram(self, name: str, description: str = ""):
        """Create a histogram metric."""
        meter = self.get_meter()
        if meter:
            return meter.create_histogram(name, description=description)
        return None

    def create_gauge(self, name: str, description: str = ""):
        """Create a gauge metric."""
        meter = self.get_meter()
        if meter:
            return meter.create_observable_gauge(name, description=description)
        return None

    def health_check(self) -> bool:
        """Check if monitoring is healthy."""
        try:
            if not self._initialized:
                return False

            # Test trace creation
            if self.config.enable_traces and self._tracer_provider:
                with self.start_span("health_check"):
                    pass

            # Test metric recording
            if self.config.enable_metrics and self._meter_provider:
                self.record_metric("health_check_count", 1.0)

            return True

        except Exception as e:
            self.logger.error(f"Monitoring health check failed: {e}")
            return False

    def shutdown(self):
        """Shutdown monitoring providers."""
        try:
            if self._tracer_provider:
                for processor in self._tracer_provider._active_span_processor._span_processors:
                    processor.shutdown()

            if self._meter_provider:
                self._meter_provider.shutdown()

            self.logger.info("Monitoring client shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during monitoring shutdown: {e}")


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

                    # Record success metrics
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
                    # Record error metrics
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

                    # Record success metrics
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
                    # Record error metrics
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
    _global_monitoring_client.initialize()

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
        self._initialized = True

    def initialize(self) -> None:
        """Mock initialization."""
        pass

    @contextmanager
    def start_span(self, name: str, **attributes):
        """Mock span creation."""
        span_data = {
            "name": name,
            "start_time": time.time(),
            "attributes": attributes,
            "span_id": str(uuid.uuid4()),
        }

        self._traces.append(span_data)
        yield span_data

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Mock metric recording."""
        metric_data = {"name": name, "value": value, "tags": tags or {}, "timestamp": time.time()}
        self._metrics.append(metric_data)

    def get_recorded_traces(self) -> List[Dict[str, Any]]:
        """Get all recorded trace data."""
        return self._traces.copy()

    def get_recorded_metrics(self) -> List[Dict[str, Any]]:
        """Get all recorded metric data."""
        return self._metrics.copy()

    def clear_data(self):
        """Clear all recorded data."""
        self._traces.clear()
        self._metrics.clear()

    def health_check(self) -> bool:
        """Mock health check."""
        return True

    def shutdown(self):
        """Mock shutdown."""
        self.clear_data()
