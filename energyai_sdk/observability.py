"""
Unified Observability Manager for EnergyAI SDK.

This module provides a centralized system for all observability needs:
- Langfuse for LLM-specific observability (traces, generations, etc.)
- OpenTelemetry for general application monitoring (metrics, traces, logs)
- Azure Monitor integration for cloud deployments

The ObservabilityManager acts as a facade over these systems, providing
a single, consistent API for all monitoring needs.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable

from .config import ObservabilityConfig

# Import clients conditionally to handle missing dependencies gracefully
try:
    from .clients.langfuse_client import (
        LangfuseMonitoringClient,
        get_langfuse_client,
        LANGFUSE_AVAILABLE,
    )
except ImportError:
    LangfuseMonitoringClient = None
    get_langfuse_client = None
    LANGFUSE_AVAILABLE = False

try:
    from .clients.monitoring import (
        MonitoringClient,
        MonitoringConfig,
        initialize_monitoring,
        OTEL_AVAILABLE,
    )
except ImportError:
    MonitoringClient = None
    MonitoringConfig = None
    initialize_monitoring = None
    OTEL_AVAILABLE = False


class ObservabilityManager:
    """
    Unified manager for all observability and monitoring needs.
    
    This class provides a single interface for:
    - LLM observability (Langfuse)
    - Application monitoring (OpenTelemetry)
    - Logging integration
    
    It handles the lifecycle of all monitoring clients and provides
    a consistent API regardless of which backends are available.
    """
    
    def __init__(self, config: ObservabilityConfig):
        """
        Initialize the ObservabilityManager with the unified configuration.
        
        Args:
            config: ObservabilityConfig with all monitoring settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients based on configuration and available dependencies
        self.langfuse_client = None
        self.otel_client = None
        
        # Track initialization status
        self._langfuse_initialized = False
        self._otel_initialized = False
        
        # Initialize Langfuse for LLM observability
        if config.enable_langfuse and LANGFUSE_AVAILABLE:
            self._initialize_langfuse()
        
        # Initialize OpenTelemetry for application monitoring
        if config.enable_opentelemetry and OTEL_AVAILABLE:
            self._initialize_opentelemetry()
            
        # Log initialization status
        self._log_initialization_status()
    
    def _initialize_langfuse(self) -> None:
        """Initialize the Langfuse client for LLM observability."""
        try:
            self.langfuse_client = get_langfuse_client(
                public_key=self.config.langfuse_public_key,
                secret_key=self.config.langfuse_secret_key,
                host=self.config.langfuse_host,
                debug=False,  # Set based on log level if needed
                environment=self.config.environment
            )
            self._langfuse_initialized = self.langfuse_client is not None and self.langfuse_client.is_enabled()
            
            if self._langfuse_initialized:
                self.logger.info("Langfuse LLM observability initialized successfully")
            else:
                self.logger.warning("Langfuse client created but not enabled (missing credentials?)")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize Langfuse client: {e}")
            self._langfuse_initialized = False
    
    def _initialize_opentelemetry(self) -> None:
        """Initialize the OpenTelemetry client for application monitoring."""
        try:
            # Convert our unified config to the MonitoringClient's config
            monitoring_config = MonitoringConfig(
                service_name=self.config.service_name,
                service_version=self.config.service_version,
                environment=self.config.environment,
                otlp_trace_endpoint=self.config.otlp_endpoint,
                otlp_metrics_endpoint=self.config.otlp_endpoint,
                azure_monitor_connection_string=self.config.azure_monitor_connection_string,
                trace_sample_rate=self.config.trace_sample_rate,
                metrics_export_interval=self.config.metrics_export_interval,
                enable_traces=self.config.enable_traces,
                enable_metrics=self.config.enable_metrics,
                enable_logging=self.config.enable_logging
            )
            
            self.otel_client = initialize_monitoring(monitoring_config)
            self._otel_initialized = self.otel_client is not None
            
            if self._otel_initialized:
                self.logger.info("OpenTelemetry monitoring initialized successfully")
            else:
                self.logger.warning("OpenTelemetry client created but not initialized")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenTelemetry client: {e}")
            self._otel_initialized = False
    
    def _log_initialization_status(self) -> None:
        """Log the initialization status of all monitoring systems."""
        status = {
            "langfuse": "enabled" if self._langfuse_initialized else "disabled",
            "opentelemetry": "enabled" if self._otel_initialized else "disabled",
        }
        
        self.logger.info(f"ObservabilityManager initialized: {status}")
    
    # ==== Langfuse LLM Observability Methods ====
    
    def create_trace(self, name: str, user_id: Optional[str] = None, 
                    session_id: Optional[str] = None, input_data: Optional[Dict] = None,
                    metadata: Optional[Dict] = None, tags: Optional[List[str]] = None) -> Any:
        """
        Create a Langfuse trace for an LLM interaction.
        
        Args:
            name: Name of the trace (e.g., "agent-run:EnergyAdvisor")
            user_id: User identifier
            session_id: Session identifier
            input_data: Input data for the trace
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            Langfuse trace object or None if Langfuse is disabled
        """
        if not self._langfuse_initialized:
            return None
            
        return self.langfuse_client.create_trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            input_data=input_data,
            metadata=metadata,
            tags=tags
        )
    
    def create_generation(self, trace, name: str, input_data: Optional[Dict] = None,
                         model: Optional[str] = None, model_parameters: Optional[Dict] = None,
                         metadata: Optional[Dict] = None) -> Any:
        """
        Create a Langfuse generation for an LLM call.
        
        Args:
            trace: Parent trace object
            name: Generation name
            input_data: Input to the LLM
            model: Model name
            model_parameters: Model parameters
            metadata: Additional metadata
            
        Returns:
            Langfuse generation object or None if Langfuse is disabled
        """
        if not self._langfuse_initialized or trace is None:
            return None
            
        return self.langfuse_client.create_generation(
            trace=trace,
            name=name,
            input_data=input_data,
            model=model,
            model_parameters=model_parameters,
            metadata=metadata
        )
    
    def end_generation(self, generation, output: Optional[Any] = None,
                      usage: Optional[Dict[str, int]] = None, level: str = "DEFAULT",
                      status_message: Optional[str] = None) -> None:
        """
        End a Langfuse generation with output and usage information.
        
        Args:
            generation: Generation object to end
            output: Generation output
            usage: Token usage information
            level: Log level (DEFAULT, WARNING, ERROR)
            status_message: Status message for errors
        """
        if not self._langfuse_initialized or generation is None:
            return
            
        self.langfuse_client.end_generation(
            generation=generation,
            output=output,
            usage=usage,
            level=level,
            status_message=status_message
        )
    
    def update_trace(self, trace, output: Optional[Any] = None,
                    level: str = "DEFAULT", status_message: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> None:
        """
        Update a Langfuse trace with final information.
        
        Args:
            trace: Trace object to update
            output: Final output
            level: Log level (DEFAULT, WARNING, ERROR)
            status_message: Status message
            metadata: Additional metadata to merge
        """
        if not self._langfuse_initialized or trace is None:
            return
            
        self.langfuse_client.update_trace(
            trace=trace,
            output=output,
            level=level,
            status_message=status_message,
            metadata=metadata
        )
    
    # ==== OpenTelemetry Application Monitoring Methods ====
    
    def start_span(self, name: str, **attributes) -> Any:
        """
        Start an OpenTelemetry span for operation tracking.
        
        Args:
            name: Span name
            **attributes: Span attributes
            
        Returns:
            Context manager for the span
        """
        if not self._otel_initialized:
            # Return a dummy context manager if OpenTelemetry is not available
            class DummyContextManager:
                def __enter__(self): return None
                def __exit__(self, *args): pass
                
            return DummyContextManager()
            
        return self.otel_client.start_span(name, **attributes)
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value with OpenTelemetry.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Tags for the metric
        """
        if not self._otel_initialized:
            return
            
        self.otel_client.record_metric(name, value, tags)
    
    def trace_function(self, name: str = None, **default_attributes) -> Callable:
        """
        Decorator to automatically trace function execution.
        
        Args:
            name: Optional name for the span
            **default_attributes: Default attributes for the span
            
        Returns:
            Decorated function
        """
        if not self._otel_initialized:
            # Return a no-op decorator if OpenTelemetry is not available
            def no_op_decorator(func):
                return func
            return no_op_decorator
            
        return self.otel_client.trace_function(name, **default_attributes)
    
    # ==== Unified Health Check and Management Methods ====
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check the health of all monitoring systems.
        
        Returns:
            Dictionary with health status of each system
        """
        health = {
            "langfuse": False,
            "opentelemetry": False,
        }
        
        # Check Langfuse health
        if self._langfuse_initialized:
            try:
                health["langfuse"] = self.langfuse_client.health_check()
            except Exception:
                health["langfuse"] = False
        
        # Check OpenTelemetry health
        if self._otel_initialized:
            try:
                health["opentelemetry"] = self.otel_client.health_check()
            except Exception:
                health["opentelemetry"] = False
        
        return health
    
    def flush(self) -> None:
        """Flush all pending telemetry data to their respective backends."""
        # Flush Langfuse data
        if self._langfuse_initialized:
            try:
                self.langfuse_client.flush()
            except Exception as e:
                self.logger.warning(f"Failed to flush Langfuse data: {e}")
        
        # OpenTelemetry doesn't have an explicit flush method
        # It uses periodic exporters that flush automatically
    
    def shutdown(self) -> None:
        """Shutdown all monitoring clients."""
        # Shutdown Langfuse
        if self._langfuse_initialized:
            try:
                self.langfuse_client.flush()
                self.logger.info("Langfuse client shutdown complete")
            except Exception as e:
                self.logger.warning(f"Error during Langfuse shutdown: {e}")
        
        # Shutdown OpenTelemetry
        if self._otel_initialized:
            try:
                self.otel_client.shutdown()
                self.logger.info("OpenTelemetry client shutdown complete")
            except Exception as e:
                self.logger.warning(f"Error during OpenTelemetry shutdown: {e}")


# Global singleton instance
_observability_manager: Optional[ObservabilityManager] = None


def initialize_observability(config: ObservabilityConfig) -> ObservabilityManager:
    """
    Initialize the global ObservabilityManager instance.
    
    Args:
        config: ObservabilityConfig with all monitoring settings
        
    Returns:
        Initialized ObservabilityManager
    """
    global _observability_manager
    
    if _observability_manager is None:
        _observability_manager = ObservabilityManager(config)
    
    return _observability_manager


def get_observability_manager() -> Optional[ObservabilityManager]:
    """
    Get the global ObservabilityManager instance.
    
    Returns:
        ObservabilityManager instance or None if not initialized
    """
    return _observability_manager


def shutdown_observability() -> None:
    """Shutdown the global ObservabilityManager instance."""
    global _observability_manager
    
    if _observability_manager is not None:
        _observability_manager.shutdown()
        _observability_manager = None


# Convenience function for monitoring decorator
def monitor(name: str, **attributes) -> Callable:
    """
    Decorator to monitor function execution with the appropriate backend.
    
    This is a unified decorator that will use either Langfuse or OpenTelemetry
    depending on the context and configuration.
    
    Args:
        name: Name for the monitored operation
        **attributes: Additional attributes for the span
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_observability_manager()
            if manager is None:
                return func(*args, **kwargs)
                
            with manager.start_span(name, **attributes):
                return func(*args, **kwargs)
                
        return wrapper
    
    return decorator
