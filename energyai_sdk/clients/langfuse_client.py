"""
Langfuse Monitoring Client for EnergyAI SDK.

This client provides comprehensive observability for agent interactions using Langfuse,
including trace creation, generation tracking, and performance monitoring.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    from langfuse import Langfuse
    from langfuse.model import CreateTrace, CreateGeneration, CreateSpan
    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    CreateTrace = None
    CreateGeneration = None
    CreateSpan = None
    LANGFUSE_AVAILABLE = False


class LangfuseMonitoringClient:
    """
    Langfuse client for monitoring agent interactions and LLM calls.
    
    Provides comprehensive tracing capabilities including:
    - Session-level traces for multi-turn conversations
    - Generation tracking for LLM calls  
    - Agent performance metrics
    - Error tracking and debugging
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        debug: bool = False,
        environment: str = "production"
    ):
        """
        Initialize the Langfuse monitoring client.
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key  
            host: Langfuse API host
            debug: Enable debug logging
            environment: Environment name (production, staging, development)
        """
        self.logger = logging.getLogger(__name__)
        self.environment = environment
        self.debug = debug
        
        if not LANGFUSE_AVAILABLE:
            self.logger.warning("Langfuse not available. Install with: pip install langfuse")
            self.client = None
            return
            
        if not public_key or not secret_key:
            self.logger.warning("Langfuse credentials not provided. Monitoring disabled.")
            self.client = None
            return
            
        try:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=debug
            )
            self.logger.info(f"Langfuse client initialized for environment: {environment}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Langfuse client: {e}")
            self.client = None

    def is_enabled(self) -> bool:
        """Check if Langfuse monitoring is enabled and available."""
        return self.client is not None

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        Create a new Langfuse trace for agent interaction.
        
        Args:
            name: Trace name (e.g., "agent-run:EnergyConsultant")
            user_id: User identifier
            session_id: Session identifier for conversation tracking
            input_data: Input data for the trace
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            Langfuse trace object or None if disabled
        """
        if not self.is_enabled():
            return None
            
        try:
            # Prepare trace data
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
                
            # Combine metadata
            combined_metadata = {
                "environment": self.environment,
                "sdk_version": "1.0.0",
                "trace_id": str(uuid.uuid4()),
            }
            if metadata:
                combined_metadata.update(metadata)
            trace_data["metadata"] = combined_metadata
            
            if tags:
                trace_data["tags"] = tags
                
            trace = self.client.trace(**trace_data)
            
            if self.debug:
                self.logger.debug(f"Created trace: {name} for user: {user_id}, session: {session_id}")
                
            return trace
            
        except Exception as e:
            self.logger.error(f"Failed to create trace: {e}")
            return None

    def create_generation(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Create a generation within a trace for LLM calls.
        
        Args:
            trace: Parent trace object
            name: Generation name (e.g., "planner-invocation")
            input_data: Input to the LLM
            model: Model name
            model_parameters: Model parameters (temperature, max_tokens, etc.)
            metadata: Additional metadata
            
        Returns:
            Langfuse generation object or None if disabled
        """
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
                
            # Combine metadata
            combined_metadata = {
                "generation_id": str(uuid.uuid4()),
            }
            if metadata:
                combined_metadata.update(metadata)
            generation_data["metadata"] = combined_metadata
            
            generation = trace.generation(**generation_data)
            
            if self.debug:
                self.logger.debug(f"Created generation: {name} with model: {model}")
                
            return generation
            
        except Exception as e:
            self.logger.error(f"Failed to create generation: {e}")
            return None

    def end_generation(
        self,
        generation,
        output: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        End a generation with output and usage information.
        
        Args:
            generation: Generation object to end
            output: Generation output
            usage: Token usage information
            level: Log level (DEFAULT, WARNING, ERROR)
            status_message: Status message for errors
            end_time: End timestamp
        """
        if not self.is_enabled() or not generation:
            return
            
        try:
            end_data = {
                "end_time": end_time or datetime.now(timezone.utc),
                "level": level
            }
            
            if output is not None:
                end_data["output"] = output
            if usage:
                end_data["usage"] = usage
            if status_message:
                end_data["status_message"] = status_message
                
            generation.end(**end_data)
            
            if self.debug:
                self.logger.debug(f"Ended generation with level: {level}")
                
        except Exception as e:
            self.logger.error(f"Failed to end generation: {e}")

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Create a span within a trace for operation tracking.
        
        Args:
            trace: Parent trace object
            name: Span name
            input_data: Input data
            metadata: Additional metadata
            
        Returns:
            Langfuse span object or None if disabled
        """
        if not self.is_enabled() or not trace:
            return None
            
        try:
            span_data = {
                "name": name,
                "start_time": datetime.now(timezone.utc),
            }
            
            if input_data:
                span_data["input"] = input_data
                
            # Combine metadata
            combined_metadata = {
                "span_id": str(uuid.uuid4()),
            }
            if metadata:
                combined_metadata.update(metadata)
            span_data["metadata"] = combined_metadata
            
            span = trace.span(**span_data)
            
            if self.debug:
                self.logger.debug(f"Created span: {name}")
                
            return span
            
        except Exception as e:
            self.logger.error(f"Failed to create span: {e}")
            return None

    def end_span(
        self,
        span,
        output: Optional[Any] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        End a span with output information.
        
        Args:
            span: Span object to end
            output: Span output
            level: Log level (DEFAULT, WARNING, ERROR)
            status_message: Status message for errors
            end_time: End timestamp
        """
        if not self.is_enabled() or not span:
            return
            
        try:
            end_data = {
                "end_time": end_time or datetime.now(timezone.utc),
                "level": level
            }
            
            if output is not None:
                end_data["output"] = output
            if status_message:
                end_data["status_message"] = status_message
                
            span.end(**end_data)
            
            if self.debug:
                self.logger.debug(f"Ended span with level: {level}")
                
        except Exception as e:
            self.logger.error(f"Failed to end span: {e}")

    def update_trace(
        self,
        trace,
        output: Optional[Any] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a trace with final information.
        
        Args:
            trace: Trace object to update
            output: Final output
            level: Log level (DEFAULT, WARNING, ERROR)
            status_message: Status message
            metadata: Additional metadata to merge
        """
        if not self.is_enabled() or not trace:
            return
            
        try:
            update_data = {
                "level": level
            }
            
            if output is not None:
                update_data["output"] = output
            if status_message:
                update_data["status_message"] = status_message
            if metadata:
                update_data["metadata"] = metadata
                
            trace.update(**update_data)
            
            if self.debug:
                self.logger.debug(f"Updated trace with level: {level}")
                
        except Exception as e:
            self.logger.error(f"Failed to update trace: {e}")

    def flush(self) -> None:
        """
        Flush all pending telemetry data to Langfuse.
        """
        if not self.is_enabled():
            return
            
        try:
            self.client.flush()
            if self.debug:
                self.logger.debug("Flushed Langfuse telemetry data")
                
        except Exception as e:
            self.logger.error(f"Failed to flush Langfuse data: {e}")

    def health_check(self) -> bool:
        """
        Check if the Langfuse client is healthy and can connect.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.is_enabled():
            return False
            
        try:
            # Simple health check - create and immediately flush a test trace
            test_trace = self.create_trace(
                name="health-check",
                metadata={"test": True, "timestamp": time.time()}
            )
            if test_trace:
                self.update_trace(test_trace, output="health-check-success")
                self.flush()
                return True
            return False
            
        except Exception as e:
            self.logger.warning(f"Langfuse health check failed: {e}")
            return False


# Singleton instance for easy access
_langfuse_client: Optional[LangfuseMonitoringClient] = None


def get_langfuse_client(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: str = "https://cloud.langfuse.com",
    debug: bool = False,
    environment: str = "production"
) -> LangfuseMonitoringClient:
    """
    Get or create the singleton Langfuse monitoring client.
    
    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse API host
        debug: Enable debug logging
        environment: Environment name
        
    Returns:
        LangfuseMonitoringClient instance
    """
    global _langfuse_client
    
    if _langfuse_client is None:
        _langfuse_client = LangfuseMonitoringClient(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            debug=debug,
            environment=environment
        )
    
    return _langfuse_client


def configure_langfuse(
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com",
    debug: bool = False,
    environment: str = "production"
) -> LangfuseMonitoringClient:
    """
    Configure the global Langfuse monitoring client.
    
    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse API host
        debug: Enable debug logging
        environment: Environment name
        
    Returns:
        Configured LangfuseMonitoringClient
    """
    global _langfuse_client
    
    _langfuse_client = LangfuseMonitoringClient(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        debug=debug,
        environment=environment
    )
    
    return _langfuse_client
