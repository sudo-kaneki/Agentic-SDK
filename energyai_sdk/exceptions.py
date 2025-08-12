# EnergyAI Agentic SDK - Custom Exceptions
# File: energyai_sdk/exceptions.py

"""
Custom exception classes for the EnergyAI SDK.

This module provides a comprehensive set of exception classes for handling
various error conditions that can occur in the EnergyAI SDK.
"""

from typing import Any, Optional

# ==============================================================================
# BASE EXCEPTION CLASSES
# ==============================================================================


class EnergyAISDKError(Exception):
    """Base exception class for all EnergyAI SDK errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """String representation of the exception."""
        if self.error_code == self.__class__.__name__:
            return self.message
        return f"[{self.error_code}] {self.message}"


class EnergyAIConfigurationError(EnergyAISDKError):
    """Exception raised for configuration-related errors."""

    pass


class EnergyAIValidationError(EnergyAISDKError):
    """Exception raised for validation errors."""

    pass


# ==============================================================================
# AGENT-RELATED EXCEPTIONS
# ==============================================================================


class AgentError(EnergyAISDKError):
    """Base exception for agent-related errors."""

    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id
        if agent_id:
            self.details["agent_id"] = agent_id


class AgentNotFoundError(AgentError):
    """Exception raised when a requested agent is not found."""

    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent '{agent_id}' not found in registry",
            agent_id=agent_id,
            error_code="AGENT_NOT_FOUND",
        )


class AgentInitializationError(AgentError):
    """Exception raised when agent initialization fails."""

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        initialization_step: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message, agent_id=agent_id, error_code="AGENT_INITIALIZATION_FAILED", cause=cause
        )
        if initialization_step:
            self.details["initialization_step"] = initialization_step


class AgentExecutionError(AgentError):
    """Exception raised when agent execution fails."""

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        request_id: Optional[str] = None,
        execution_phase: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message, agent_id=agent_id, error_code="AGENT_EXECUTION_FAILED", cause=cause
        )
        if request_id:
            self.details["request_id"] = request_id
        if execution_phase:
            self.details["execution_phase"] = execution_phase


class AgentTimeoutError(AgentError):
    """Exception raised when agent operation times out."""

    def __init__(
        self, message: str, agent_id: Optional[str] = None, timeout_seconds: Optional[float] = None
    ):
        super().__init__(message, agent_id=agent_id, error_code="AGENT_TIMEOUT")
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class AgentModelError(AgentError):
    """Exception raised for model-related errors in agents."""

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        model_id: Optional[str] = None,
        model_error: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, agent_id=agent_id, error_code="AGENT_MODEL_ERROR", cause=cause)
        if model_id:
            self.details["model_id"] = model_id
        if model_error:
            self.details["model_error"] = model_error


# ==============================================================================
# REGISTRY EXCEPTIONS
# ==============================================================================


class RegistryError(EnergyAISDKError):
    """Base exception for registry-related errors."""

    pass


class ComponentNotFoundError(RegistryError):
    """Exception raised when a registry component is not found."""

    def __init__(self, component_type: str, component_name: str):
        super().__init__(
            f"{component_type.title()} '{component_name}' not found in registry",
            error_code="COMPONENT_NOT_FOUND",
        )
        self.details.update({"component_type": component_type, "component_name": component_name})


class ComponentRegistrationError(RegistryError):
    """Exception raised when component registration fails."""

    def __init__(
        self,
        component_type: str,
        component_name: str,
        reason: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"Failed to register {component_type} '{component_name}': {reason}",
            error_code="COMPONENT_REGISTRATION_FAILED",
            cause=cause,
        )
        self.details.update(
            {"component_type": component_type, "component_name": component_name, "reason": reason}
        )


class DuplicateComponentError(RegistryError):
    """Exception raised when trying to register a component with an existing name."""

    def __init__(self, component_type: str, component_name: str):
        super().__init__(
            f"{component_type.title()} '{component_name}' is already registered",
            error_code="DUPLICATE_COMPONENT",
        )
        self.details.update({"component_type": component_type, "component_name": component_name})


# ==============================================================================
# TOOL AND SKILL EXCEPTIONS
# ==============================================================================


class ToolError(EnergyAISDKError):
    """Base exception for tool-related errors."""

    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        if tool_name:
            self.details["tool_name"] = tool_name


class ToolExecutionError(ToolError):
    """Exception raised when tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message, tool_name=tool_name, error_code="TOOL_EXECUTION_FAILED", cause=cause
        )
        if parameters:
            self.details["parameters"] = parameters


class ToolParameterError(ToolError):
    """Exception raised for tool parameter validation errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        parameter_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        received_value: Optional[Any] = None,
    ):
        super().__init__(message, tool_name=tool_name, error_code="TOOL_PARAMETER_ERROR")
        if parameter_name:
            self.details["parameter_name"] = parameter_name
        if expected_type:
            self.details["expected_type"] = expected_type
        if received_value is not None:
            self.details["received_value"] = str(received_value)


class SkillError(EnergyAISDKError):
    """Base exception for skill-related errors."""

    def __init__(self, message: str, skill_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.skill_name = skill_name
        if skill_name:
            self.details["skill_name"] = skill_name


class SkillLoadError(SkillError):
    """Exception raised when skill loading fails."""

    def __init__(
        self, message: str, skill_name: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(
            message, skill_name=skill_name, error_code="SKILL_LOAD_FAILED", cause=cause
        )


# ==============================================================================
# MIDDLEWARE EXCEPTIONS
# ==============================================================================


class MiddlewareError(EnergyAISDKError):
    """Base exception for middleware-related errors."""

    def __init__(self, message: str, middleware_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.middleware_name = middleware_name
        if middleware_name:
            self.details["middleware_name"] = middleware_name


class MiddlewareExecutionError(MiddlewareError):
    """Exception raised when middleware execution fails."""

    def __init__(
        self,
        message: str,
        middleware_name: Optional[str] = None,
        phase: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            middleware_name=middleware_name,
            error_code="MIDDLEWARE_EXECUTION_FAILED",
            cause=cause,
        )
        if phase:
            self.details["phase"] = phase


class AuthenticationError(MiddlewareError):
    """Exception raised for authentication failures."""

    def __init__(self, message: str = "Authentication failed", auth_method: Optional[str] = None):
        super().__init__(message, error_code="AUTHENTICATION_FAILED")
        if auth_method:
            self.details["auth_method"] = auth_method


class AuthorizationError(MiddlewareError):
    """Exception raised for authorization failures."""

    def __init__(
        self,
        message: str = "Authorization failed",
        user_role: Optional[str] = None,
        required_permission: Optional[str] = None,
    ):
        super().__init__(message, error_code="AUTHORIZATION_FAILED")
        if user_role:
            self.details["user_role"] = user_role
        if required_permission:
            self.details["required_permission"] = required_permission


class RateLimitExceededError(MiddlewareError):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: Optional[str] = None,
        limit_value: Optional[int] = None,
        retry_after_seconds: Optional[int] = None,
    ):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        if limit_type:
            self.details["limit_type"] = limit_type
        if limit_value:
            self.details["limit_value"] = limit_value
        if retry_after_seconds:
            self.details["retry_after_seconds"] = retry_after_seconds


# ==============================================================================
# APPLICATION EXCEPTIONS
# ==============================================================================


class ApplicationError(EnergyAISDKError):
    """Base exception for application-level errors."""

    pass


class ApplicationStartupError(ApplicationError):
    """Exception raised during application startup."""

    def __init__(
        self, message: str, component: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message, error_code="APPLICATION_STARTUP_FAILED", cause=cause)
        if component:
            self.details["component"] = component


class ApplicationShutdownError(ApplicationError):
    """Exception raised during application shutdown."""

    def __init__(
        self, message: str, component: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message, error_code="APPLICATION_SHUTDOWN_FAILED", cause=cause)
        if component:
            self.details["component"] = component


# ==============================================================================
# TELEMETRY EXCEPTIONS
# ==============================================================================


class TelemetryError(EnergyAISDKError):
    """Base exception for telemetry-related errors."""

    pass


class TelemetryConfigurationError(TelemetryError):
    """Exception raised for telemetry configuration errors."""

    def __init__(
        self, message: str, provider: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message, error_code="TELEMETRY_CONFIGURATION_FAILED", cause=cause)
        if provider:
            self.details["provider"] = provider


class TelemetryTraceError(TelemetryError):
    """Exception raised when telemetry tracing fails."""

    def __init__(
        self, message: str, operation: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message, error_code="TELEMETRY_TRACE_FAILED", cause=cause)
        if operation:
            self.details["operation"] = operation


# ==============================================================================
# MODEL AND API EXCEPTIONS
# ==============================================================================


class ModelError(EnergyAISDKError):
    """Base exception for model-related errors."""

    def __init__(
        self, message: str, model_id: Optional[str] = None, provider: Optional[str] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.model_id = model_id
        self.provider = provider
        if model_id:
            self.details["model_id"] = model_id
        if provider:
            self.details["provider"] = provider


class ModelConfigurationError(ModelError):
    """Exception raised for model configuration errors."""

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        configuration_key: Optional[str] = None,
    ):
        super().__init__(
            message, model_id=model_id, provider=provider, error_code="MODEL_CONFIGURATION_ERROR"
        )
        if configuration_key:
            self.details["configuration_key"] = configuration_key


class APIError(EnergyAISDKError):
    """Exception raised for external API errors."""

    def __init__(
        self,
        message: str,
        api_endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code="API_ERROR", cause=cause)
        if api_endpoint:
            self.details["api_endpoint"] = api_endpoint
        if status_code:
            self.details["status_code"] = status_code
        if response_body:
            self.details["response_body"] = response_body


class APITimeoutError(APIError):
    """Exception raised when API calls timeout."""

    def __init__(
        self,
        message: str = "API request timed out",
        api_endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        super().__init__(message, api_endpoint=api_endpoint, error_code="API_TIMEOUT")
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class APIRateLimitError(APIError):
    """Exception raised when API rate limits are exceeded."""

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        api_endpoint: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
    ):
        super().__init__(message, api_endpoint=api_endpoint, error_code="API_RATE_LIMIT_EXCEEDED")
        if retry_after_seconds:
            self.details["retry_after_seconds"] = retry_after_seconds


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def format_exception_chain(exception: Exception) -> list[dict[str, Any]]:
    """Format exception chain for logging or debugging."""
    chain = []
    current = exception

    while current:
        if isinstance(current, EnergyAISDKError):
            chain.append(current.to_dict())
        else:
            chain.append({"error_type": current.__class__.__name__, "message": str(current)})

        current = getattr(current, "cause", None) or current.__cause__

    return chain


def create_error_response(exception: Exception) -> dict[str, Any]:
    """Create a standardized error response from an exception."""
    if isinstance(exception, EnergyAISDKError):
        return {
            "success": False,
            "error": exception.to_dict(),
            "exception_chain": format_exception_chain(exception),
        }
    else:
        return {
            "success": False,
            "error": {"error_type": exception.__class__.__name__, "message": str(exception)},
        }


def is_retryable_error(exception: Exception) -> bool:
    """Determine if an error is retryable."""
    retryable_errors = {APITimeoutError, APIRateLimitError, AgentTimeoutError, TelemetryTraceError}

    return isinstance(exception, tuple(retryable_errors))


# ==============================================================================
# ERROR CONTEXT MANAGER
# ==============================================================================


class ErrorContext:
    """Context manager for enhanced error handling."""

    def __init__(
        self,
        operation: str,
        component: Optional[str] = None,
        additional_context: Optional[dict[str, Any]] = None,
    ):
        self.operation = operation
        self.component = component
        self.additional_context = additional_context or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val and isinstance(exc_val, EnergyAISDKError):
            # Enhance the exception with context
            exc_val.details.update({"operation": self.operation, **self.additional_context})
            if self.component:
                exc_val.details["component"] = self.component

        return False  # Don't suppress the exception


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Example of custom exception usage
    try:
        raise AgentNotFoundError("TestAgent")
    except EnergyAISDKError as e:
        print("Exception details:")
        print(e.to_dict())
        print("\nException chain:")
        for item in format_exception_chain(e):
            print(f"  - {item}")

    # Example of error context
    try:
        with ErrorContext("agent_initialization", "SemanticKernelAgent"):
            raise AgentInitializationError("Failed to initialize kernel")
    except EnergyAISDKError as e:
        print("\nWith context:")
        print(e.to_dict())
