# Middleware framework

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

# Import from our core foundation
from .core import AgentRequest, AgentResponse, monitor, telemetry_manager

# Middleware context and pipeline


@dataclass
class MiddlewareContext:
    """Context object passed through middleware pipeline."""

    request: AgentRequest
    response: Optional[AgentResponse] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_phases: list[dict[str, Any]] = field(default_factory=list)
    telemetry_data: dict[str, Any] = field(default_factory=dict)
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)

    def add_execution_phase(self, phase_name: str, data: dict[str, Any]):
        """Add execution phase information."""
        self.execution_phases.append(
            {"phase": phase_name, "timestamp": datetime.now(timezone.utc).isoformat(), "data": data}
        )

    def get_execution_time_ms(self) -> int:
        """Get total execution time in milliseconds."""
        return int((datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000)

    def set_error(self, error: Exception, phase: str = "unknown"):
        """Set error with context."""
        self.error = error
        self.add_execution_phase(
            f"error_{phase}", {"error_type": type(error).__name__, "error_message": str(error)}
        )


class AgentMiddleware(ABC):
    """Base class for agent middleware components."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.enabled = True
        self.priority = 0  # Lower numbers execute first

    @abstractmethod
    async def process(
        self, context: MiddlewareContext, next_middleware: Callable[[MiddlewareContext], Any]
    ) -> Any:
        """Process middleware with context."""
        pass

    def is_applicable(self, context: MiddlewareContext) -> bool:
        """Check if middleware should be applied to this context."""
        return self.enabled


class MiddlewarePipeline:
    """Manages execution of middleware components in phases."""

    def __init__(self):
        self.preprocessing_middleware: list[AgentMiddleware] = []
        self.processing_middleware: list[AgentMiddleware] = []
        self.postprocessing_middleware: list[AgentMiddleware] = []
        self.error_handling_middleware: list[AgentMiddleware] = []
        self.global_middleware: list[AgentMiddleware] = []

    def add_preprocessing(self, middleware: AgentMiddleware) -> "MiddlewarePipeline":
        """Add preprocessing middleware."""
        self.preprocessing_middleware.append(middleware)
        self._sort_middleware(self.preprocessing_middleware)
        return self

    def add_processing(self, middleware: AgentMiddleware) -> "MiddlewarePipeline":
        """Add core processing middleware."""
        self.processing_middleware.append(middleware)
        self._sort_middleware(self.processing_middleware)
        return self

    def add_postprocessing(self, middleware: AgentMiddleware) -> "MiddlewarePipeline":
        """Add postprocessing middleware."""
        self.postprocessing_middleware.append(middleware)
        self._sort_middleware(self.postprocessing_middleware)
        return self

    def add_error_handling(self, middleware: AgentMiddleware) -> "MiddlewarePipeline":
        """Add error handling middleware."""
        self.error_handling_middleware.append(middleware)
        self._sort_middleware(self.error_handling_middleware)
        return self

    def add_global(self, middleware: AgentMiddleware) -> "MiddlewarePipeline":
        """Add global middleware that runs for all phases."""
        self.global_middleware.append(middleware)
        self._sort_middleware(self.global_middleware)
        return self

    def _sort_middleware(self, middleware_list: list[AgentMiddleware]):
        """Sort middleware by priority."""
        middleware_list.sort(key=lambda m: m.priority)

    @monitor("middleware.pipeline.execute")
    async def execute(self, context: MiddlewareContext) -> MiddlewareContext:
        """Execute complete middleware pipeline."""
        try:
            # Execute global middleware first
            if self.global_middleware:
                await self._execute_phase("global", self.global_middleware, context)

            if not context.error:
                # Execute preprocessing
                await self._execute_phase("preprocessing", self.preprocessing_middleware, context)

            if not context.error:
                # Execute core processing
                await self._execute_phase("processing", self.processing_middleware, context)

            if not context.error:
                # Execute postprocessing
                await self._execute_phase("postprocessing", self.postprocessing_middleware, context)

        except Exception as e:
            context.set_error(e, "pipeline")

        finally:
            # Always execute error handling if there's an error
            if context.error and self.error_handling_middleware:
                try:
                    await self._execute_phase(
                        "error_handling", self.error_handling_middleware, context
                    )
                except Exception as e:
                    # Log error handling failures but don't propagate
                    context.add_execution_phase(
                        "error_handling_failed",
                        {"error_type": type(e).__name__, "error_message": str(e)},
                    )

        return context

    async def _execute_phase(
        self, phase_name: str, middleware_list: list[AgentMiddleware], context: MiddlewareContext
    ):
        """Execute a specific phase of middleware."""
        applicable_middleware = [m for m in middleware_list if m.is_applicable(context)]
        context.add_execution_phase(phase_name, {"middleware_count": len(applicable_middleware)})

        async def execute_at_index(index: int):
            if index >= len(applicable_middleware) or context.error:
                return

            middleware = applicable_middleware[index]

            async def next_middleware(ctx: MiddlewareContext):
                await execute_at_index(index + 1)

            try:
                await middleware.process(context, next_middleware)
            except Exception as e:
                context.set_error(e, f"{phase_name}_{middleware.name}")

        await execute_at_index(0)


# ==============================================================================
# AUTHENTICATION AND AUTHORIZATION MIDDLEWARE
# ==============================================================================


class AuthenticationMiddleware(AgentMiddleware):
    """Middleware for handling authentication and authorization."""

    def __init__(
        self,
        required_auth: bool = True,
        api_keys: Optional[set[str]] = None,
        bearer_tokens: Optional[set[str]] = None,
        auth_header: str = "Authorization",
        api_key_header: str = "X-API-Key",
    ):
        super().__init__("Authentication")
        self.required_auth = required_auth
        self.api_keys = api_keys or set()
        self.bearer_tokens = bearer_tokens or set()
        self.auth_header = auth_header
        self.api_key_header = api_key_header
        self.priority = 10  # High priority - run early

    async def process(self, context: MiddlewareContext, next_middleware: Callable):
        """Authenticate request."""
        if not self.required_auth:
            await next_middleware(context)
            return

        # Check for API key in request metadata
        api_key = context.request.metadata.get("api_key") or context.request.metadata.get(
            self.api_key_header
        )

        # Check for bearer token
        auth_header = context.request.metadata.get(self.auth_header, "")
        bearer_token = None
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:]

        # Validate credentials
        is_authenticated = False
        auth_method = None

        if api_key and (not self.api_keys or api_key in self.api_keys):
            is_authenticated = True
            auth_method = "api_key"
        elif bearer_token and (not self.bearer_tokens or bearer_token in self.bearer_tokens):
            is_authenticated = True
            auth_method = "bearer_token"
        elif not self.api_keys and not self.bearer_tokens:
            # No specific credentials required
            is_authenticated = True
            auth_method = "no_auth_required"

        if not is_authenticated:
            context.set_error(
                ValueError("Authentication failed: Invalid or missing credentials"),
                "authentication",
            )
            return

        context.metadata.update({"authenticated": True, "auth_method": auth_method})

        await next_middleware(context)


class AuthorizationMiddleware(AgentMiddleware):
    """Middleware for role-based authorization."""

    def __init__(
        self,
        role_permissions: Optional[dict[str, list[str]]] = None,
        agent_permissions: Optional[dict[str, list[str]]] = None,
        default_role: str = "user",
    ):
        super().__init__("Authorization")
        self.role_permissions = role_permissions or {
            "admin": ["*"],
            "user": ["chat", "query"],
            "readonly": ["query"],
        }
        self.agent_permissions = agent_permissions or {}
        self.default_role = default_role
        self.priority = 15  # After authentication

    async def process(self, context: MiddlewareContext, next_middleware: Callable):
        """Authorize request based on user role and agent permissions."""
        # Get user role from metadata
        user_role = context.request.metadata.get("user_role", self.default_role)
        agent_id = context.request.agent_id

        # Check role permissions
        role_perms = self.role_permissions.get(user_role, [])
        agent_perms = self.agent_permissions.get(agent_id, ["*"])

        # Admin has access to everything
        if "*" in role_perms:
            context.metadata["authorized"] = True
            await next_middleware(context)
            return

        # Check if user role has permission for this operation
        operation = context.request.metadata.get("operation", "chat")

        has_role_permission = operation in role_perms or "*" in role_perms
        has_agent_permission = "*" in agent_perms or user_role in agent_perms

        if not (has_role_permission and has_agent_permission):
            context.set_error(
                ValueError(
                    f"Authorization failed: Role '{user_role}' not authorized for operation '{operation}' on agent '{agent_id}'"
                ),
                "authorization",
            )
            return

        context.metadata["authorized"] = True
        await next_middleware(context)


# ==============================================================================
# RATE LIMITING MIDDLEWARE
# ==============================================================================


class RateLimitingMiddleware(AgentMiddleware):
    """Middleware for rate limiting requests."""

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_requests_per_hour: int = 1000,
        max_requests_per_day: int = 10000,
        burst_limit: int = 10,
        key_function: Optional[Callable[[AgentRequest], str]] = None,
    ):
        super().__init__("RateLimiting")
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.max_requests_per_day = max_requests_per_day
        self.burst_limit = burst_limit
        self.key_function = key_function or (lambda req: req.user_id or "anonymous")

        # Request tracking
        self.request_times: dict[str, list[datetime]] = defaultdict(list)
        self.burst_tracking: dict[str, list[datetime]] = defaultdict(list)

        self.priority = 20  # After auth

    def _clean_old_requests(self, request_list: list[datetime], max_age_seconds: int):
        """Remove requests older than max_age_seconds."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        return [req_time for req_time in request_list if req_time > cutoff_time]

    def _check_rate_limit(
        self, user_key: str, request_list: list[datetime], max_requests: int, window_seconds: int
    ) -> bool:
        """Check if rate limit is exceeded."""
        # Clean old requests
        cleaned_requests = self._clean_old_requests(request_list, window_seconds)
        self.request_times[user_key] = cleaned_requests

        return len(cleaned_requests) < max_requests

    async def process(self, context: MiddlewareContext, next_middleware: Callable):
        """Apply rate limiting."""
        user_key = self.key_function(context.request)
        current_time = datetime.now(timezone.utc)

        # Check burst limit (last 10 seconds)
        burst_list = self.burst_tracking[user_key]
        if not self._check_rate_limit(user_key, burst_list, self.burst_limit, 10):
            context.set_error(
                ValueError("Rate limit exceeded: Too many requests in burst window"),
                "rate_limit_burst",
            )
            return

        # Check minute limit
        minute_list = self.request_times.get(f"{user_key}_minute", [])
        if not self._check_rate_limit(
            f"{user_key}_minute", minute_list, self.max_requests_per_minute, 60
        ):
            context.set_error(
                ValueError("Rate limit exceeded: Too many requests per minute"), "rate_limit_minute"
            )
            return

        # Check hour limit
        hour_list = self.request_times.get(f"{user_key}_hour", [])
        if not self._check_rate_limit(
            f"{user_key}_hour", hour_list, self.max_requests_per_hour, 3600
        ):
            context.set_error(
                ValueError("Rate limit exceeded: Too many requests per hour"), "rate_limit_hour"
            )
            return

        # Check day limit
        day_list = self.request_times.get(f"{user_key}_day", [])
        if not self._check_rate_limit(
            f"{user_key}_day", day_list, self.max_requests_per_day, 86400
        ):
            context.set_error(
                ValueError("Rate limit exceeded: Too many requests per day"), "rate_limit_day"
            )
            return

        # Record this request
        self.burst_tracking[user_key].append(current_time)
        self.request_times[f"{user_key}_minute"].append(current_time)
        self.request_times[f"{user_key}_hour"].append(current_time)
        self.request_times[f"{user_key}_day"].append(current_time)

        context.metadata.update({"rate_limit_checked": True, "rate_limit_user_key": user_key})

        await next_middleware(context)


# ==============================================================================
# VALIDATION MIDDLEWARE
# ==============================================================================


class ValidationMiddleware(AgentMiddleware):
    """Middleware for comprehensive request validation."""

    def __init__(
        self,
        max_message_length: int = 10000,
        min_message_length: int = 1,
        blocked_patterns: Optional[list[str]] = None,
        required_fields: Optional[list[str]] = None,
        sanitize_input: bool = True,
    ):
        super().__init__("Validation")
        self.max_message_length = max_message_length
        self.min_message_length = min_message_length
        self.blocked_patterns = blocked_patterns or []
        self.required_fields = required_fields or ["message", "agent_id"]
        self.sanitize_input = sanitize_input
        self.priority = 30  # After auth and rate limiting

    def _sanitize_message(self, message: str) -> str:
        """Sanitize input message."""
        if not self.sanitize_input:
            return message

        # Basic sanitization
        sanitized = message.strip()

        # Remove potentially dangerous patterns
        dangerous_patterns = [
            "<script",
            "javascript:",
            "data:text/html",
            "vbscript:",
        ]

        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, "")

        return sanitized

    async def process(self, context: MiddlewareContext, next_middleware: Callable):
        """Validate request."""
        request = context.request

        # Check required fields
        for field in self.required_fields:
            if not hasattr(request, field) or not getattr(request, field):
                context.set_error(
                    ValueError(f"Required field missing or empty: {field}"),
                    "validation_required_field",
                )
                return

        # Validate message length
        message_length = len(request.message)
        if message_length > self.max_message_length:
            context.set_error(
                ValueError(
                    f"Message too long. Maximum length is {self.max_message_length}, got {message_length}"
                ),
                "validation_message_length",
            )
            return

        if message_length < self.min_message_length:
            context.set_error(
                ValueError(
                    f"Message too short. Minimum length is {self.min_message_length}, got {message_length}"
                ),
                "validation_message_length",
            )
            return

        # Check blocked patterns
        message_lower = request.message.lower()
        for pattern in self.blocked_patterns:
            if pattern.lower() in message_lower:
                context.set_error(
                    ValueError("Message contains blocked content"), "validation_blocked_pattern"
                )
                return

        # Sanitize input
        sanitized_message = self._sanitize_message(request.message)
        if sanitized_message != request.message:
            request.message = sanitized_message
            context.metadata["message_sanitized"] = True

        context.metadata.update(
            {
                "validated": True,
                "original_message_length": message_length,
                "sanitized": sanitized_message != request.message,
            }
        )

        await next_middleware(context)


# ==============================================================================
# CACHING MIDDLEWARE
# ==============================================================================


class CachingMiddleware(AgentMiddleware):
    """Middleware for response caching."""

    def __init__(
        self,
        cache_ttl_seconds: int = 300,
        max_cache_size: int = 1000,
        cache_key_function: Optional[Callable[[AgentRequest], str]] = None,
        cacheable_agents: Optional[set[str]] = None,
    ):
        super().__init__("Caching")
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self.cache_key_function = cache_key_function or self._default_cache_key
        self.cacheable_agents = cacheable_agents  # None means all agents cacheable

        # Simple in-memory cache
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_access_times: dict[str, datetime] = {}

        self.priority = 35  # After validation, before processing

    def _default_cache_key(self, request: AgentRequest) -> str:
        """Generate default cache key."""
        key_data = (
            f"{request.agent_id}:{request.message}:{json.dumps(request.metadata, sort_keys=True)}"
        )
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()

    def _is_cache_valid(self, cached_item: dict[str, Any]) -> bool:
        """Check if cached item is still valid."""
        cached_time = datetime.fromisoformat(cached_item["timestamp"])
        return (datetime.now(timezone.utc) - cached_time).seconds < self.cache_ttl_seconds

    def _cleanup_cache(self):
        """Remove expired and least recently used items."""
        datetime.now(timezone.utc)

        # Remove expired items
        expired_keys = []
        for key, item in self.cache.items():
            if not self._is_cache_valid(item):
                expired_keys.append(key)

        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_access_times.pop(key, None)

        # Remove LRU items if cache is too large
        if len(self.cache) > self.max_cache_size:
            # Sort by access time and remove oldest
            sorted_keys = sorted(
                self.cache_access_times.keys(), key=lambda k: self.cache_access_times[k]
            )

            keys_to_remove = sorted_keys[: len(self.cache) - self.max_cache_size]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.cache_access_times.pop(key, None)

    def is_applicable(self, context: MiddlewareContext) -> bool:
        """Check if caching should be applied."""
        if not super().is_applicable(context):
            return False

        # Only cache for specific agents if specified
        if self.cacheable_agents is not None:
            return context.request.agent_id in self.cacheable_agents

        return True

    async def process(self, context: MiddlewareContext, next_middleware: Callable):
        """Handle caching."""
        cache_key = self.cache_key_function(context.request)

        # Check cache
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            cached_response_data = self.cache[cache_key]["response"]

            # Recreate response object
            context.response = AgentResponse(**cached_response_data)
            context.metadata.update({"cache_hit": True, "cache_key": cache_key})

            # Update access time
            self.cache_access_times[cache_key] = datetime.now(timezone.utc)
            return

        # Execute next middleware
        await next_middleware(context)

        # Cache response if successful
        if context.response and not context.error:
            response_data = {
                "content": context.response.content,
                "agent_id": context.response.agent_id,
                "session_id": context.response.session_id,
                "execution_time_ms": context.response.execution_time_ms,
                "metadata": context.response.metadata,
                "error": context.response.error,
                "timestamp": context.response.timestamp,
            }

            self.cache[cache_key] = {
                "response": response_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.cache_access_times[cache_key] = datetime.now(timezone.utc)

            context.metadata.update({"cached": True, "cache_key": cache_key})

            # Cleanup cache periodically
            if len(self.cache) % 10 == 0:  # Every 10 cache operations
                self._cleanup_cache()


# ==============================================================================
# TELEMETRY MIDDLEWARE
# ==============================================================================


class TelemetryMiddleware(AgentMiddleware):
    """Middleware for comprehensive telemetry integration."""

    def __init__(
        self,
        trace_requests: bool = True,
        trace_responses: bool = True,
        include_metadata: bool = True,
        sample_rate: float = 1.0,
    ):
        super().__init__("Telemetry")
        self.trace_requests = trace_requests
        self.trace_responses = trace_responses
        self.include_metadata = include_metadata
        self.sample_rate = sample_rate
        self.priority = 1  # Very high priority - run early for full tracing

    def _should_trace(self) -> bool:
        """Determine if this request should be traced based on sample rate."""
        import random

        return random.random() <= self.sample_rate  # nosec B311

    async def process(self, context: MiddlewareContext, next_middleware: Callable):
        """Add telemetry tracing."""
        if not self._should_trace():
            await next_middleware(context)
            return

        trace_metadata = {
            "agent_id": context.request.agent_id,
            "user_id": context.request.user_id,
            "session_id": context.request.session_id,
            "request_id": context.request.request_id,
        }

        if self.include_metadata:
            trace_metadata.update(context.request.metadata)

        with telemetry_manager.trace_operation(
            f"middleware_request_{context.request.agent_id}", trace_metadata
        ) as trace_id:
            context.telemetry_data["trace_id"] = trace_id

            # Execute next middleware
            await next_middleware(context)

            # Add response data to telemetry
            if context.response:
                context.telemetry_data.update(
                    {
                        "response_length": len(context.response.content),
                        "execution_time_ms": context.response.execution_time_ms,
                        "success": context.response.error is None,
                    }
                )


# ==============================================================================
# ERROR HANDLING MIDDLEWARE
# ==============================================================================


class ErrorHandlingMiddleware(AgentMiddleware):
    """Middleware for comprehensive error handling."""

    def __init__(
        self,
        enable_detailed_errors: bool = True,
        log_errors: bool = True,
        error_response_template: Optional[str] = None,
    ):
        super().__init__("ErrorHandling")
        self.enable_detailed_errors = enable_detailed_errors
        self.log_errors = log_errors
        self.error_response_template = (
            error_response_template or "An error occurred: {error_message}"
        )
        self.priority = 1000  # Very low priority - run at the end

    async def process(self, context: MiddlewareContext, next_middleware: Callable):
        """Handle errors gracefully."""
        if not context.error:
            return

        error = context.error
        error_type = type(error).__name__

        if self.log_errors:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error in middleware pipeline: {error_type}: {error}")

        # Create error response
        if self.enable_detailed_errors:
            error_message = self.error_response_template.format(
                error_message=str(error), error_type=error_type
            )
        else:
            error_message = "An error occurred while processing your request."

        # Create error response
        context.response = AgentResponse(
            content=error_message,
            agent_id=context.request.agent_id,
            session_id=context.request.session_id,
            execution_time_ms=context.get_execution_time_ms(),
            error=str(error),
            metadata={
                **context.metadata,
                "error": True,
                "error_type": error_type,
                "error_handled": True,
                "execution_phases": context.execution_phases,
            },
        )

        # Clear error to prevent further propagation
        context.error = None


# ==============================================================================
# PIPELINE FACTORY AND CONVENIENCE FUNCTIONS
# ==============================================================================


def create_default_pipeline(
    enable_auth: bool = True,
    enable_rate_limiting: bool = True,
    enable_validation: bool = True,
    enable_caching: bool = True,
    enable_telemetry: bool = True,
    api_keys: Optional[set[str]] = None,
    max_requests_per_minute: int = 60,
) -> MiddlewarePipeline:
    """Create a default middleware pipeline with common middleware."""
    pipeline = MiddlewarePipeline()

    # Add telemetry first for full request tracing
    if enable_telemetry:
        pipeline.add_global(TelemetryMiddleware())

    # Preprocessing middleware
    if enable_auth:
        pipeline.add_preprocessing(
            AuthenticationMiddleware(required_auth=api_keys is not None, api_keys=api_keys)
        )
        pipeline.add_preprocessing(AuthorizationMiddleware())

    if enable_rate_limiting:
        pipeline.add_preprocessing(
            RateLimitingMiddleware(max_requests_per_minute=max_requests_per_minute)
        )

    if enable_validation:
        pipeline.add_preprocessing(ValidationMiddleware())

    if enable_caching:
        pipeline.add_preprocessing(CachingMiddleware())

    # Error handling
    pipeline.add_error_handling(ErrorHandlingMiddleware())

    return pipeline


def create_production_pipeline(
    api_keys: set[str],
    max_requests_per_minute: int = 100,
    max_requests_per_hour: int = 5000,
    cache_ttl_seconds: int = 600,
    enable_detailed_errors: bool = False,
) -> MiddlewarePipeline:
    """Create a production-ready middleware pipeline."""
    pipeline = MiddlewarePipeline()

    # Comprehensive telemetry
    pipeline.add_global(TelemetryMiddleware(sample_rate=1.0))

    # Authentication and authorization
    pipeline.add_preprocessing(AuthenticationMiddleware(required_auth=True, api_keys=api_keys))
    pipeline.add_preprocessing(AuthorizationMiddleware())

    # Aggressive rate limiting
    pipeline.add_preprocessing(
        RateLimitingMiddleware(
            max_requests_per_minute=max_requests_per_minute,
            max_requests_per_hour=max_requests_per_hour,
            burst_limit=5,
        )
    )

    # Strict validation
    pipeline.add_preprocessing(
        ValidationMiddleware(
            max_message_length=5000,
            blocked_patterns=["<script", "javascript:", "eval("],
            sanitize_input=True,
        )
    )

    # Long-term caching
    pipeline.add_preprocessing(
        CachingMiddleware(cache_ttl_seconds=cache_ttl_seconds, max_cache_size=5000)
    )

    # Production error handling
    pipeline.add_error_handling(
        ErrorHandlingMiddleware(enable_detailed_errors=enable_detailed_errors, log_errors=True)
    )

    return pipeline


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    import asyncio

    async def example_usage():
        # Create a middleware pipeline
        pipeline = create_default_pipeline(
            enable_auth=False,  # Disable for testing
            enable_rate_limiting=False,  # Disable for testing
            max_requests_per_minute=100,
        )

        # Create a test request
        request = AgentRequest(
            message="What is the LCOE for a 100MW solar farm?",
            agent_id="EnergyAnalyst",
            user_id="test_user",
            session_id="test_session",
        )

        # Create context
        context = MiddlewareContext(request=request)

        # Execute pipeline
        result_context = await pipeline.execute(context)

        print("Pipeline execution completed")
        print(f"Error: {result_context.error}")
        print(f"Execution phases: {len(result_context.execution_phases)}")
        print(f"Metadata: {result_context.metadata}")

        for phase in result_context.execution_phases:
            print(f"Phase: {phase['phase']} at {phase['timestamp']}")

    asyncio.run(example_usage())
