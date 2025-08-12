# tests/test_middleware.py
"""
Test middleware pipeline system and individual middleware components.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from energyai_sdk import AgentRequest, AgentResponse
from energyai_sdk.middleware import (
    AgentMiddleware,
    AuthenticationMiddleware,
    CachingMiddleware,
    ErrorHandlingMiddleware,
    MiddlewareContext,
    MiddlewarePipeline,
    RateLimitingMiddleware,
    ValidationMiddleware,
    create_default_pipeline,
    create_production_pipeline,
)


class TestMiddlewareContext:
    """Test MiddlewareContext functionality."""

    def test_context_initialization(self, sample_agent_request):
        """Test middleware context initialization."""

        context = MiddlewareContext(request=sample_agent_request)

        assert context.request == sample_agent_request
        assert context.response is None
        assert context.error is None
        assert isinstance(context.metadata, dict)
        assert isinstance(context.execution_phases, list)
        assert isinstance(context.telemetry_data, dict)
        assert context.start_time is not None

    def test_add_execution_phase(self, sample_middleware_context):
        """Test adding execution phases."""

        context = sample_middleware_context

        context.add_execution_phase("test_phase", {"key": "value"})

        assert len(context.execution_phases) == 1
        phase = context.execution_phases[0]
        assert phase["phase"] == "test_phase"
        assert phase["data"]["key"] == "value"
        assert "timestamp" in phase

    def test_get_execution_time_ms(self, sample_middleware_context):
        """Test execution time calculation."""

        context = sample_middleware_context

        # Small delay to measure
        time.sleep(0.01)

        execution_time = context.get_execution_time_ms()
        assert execution_time > 0
        assert execution_time < 1000  # Should be less than 1 second

    def test_set_error(self, sample_middleware_context):
        """Test error setting with context."""

        context = sample_middleware_context
        error = ValueError("Test error")

        context.set_error(error, "test_phase")

        assert context.error == error
        assert len(context.execution_phases) == 1
        assert context.execution_phases[0]["phase"] == "error_test_phase"
        assert context.execution_phases[0]["data"]["error_type"] == "ValueError"


class TestAgentMiddleware:
    """Test AgentMiddleware base class."""

    def test_middleware_abstract_methods(self):
        """Test that AgentMiddleware process method must be implemented."""

        class IncompleteMiddleware(AgentMiddleware):
            pass

        middleware = IncompleteMiddleware("TestMiddleware")

        # Should raise TypeError when calling process
        with pytest.raises(TypeError):
            asyncio.run(middleware.process(None, None))

    def test_middleware_is_applicable(self, sample_middleware_context):
        """Test is_applicable method."""

        class TestMiddleware(AgentMiddleware):
            async def process(self, context, next_middleware):
                pass

        middleware = TestMiddleware("TestMiddleware")

        # Should be applicable by default when enabled
        assert middleware.is_applicable(sample_middleware_context) is True

        # Should not be applicable when disabled
        middleware.enabled = False
        assert middleware.is_applicable(sample_middleware_context) is False


class TestMiddlewarePipeline:
    """Test MiddlewarePipeline functionality."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""

        pipeline = MiddlewarePipeline()

        assert isinstance(pipeline.preprocessing_middleware, list)
        assert isinstance(pipeline.processing_middleware, list)
        assert isinstance(pipeline.postprocessing_middleware, list)
        assert isinstance(pipeline.error_handling_middleware, list)
        assert isinstance(pipeline.global_middleware, list)

    def test_add_middleware_methods(self, test_middleware):
        """Test adding middleware to different phases."""

        pipeline = MiddlewarePipeline()

        # Test adding to different phases
        pipeline.add_preprocessing(test_middleware)
        assert test_middleware in pipeline.preprocessing_middleware

        pipeline.add_processing(test_middleware)
        assert test_middleware in pipeline.processing_middleware

        pipeline.add_postprocessing(test_middleware)
        assert test_middleware in pipeline.postprocessing_middleware

        pipeline.add_error_handling(test_middleware)
        assert test_middleware in pipeline.error_handling_middleware

        pipeline.add_global(test_middleware)
        assert test_middleware in pipeline.global_middleware

    def test_middleware_priority_sorting(self):
        """Test middleware sorting by priority."""

        class LowPriorityMiddleware(AgentMiddleware):
            def __init__(self):
                super().__init__("LowPriority")
                self.priority = 100

            async def process(self, context, next_middleware):
                pass

        class HighPriorityMiddleware(AgentMiddleware):
            def __init__(self):
                super().__init__("HighPriority")
                self.priority = 1

            async def process(self, context, next_middleware):
                pass

        pipeline = MiddlewarePipeline()
        low_priority = LowPriorityMiddleware()
        high_priority = HighPriorityMiddleware()

        # Add in reverse priority order
        pipeline.add_preprocessing(low_priority)
        pipeline.add_preprocessing(high_priority)

        # Should be sorted by priority (low number = high priority)
        assert pipeline.preprocessing_middleware[0] == high_priority
        assert pipeline.preprocessing_middleware[1] == low_priority

    @pytest.mark.asyncio
    async def test_pipeline_execution_success(self, sample_middleware_context):
        """Test successful pipeline execution."""

        execution_order = []

        class OrderTrackingMiddleware(AgentMiddleware):
            def __init__(self, name, phase):
                super().__init__(name)
                self.phase = phase

            async def process(self, context, next_middleware):
                execution_order.append(f"{self.phase}_{self.name}")
                await next_middleware(context)

        pipeline = MiddlewarePipeline()
        pipeline.add_preprocessing(OrderTrackingMiddleware("Pre1", "pre"))
        pipeline.add_preprocessing(OrderTrackingMiddleware("Pre2", "pre"))
        pipeline.add_processing(OrderTrackingMiddleware("Proc1", "proc"))
        pipeline.add_postprocessing(OrderTrackingMiddleware("Post1", "post"))

        result_context = await pipeline.execute(sample_middleware_context)

        assert result_context == sample_middleware_context
        assert result_context.error is None

        # Check execution order
        expected_order = ["pre_Pre1", "pre_Pre2", "proc_Proc1", "post_Post1"]
        assert execution_order == expected_order

    @pytest.mark.asyncio
    async def test_pipeline_execution_with_error(self, sample_middleware_context):
        """Test pipeline execution when error occurs."""

        class ErrorMiddleware(AgentMiddleware):
            async def process(self, context, next_middleware):
                raise ValueError("Test middleware error")

        class ErrorHandlingMiddleware(AgentMiddleware):
            def __init__(self):
                super().__init__("ErrorHandler")
                self.handled_error = False

            async def process(self, context, next_middleware):
                self.handled_error = True
                context.error = None  # Clear error

        pipeline = MiddlewarePipeline()
        error_handler = ErrorHandlingMiddleware()

        pipeline.add_preprocessing(ErrorMiddleware("ErrorMiddleware"))
        pipeline.add_error_handling(error_handler)

        result_context = await pipeline.execute(sample_middleware_context)

        # Error should be handled
        assert result_context.error is None
        assert error_handler.handled_error is True

    @pytest.mark.asyncio
    async def test_pipeline_short_circuit_on_error(self, sample_middleware_context):
        """Test that pipeline stops processing on error."""

        execution_order = []

        class TrackingMiddleware(AgentMiddleware):
            def __init__(self, name):
                super().__init__(name)

            async def process(self, context, next_middleware):
                execution_order.append(self.name)
                if self.name == "ErrorMiddleware":
                    context.set_error(ValueError("Test error"), "test")
                    return
                await next_middleware(context)

        pipeline = MiddlewarePipeline()
        pipeline.add_preprocessing(TrackingMiddleware("Pre1"))
        pipeline.add_preprocessing(TrackingMiddleware("ErrorMiddleware"))
        pipeline.add_preprocessing(TrackingMiddleware("Pre3"))  # Should not execute
        pipeline.add_processing(TrackingMiddleware("Proc1"))  # Should not execute

        result_context = await pipeline.execute(sample_middleware_context)

        assert result_context.error is not None
        # Only Pre1 and ErrorMiddleware should have executed
        assert execution_order == ["Pre1", "ErrorMiddleware"]


class TestAuthenticationMiddleware:
    """Test AuthenticationMiddleware."""

    def test_auth_middleware_initialization(self):
        """Test authentication middleware initialization."""

        api_keys = {"key1", "key2", "key3"}
        middleware = AuthenticationMiddleware(
            required_auth=True, api_keys=api_keys, bearer_tokens={"token1", "token2"}
        )

        assert middleware.required_auth is True
        assert middleware.api_keys == api_keys
        assert middleware.bearer_tokens == {"token1", "token2"}

    @pytest.mark.asyncio
    async def test_auth_middleware_no_auth_required(self, sample_middleware_context):
        """Test middleware when authentication is not required."""

        middleware = AuthenticationMiddleware(required_auth=False)

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is True
        assert sample_middleware_context.error is None

    @pytest.mark.asyncio
    async def test_auth_middleware_valid_api_key(self, sample_middleware_context):
        """Test middleware with valid API key."""

        middleware = AuthenticationMiddleware(required_auth=True, api_keys={"valid_key"})

        # Add API key to request metadata
        sample_middleware_context.request.metadata["api_key"] = "valid_key"

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is True
        assert sample_middleware_context.error is None
        assert sample_middleware_context.metadata["authenticated"] is True
        assert sample_middleware_context.metadata["auth_method"] == "api_key"

    @pytest.mark.asyncio
    async def test_auth_middleware_invalid_api_key(self, sample_middleware_context):
        """Test middleware with invalid API key."""

        middleware = AuthenticationMiddleware(required_auth=True, api_keys={"valid_key"})

        # Add invalid API key
        sample_middleware_context.request.metadata["api_key"] = "invalid_key"

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is False
        assert sample_middleware_context.error is not None
        assert "Authentication failed" in str(sample_middleware_context.error)

    @pytest.mark.asyncio
    async def test_auth_middleware_bearer_token(self, sample_middleware_context):
        """Test middleware with valid bearer token."""

        middleware = AuthenticationMiddleware(required_auth=True, bearer_tokens={"valid_token"})

        # Add bearer token to request metadata
        sample_middleware_context.request.metadata["Authorization"] = "Bearer valid_token"

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is True
        assert sample_middleware_context.error is None
        assert sample_middleware_context.metadata["auth_method"] == "bearer_token"


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware."""

    def test_rate_limiting_initialization(self):
        """Test rate limiting middleware initialization."""

        middleware = RateLimitingMiddleware(
            max_requests_per_minute=100, max_requests_per_hour=1000, burst_limit=10
        )

        assert middleware.max_requests_per_minute == 100
        assert middleware.max_requests_per_hour == 1000
        assert middleware.burst_limit == 10

    @pytest.mark.asyncio
    async def test_rate_limiting_within_limits(self, sample_middleware_context):
        """Test rate limiting when within limits."""

        middleware = RateLimitingMiddleware(max_requests_per_minute=10, burst_limit=5)

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is True
        assert sample_middleware_context.error is None
        assert sample_middleware_context.metadata["rate_limit_checked"] is True

    @pytest.mark.asyncio
    async def test_rate_limiting_burst_exceeded(self, sample_middleware_context):
        """Test rate limiting when burst limit is exceeded."""

        middleware = RateLimitingMiddleware(
            max_requests_per_minute=100, burst_limit=2  # Low burst limit
        )

        user_key = middleware.key_function(sample_middleware_context.request)

        # Simulate burst requests
        current_time = datetime.now(timezone.utc)
        middleware.burst_tracking[user_key] = [current_time] * 3  # Exceed burst limit

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is False
        assert sample_middleware_context.error is not None
        assert "burst window" in str(sample_middleware_context.error)

    def test_rate_limiting_clean_old_requests(self):
        """Test cleaning of old requests."""

        middleware = RateLimitingMiddleware()

        old_time = datetime.now(timezone.utc) - timedelta(seconds=70)
        recent_time = datetime.now(timezone.utc) - timedelta(seconds=30)

        requests = [old_time, recent_time, datetime.now(timezone.utc)]
        cleaned = middleware._clean_old_requests(requests, 60)

        # Only recent requests should remain
        assert len(cleaned) == 2
        assert old_time not in cleaned


class TestValidationMiddleware:
    """Test ValidationMiddleware."""

    def test_validation_middleware_initialization(self):
        """Test validation middleware initialization."""

        middleware = ValidationMiddleware(
            max_message_length=5000,
            min_message_length=5,
            blocked_patterns=["badword", "spam"],
            sanitize_input=True,
        )

        assert middleware.max_message_length == 5000
        assert middleware.min_message_length == 5
        assert "badword" in middleware.blocked_patterns
        assert middleware.sanitize_input is True

    @pytest.mark.asyncio
    async def test_validation_middleware_valid_message(self, sample_middleware_context):
        """Test validation with valid message."""

        middleware = ValidationMiddleware(max_message_length=1000)

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is True
        assert sample_middleware_context.error is None
        assert sample_middleware_context.metadata["validated"] is True

    @pytest.mark.asyncio
    async def test_validation_middleware_message_too_long(self, sample_middleware_context):
        """Test validation with message too long."""

        middleware = ValidationMiddleware(max_message_length=10)

        # Set long message
        sample_middleware_context.request.message = "a" * 20

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is False
        assert sample_middleware_context.error is not None
        assert "Message too long" in str(sample_middleware_context.error)

    @pytest.mark.asyncio
    async def test_validation_middleware_blocked_pattern(self, sample_middleware_context):
        """Test validation with blocked pattern."""

        middleware = ValidationMiddleware(blocked_patterns=["badword"])

        # Set message with blocked content
        sample_middleware_context.request.message = "This contains badword content"

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is False
        assert sample_middleware_context.error is not None
        assert "blocked content" in str(sample_middleware_context.error)

    def test_validation_sanitize_message(self):
        """Test message sanitization."""

        middleware = ValidationMiddleware(sanitize_input=True)

        dangerous_message = "Hello <script>alert('xss')</script> world"
        sanitized = middleware._sanitize_message(dangerous_message)

        assert "<script>" not in sanitized
        assert "alert" not in sanitized
        assert "Hello" in sanitized
        assert "world" in sanitized


class TestCachingMiddleware:
    """Test CachingMiddleware."""

    def test_caching_middleware_initialization(self):
        """Test caching middleware initialization."""

        middleware = CachingMiddleware(cache_ttl_seconds=600, max_cache_size=1000)

        assert middleware.cache_ttl_seconds == 600
        assert middleware.max_cache_size == 1000
        assert isinstance(middleware.cache, dict)

    @pytest.mark.asyncio
    async def test_caching_middleware_cache_miss(
        self, sample_middleware_context, sample_agent_response
    ):
        """Test caching middleware on cache miss."""

        middleware = CachingMiddleware(cache_ttl_seconds=300)

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True
            context.response = sample_agent_response

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is True
        assert sample_middleware_context.response == sample_agent_response
        assert sample_middleware_context.metadata.get("cache_hit") is not True
        assert sample_middleware_context.metadata.get("cached") is True

    @pytest.mark.asyncio
    async def test_caching_middleware_cache_hit(
        self, sample_middleware_context, sample_agent_response
    ):
        """Test caching middleware on cache hit."""

        middleware = CachingMiddleware(cache_ttl_seconds=300)

        # Pre-populate cache
        cache_key = middleware.cache_key_function(sample_middleware_context.request)
        middleware.cache[cache_key] = {
            "response": {
                "content": sample_agent_response.content,
                "agent_id": sample_agent_response.agent_id,
                "session_id": sample_agent_response.session_id,
                "execution_time_ms": sample_agent_response.execution_time_ms,
                "metadata": sample_agent_response.metadata,
                "error": sample_agent_response.error,
                "timestamp": sample_agent_response.timestamp,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        middleware.cache_access_times[cache_key] = datetime.now(timezone.utc)

        next_called = False

        async def next_middleware(context):
            nonlocal next_called
            next_called = True

        await middleware.process(sample_middleware_context, next_middleware)

        assert next_called is False  # Should not call next due to cache hit
        assert sample_middleware_context.response is not None
        assert sample_middleware_context.metadata["cache_hit"] is True

    def test_caching_cache_key_generation(self, sample_agent_request):
        """Test cache key generation."""

        middleware = CachingMiddleware()

        key1 = middleware._default_cache_key(sample_agent_request)
        key2 = middleware._default_cache_key(sample_agent_request)

        # Same request should generate same key
        assert key1 == key2

        # Different request should generate different key
        different_request = AgentRequest(
            message="Different message", agent_id=sample_agent_request.agent_id
        )
        key3 = middleware._default_cache_key(different_request)
        assert key1 != key3


class TestErrorHandlingMiddleware:
    """Test ErrorHandlingMiddleware."""

    @pytest.mark.asyncio
    async def test_error_handling_middleware_with_error(self, sample_middleware_context):
        """Test error handling middleware when error is present."""

        middleware = ErrorHandlingMiddleware(enable_detailed_errors=True)

        # Set an error in context
        test_error = ValueError("Test error message")
        sample_middleware_context.error = test_error

        await middleware.process(sample_middleware_context, lambda ctx: None)

        # Error should be cleared and response created
        assert sample_middleware_context.error is None
        assert sample_middleware_context.response is not None
        assert sample_middleware_context.response.error == "Test error message"
        assert "Test error message" in sample_middleware_context.response.content
        assert sample_middleware_context.response.metadata["error_handled"] is True

    @pytest.mark.asyncio
    async def test_error_handling_middleware_no_error(self, sample_middleware_context):
        """Test error handling middleware when no error is present."""

        middleware = ErrorHandlingMiddleware()

        # No error in context
        assert sample_middleware_context.error is None

        await middleware.process(sample_middleware_context, lambda ctx: None)

        # Should not create response if no error
        assert sample_middleware_context.error is None
        assert sample_middleware_context.response is None


class TestPipelineFactories:
    """Test pipeline factory functions."""

    def test_create_default_pipeline(self):
        """Test default pipeline creation."""

        pipeline = create_default_pipeline(
            enable_auth=True,
            enable_rate_limiting=True,
            enable_validation=True,
            enable_caching=True,
            enable_telemetry=True,
            api_keys={"test_key"},
            max_requests_per_minute=60,
        )

        assert isinstance(pipeline, MiddlewarePipeline)
        assert len(pipeline.preprocessing_middleware) > 0
        assert len(pipeline.error_handling_middleware) > 0

        # Check that expected middleware types are present
        middleware_types = [type(m).__name__ for m in pipeline.preprocessing_middleware]
        assert "AuthenticationMiddleware" in middleware_types
        assert "ValidationMiddleware" in middleware_types
        assert "CachingMiddleware" in middleware_types

    def test_create_production_pipeline(self):
        """Test production pipeline creation."""

        pipeline = create_production_pipeline(
            api_keys={"prod_key_1", "prod_key_2"},
            max_requests_per_minute=100,
            max_requests_per_hour=5000,
            cache_ttl_seconds=600,
            enable_detailed_errors=False,
        )

        assert isinstance(pipeline, MiddlewarePipeline)

        # Should have stricter settings for production
        auth_middleware = None
        for middleware in pipeline.preprocessing_middleware:
            if isinstance(middleware, AuthenticationMiddleware):
                auth_middleware = middleware
                break

        assert auth_middleware is not None
        assert auth_middleware.required_auth is True
        assert "prod_key_1" in auth_middleware.api_keys

    def test_create_pipeline_with_disabled_features(self):
        """Test pipeline creation with features disabled."""

        pipeline = create_default_pipeline(
            enable_auth=False,
            enable_rate_limiting=False,
            enable_validation=False,
            enable_caching=False,
            enable_telemetry=False,
        )

        # Should still have error handling
        assert len(pipeline.error_handling_middleware) > 0

        # But preprocessing should be minimal
        preprocessing_types = [type(m).__name__ for m in pipeline.preprocessing_middleware]
        assert "AuthenticationMiddleware" not in preprocessing_types
        assert "RateLimitingMiddleware" not in preprocessing_types


@pytest.mark.integration
class TestMiddlewareIntegration:
    """Integration tests for middleware pipeline."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, sample_agent_request, sample_agent_response):
        """Test complete middleware pipeline flow."""

        # Create pipeline with multiple middleware
        pipeline = create_default_pipeline(
            enable_auth=False,  # Disable auth for testing
            enable_rate_limiting=False,  # Disable rate limiting for testing
            enable_validation=True,
            enable_caching=True,
            enable_telemetry=False,  # Disable telemetry for testing
        )

        # Create context
        context = MiddlewareContext(request=sample_agent_request)

        # Mock processing middleware to set response
        class MockProcessingMiddleware(AgentMiddleware):
            async def process(self, context, next_middleware):
                context.response = sample_agent_response
                await next_middleware(context)

        pipeline.add_processing(MockProcessingMiddleware("MockProcessor"))

        # Execute pipeline
        result_context = await pipeline.execute(context)

        assert result_context.error is None
        assert result_context.response == sample_agent_response
        assert len(result_context.execution_phases) > 0
        assert result_context.metadata.get("validated") is True

    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(self, sample_agent_request):
        """Test error propagation through pipeline."""

        pipeline = MiddlewarePipeline()

        # Add middleware that raises error
        class ErrorMiddleware(AgentMiddleware):
            async def process(self, context, next_middleware):
                raise RuntimeError("Pipeline error")

        # Add error handler
        class TestErrorHandler(AgentMiddleware):
            def __init__(self):
                super().__init__("TestErrorHandler")
                self.handled = False

            async def process(self, context, next_middleware):
                self.handled = True
                context.response = AgentResponse(content="Error handled", agent_id="ErrorHandler")
                context.error = None

        error_handler = TestErrorHandler()
        pipeline.add_preprocessing(ErrorMiddleware("ErrorMiddleware"))
        pipeline.add_error_handling(error_handler)

        context = MiddlewareContext(request=sample_agent_request)
        result_context = await pipeline.execute(context)

        assert error_handler.handled is True
        assert result_context.error is None
        assert result_context.response.content == "Error handled"

    @pytest.mark.asyncio
    async def test_middleware_order_execution(self, sample_agent_request):
        """Test that middleware executes in correct order."""

        execution_order = []

        class OrderedMiddleware(AgentMiddleware):
            def __init__(self, name, priority):
                super().__init__(name)
                self.priority = priority

            async def process(self, context, next_middleware):
                execution_order.append(self.name)
                await next_middleware(context)

        pipeline = MiddlewarePipeline()

        # Add middleware with different priorities
        pipeline.add_preprocessing(OrderedMiddleware("Third", 30))
        pipeline.add_preprocessing(OrderedMiddleware("First", 10))
        pipeline.add_preprocessing(OrderedMiddleware("Second", 20))

        context = MiddlewareContext(request=sample_agent_request)
        await pipeline.execute(context)

        assert execution_order == ["First", "Second", "Third"]


@pytest.mark.slow
class TestMiddlewarePerformance:
    """Performance tests for middleware."""

    @pytest.mark.asyncio
    async def test_pipeline_performance(self, sample_agent_request, performance_monitor):
        """Test middleware pipeline performance."""

        # Create pipeline with multiple middleware
        pipeline = create_default_pipeline()

        # Add processing middleware to complete the flow
        class MockProcessingMiddleware(AgentMiddleware):
            async def process(self, context, next_middleware):
                context.response = AgentResponse(content="Test response", agent_id="TestAgent")
                await next_middleware(context)

        pipeline.add_processing(MockProcessingMiddleware("MockProcessor"))

        context = MiddlewareContext(request=sample_agent_request)

        monitor = performance_monitor.start("pipeline_execution")
        await pipeline.execute(context)
        duration = monitor.stop("pipeline_execution")

        # Pipeline should execute quickly
        assert duration < 1.0  # Less than 1 second
        assert context.error is None

    @pytest.mark.asyncio
    async def test_caching_performance_improvement(self, sample_agent_request):
        """Test that caching improves performance."""

        class SlowProcessingMiddleware(AgentMiddleware):
            def __init__(self):
                super().__init__("SlowProcessor")
                self.call_count = 0

            async def process(self, context, next_middleware):
                self.call_count += 1
                await asyncio.sleep(0.1)  # Simulate slow processing
                context.response = AgentResponse(
                    content=f"Response {self.call_count}", agent_id="TestAgent"
                )
                await next_middleware(context)

        pipeline = MiddlewarePipeline()
        slow_processor = SlowProcessingMiddleware()

        pipeline.add_preprocessing(CachingMiddleware(cache_ttl_seconds=300))
        pipeline.add_processing(slow_processor)

        # First request (cache miss)
        context1 = MiddlewareContext(request=sample_agent_request)
        start_time = time.time()
        await pipeline.execute(context1)
        first_duration = time.time() - start_time

        # Second identical request (cache hit)
        context2 = MiddlewareContext(request=sample_agent_request)
        start_time = time.time()
        await pipeline.execute(context2)
        second_duration = time.time() - start_time

        # Second request should be significantly faster
        assert second_duration < first_duration * 0.5
        assert slow_processor.call_count == 1  # Only called once due to caching
        assert context2.metadata.get("cache_hit") is True


if __name__ == "__main__":
    pytest.main([__file__])
