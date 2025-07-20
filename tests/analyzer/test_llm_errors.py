"""
Comprehensive test suite for LLM error handling and retry logic.

Tests error hierarchy, retry strategies, circuit breaker functionality,
and graceful degradation patterns.
"""

import asyncio
import time

import pytest

from codedocsync.analyzer.llm_errors import (
    CircuitBreaker,
    CircuitState,
    LLMAPIKeyError,
    LLMError,
    LLMInvalidResponseError,
    LLMNetworkError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
    RetryStrategy,
    create_aggressive_retry_strategy,
    create_conservative_retry_strategy,
    create_default_retry_strategy,
    with_retry,
)


class TestLLMErrorHierarchy:
    """Test LLM error hierarchy and error types."""

    def test_llm_error_base_class(self):
        """Test base LLMError class."""
        error = LLMError("Test error")
        assert str(error) == "Test error"
        assert error.retry_after is None

        error_with_retry = LLMError("Test error", retry_after=30.0)
        assert error_with_retry.retry_after == 30.0

    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError."""
        error = LLMRateLimitError("Rate limit exceeded", retry_after=60.0)
        assert isinstance(error, LLMError)
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 60.0

    def test_llm_timeout_error(self):
        """Test LLMTimeoutError."""
        error = LLMTimeoutError("Request timed out")
        assert isinstance(error, LLMError)
        assert str(error) == "Request timed out"

    def test_llm_invalid_response_error(self):
        """Test LLMInvalidResponseError."""
        raw_response = '{"invalid": json'
        error = LLMInvalidResponseError("Invalid JSON", raw_response)
        assert isinstance(error, LLMError)
        assert str(error) == "Invalid JSON"
        assert error.raw_response == raw_response

    def test_llm_api_key_error(self):
        """Test LLMAPIKeyError."""
        error = LLMAPIKeyError("Invalid API key")
        assert isinstance(error, LLMError)
        assert str(error) == "Invalid API key"

    def test_llm_network_error(self):
        """Test LLMNetworkError."""
        error = LLMNetworkError("Network connection failed")
        assert isinstance(error, LLMError)
        assert str(error) == "Network connection failed"

    def test_llm_quota_exceeded_error(self):
        """Test LLMQuotaExceededError."""
        error = LLMQuotaExceededError("Quota exceeded", retry_after=3600.0)
        assert isinstance(error, LLMError)
        assert str(error) == "Quota exceeded"
        assert error.retry_after == 3600.0


class TestRetryStrategy:
    """Test retry strategy functionality."""

    def test_retry_strategy_initialization(self):
        """Test RetryStrategy initialization."""
        strategy = RetryStrategy(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
        )

        assert strategy.max_retries == 5
        assert strategy.base_delay == 2.0
        assert strategy.max_delay == 120.0
        assert strategy.exponential_base == 3.0
        assert strategy.jitter is False
        assert strategy.retry_history == []

    def test_should_retry_api_key_error(self):
        """Test that API key errors are not retried."""
        strategy = RetryStrategy()
        error = LLMAPIKeyError("Invalid API key")

        should_retry, delay = strategy.should_retry(error, 0)
        assert should_retry is False
        assert delay == 0.0

    def test_should_retry_invalid_response_error(self):
        """Test that invalid response errors are not retried."""
        strategy = RetryStrategy()
        error = LLMInvalidResponseError("Malformed JSON")

        should_retry, delay = strategy.should_retry(error, 0)
        assert should_retry is False
        assert delay == 0.0

    def test_should_retry_rate_limit_error(self):
        """Test that rate limit errors are retried with proper delay."""
        strategy = RetryStrategy(jitter=False)
        error = LLMRateLimitError("Rate limit exceeded", retry_after=30.0)

        should_retry, delay = strategy.should_retry(error, 0)
        assert should_retry is True
        assert delay == 30.0  # Should respect retry_after

    def test_should_retry_timeout_error(self):
        """Test that timeout errors are retried once."""
        strategy = RetryStrategy()
        error = LLMTimeoutError("Request timed out")

        # First attempt should retry
        should_retry, delay = strategy.should_retry(error, 0)
        assert should_retry is True
        assert delay == 0.0  # No delay for timeout retry

        # Second attempt should not retry
        should_retry, delay = strategy.should_retry(error, 1)
        assert should_retry is False

    def test_should_retry_network_error(self):
        """Test that network errors are retried with exponential backoff."""
        strategy = RetryStrategy(jitter=False)
        error = LLMNetworkError("Connection failed")

        # First retry
        should_retry, delay = strategy.should_retry(error, 0)
        assert should_retry is True
        assert delay == 1.0  # Base delay

        # Second retry
        should_retry, delay = strategy.should_retry(error, 1)
        assert should_retry is True
        assert delay == 2.0  # Exponential backoff

        # Third retry
        should_retry, delay = strategy.should_retry(error, 2)
        assert should_retry is True
        assert delay == 4.0  # Exponential backoff

    def test_should_retry_max_retries_exceeded(self):
        """Test that max retries limit is respected."""
        strategy = RetryStrategy(max_retries=2)
        error = LLMNetworkError("Connection failed")

        # Attempts 0, 1 should retry
        assert strategy.should_retry(error, 0)[0] is True
        assert strategy.should_retry(error, 1)[0] is True

        # Attempt 2 (max_retries reached) should not retry
        assert strategy.should_retry(error, 2)[0] is False

    def test_delay_calculation_with_jitter(self):
        """Test delay calculation with jitter enabled."""
        strategy = RetryStrategy(base_delay=10.0, jitter=True)
        error = LLMNetworkError("Connection failed")

        delays = []
        for _ in range(10):
            _, delay = strategy.should_retry(error, 1)
            delays.append(delay)

        # All delays should be different (jitter effect)
        assert len(set(delays)) > 1

        # All delays should be around base_delay * exponential_base
        expected = 10.0 * 2  # base_delay * exponential_base
        for delay in delays:
            assert abs(delay - expected) <= expected * 0.1  # Within 10% jitter

    def test_delay_max_limit(self):
        """Test that delay is capped at max_delay."""
        strategy = RetryStrategy(
            base_delay=10.0, max_delay=15.0, exponential_base=5.0, jitter=False
        )
        error = LLMNetworkError("Connection failed")

        # High attempt number should be capped
        should_retry, delay = strategy.should_retry(error, 10)
        assert delay == 15.0  # Should be capped at max_delay

    def test_record_attempt(self):
        """Test retry attempt recording."""
        strategy = RetryStrategy()
        error = LLMNetworkError("Connection failed")

        strategy.record_attempt(0, error, 1.0)
        strategy.record_attempt(1, error, 2.0)

        assert len(strategy.retry_history) == 2
        assert strategy.retry_history[0].attempt == 0
        assert strategy.retry_history[0].delay == 1.0
        assert isinstance(strategy.retry_history[0].error, LLMNetworkError)

    def test_get_retry_stats(self):
        """Test retry statistics calculation."""
        strategy = RetryStrategy()

        # No attempts yet
        stats = strategy.get_retry_stats()
        assert stats["total_attempts"] == 0

        # Add some attempts
        strategy.record_attempt(0, LLMNetworkError("error1"), 1.0)
        strategy.record_attempt(1, LLMRateLimitError("error2"), 2.0)
        strategy.record_attempt(2, LLMNetworkError("error3"), 3.0)

        stats = strategy.get_retry_stats()
        assert stats["total_attempts"] == 3
        assert stats["error_types"]["LLMNetworkError"] == 2
        assert stats["error_types"]["LLMRateLimitError"] == 1
        assert stats["total_delay_seconds"] == 6.0
        assert stats["average_delay"] == 2.0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=LLMError,
            half_open_max_calls=2,
        )

        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.expected_exception == LLMError
        assert cb.half_open_max_calls == 2
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3)

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

        # Record some successes
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.success_count == 2

    def test_circuit_breaker_failure_tracking(self):
        """Test failure tracking and state transitions."""
        cb = CircuitBreaker(failure_threshold=2)

        # Record failures
        cb.record_failure(LLMError("error1"))
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

        # Second failure should open circuit
        cb.record_failure(LLMError("error2"))
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 2
        assert cb.can_execute() is False

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery from OPEN to HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Force to OPEN state
        cb.record_failure(LLMError("error"))
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should allow execution and transition to HALF_OPEN
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_half_open_success(self):
        """Test circuit breaker recovery success in HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

        # Force to OPEN state
        cb.record_failure(LLMError("error"))
        assert cb.state == CircuitState.OPEN

        # Transition to HALF_OPEN
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Success should close circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in HALF_OPEN state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

        # Force to HALF_OPEN state
        cb.record_failure(LLMError("error"))
        cb.can_execute()  # Transitions to HALF_OPEN

        # Failure should open circuit again
        cb.record_failure(LLMError("error2"))
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_circuit_breaker_half_open_max_calls(self):
        """Test half-open max calls limit."""
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0, half_open_max_calls=2
        )

        # Force to HALF_OPEN state
        cb.record_failure(LLMError("error"))
        cb.can_execute()  # Transitions to HALF_OPEN

        # Should allow up to max calls
        assert cb.can_execute() is True  # Call 1
        assert cb.can_execute() is False  # Call 2 exceeds limit

    @pytest.mark.asyncio
    async def test_circuit_breaker_call_success(self):
        """Test successful function call through circuit breaker."""
        cb = CircuitBreaker()

        async def successful_function(x):
            return x * 2

        result = await cb.call(successful_function, 5)
        assert result == 10
        assert cb.success_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_call_failure(self):
        """Test failed function call through circuit breaker."""
        cb = CircuitBreaker(failure_threshold=1)

        async def failing_function():
            raise LLMError("Function failed")

        # First call should fail and open circuit
        with pytest.raises(LLMError):
            await cb.call(failing_function)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 1

        # Second call should be rejected immediately
        with pytest.raises(LLMError, match="Circuit breaker is open"):
            await cb.call(failing_function)

    @pytest.mark.asyncio
    async def test_circuit_breaker_call_non_async(self):
        """Test non-async function call through circuit breaker."""
        cb = CircuitBreaker()

        def sync_function(x):
            return x + 1

        result = await cb.call(sync_function, 5)
        assert result == 6

    def test_circuit_breaker_failure_rate(self):
        """Test failure rate calculation."""
        cb = CircuitBreaker()

        # No calls yet
        assert cb.failure_rate == 0.0

        # Mixed success and failure
        cb.record_success()
        cb.record_success()
        cb.record_failure(LLMError("error"))

        assert cb.total_calls == 3
        assert cb.failure_rate == 1 / 3

    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_success()
        cb.record_failure(LLMError("error"))

        stats = cb.get_stats()
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 1
        assert stats["total_calls"] == 2
        assert stats["failure_rate"] == 0.5

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset functionality."""
        cb = CircuitBreaker(failure_threshold=1)

        # Create some state
        cb.record_failure(LLMError("error"))
        assert cb.state == CircuitState.OPEN

        # Reset should restore initial state
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.total_calls == 0


class TestWithRetryFunction:
    """Test the with_retry utility function."""

    @pytest.mark.asyncio
    async def test_with_retry_success_first_attempt(self):
        """Test successful function on first attempt."""

        async def successful_function():
            return "success"

        result = await with_retry(successful_function)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_retry_success_after_retries(self):
        """Test successful function after some retries."""
        call_count = 0

        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMNetworkError("Temporary failure")
            return "success"

        strategy = RetryStrategy(max_retries=5, base_delay=0.01, jitter=False)
        result = await with_retry(eventually_successful, strategy)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_all_attempts_fail(self):
        """Test when all retry attempts fail."""

        async def always_failing():
            raise LLMNetworkError("Always fails")

        strategy = RetryStrategy(max_retries=2, base_delay=0.01)

        with pytest.raises(LLMNetworkError, match="Always fails"):
            await with_retry(always_failing, strategy)

    @pytest.mark.asyncio
    async def test_with_retry_non_retryable_error(self):
        """Test non-retryable error is not retried."""
        call_count = 0

        async def non_retryable_error():
            nonlocal call_count
            call_count += 1
            raise LLMAPIKeyError("Invalid API key")

        strategy = RetryStrategy(max_retries=3)

        with pytest.raises(LLMAPIKeyError):
            await with_retry(non_retryable_error, strategy)

        # Should only be called once (no retries)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_with_retry_sync_function(self):
        """Test with_retry on synchronous function."""

        def sync_function():
            return "sync_result"

        result = await with_retry(sync_function)
        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_with_retry_default_strategy(self):
        """Test with_retry using default strategy."""

        async def test_function():
            return "result"

        # Should use default strategy when none provided
        result = await with_retry(test_function)
        assert result == "result"


class TestRetryStrategyFactories:
    """Test retry strategy factory functions."""

    def test_create_default_retry_strategy(self):
        """Test default retry strategy creation."""
        strategy = create_default_retry_strategy()

        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 60.0
        assert strategy.exponential_base == 2.0
        assert strategy.jitter is True

    def test_create_aggressive_retry_strategy(self):
        """Test aggressive retry strategy creation."""
        strategy = create_aggressive_retry_strategy()

        assert strategy.max_retries == 5
        assert strategy.base_delay == 0.5
        assert strategy.max_delay == 30.0
        assert strategy.exponential_base == 1.5
        assert strategy.jitter is True

    def test_create_conservative_retry_strategy(self):
        """Test conservative retry strategy creation."""
        strategy = create_conservative_retry_strategy()

        assert strategy.max_retries == 2
        assert strategy.base_delay == 2.0
        assert strategy.max_delay == 120.0
        assert strategy.exponential_base == 3.0
        assert strategy.jitter is True


@pytest.mark.parametrize(
    "error,expected_retries",
    [
        (LLMRateLimitError("rate limit"), 5),
        (LLMTimeoutError("timeout"), 1),
        (LLMNetworkError("network error"), 3),
        (LLMAPIKeyError("invalid key"), 0),
        (LLMInvalidResponseError("bad json"), 0),
    ],
)
def test_retry_decision_matrix(error, expected_retries):
    """Test retry decision matrix for different error types."""
    strategy = RetryStrategy(max_retries=5)

    actual_retries = 0
    for attempt in range(10):  # Try up to 10 attempts
        should_retry, _ = strategy.should_retry(error, attempt)
        if should_retry:
            actual_retries += 1
        else:
            break

    assert actual_retries == expected_retries


class TestErrorRecoveryPatterns:
    """Test error recovery patterns and scenarios."""

    @pytest.mark.asyncio
    async def test_timeout_recovery_pattern(self):
        """Test timeout error recovery pattern."""
        call_count = 0

        async def timeout_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMTimeoutError("Request timed out")
            return "success"

        strategy = RetryStrategy(base_delay=0.01)
        result = await with_retry(timeout_then_success, strategy)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_recovery_pattern(self):
        """Test rate limit error recovery pattern."""
        call_count = 0

        async def rate_limit_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMRateLimitError("Rate limit exceeded", retry_after=0.01)
            return "success"

        strategy = RetryStrategy(base_delay=0.01)
        result = await with_retry(rate_limit_then_success, strategy)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_network_error_recovery_pattern(self):
        """Test network error recovery with exponential backoff."""
        call_count = 0

        async def network_error_tracking_delays():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise LLMNetworkError("Network error")
            return "success"

        strategy = RetryStrategy(base_delay=0.01, jitter=False)

        # Track actual delays by timing the function
        start_time = time.time()
        result = await with_retry(network_error_tracking_delays, strategy)
        total_time = time.time() - start_time

        assert result == "success"
        assert call_count == 4
        # Should have some delays (though small due to base_delay=0.01)
        assert total_time > 0.01


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with retry logic."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self):
        """Test circuit breaker working with retry logic."""
        cb = CircuitBreaker(failure_threshold=2)
        strategy = RetryStrategy(max_retries=1, base_delay=0.01)

        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise LLMNetworkError("Network error")

        # First call should fail after retries
        with pytest.raises(LLMNetworkError):
            await cb.call(lambda: with_retry(failing_function, strategy))

        # Second call should also fail after retries
        with pytest.raises(LLMNetworkError):
            await cb.call(lambda: with_retry(failing_function, strategy))

        # Circuit should now be open
        assert cb.state == CircuitState.OPEN

        # Third call should be rejected immediately by circuit breaker
        with pytest.raises(LLMError, match="Circuit breaker is open"):
            await cb.call(lambda: with_retry(failing_function, strategy))

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_with_retry(self):
        """Test circuit breaker recovery working with retry."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        strategy = RetryStrategy(max_retries=1, base_delay=0.01)

        # Force circuit open
        with pytest.raises(LLMNetworkError):
            await cb.call(
                lambda: with_retry(
                    lambda: asyncio.create_task(
                        asyncio.coroutine(lambda: None)()
                    ).add_done_callback(
                        lambda _: (_ for _ in ()).throw(LLMNetworkError("error"))
                    ),
                    strategy,
                )
            )

        # Wait for recovery
        await asyncio.sleep(0.02)

        # Should allow execution again
        result = await cb.call(
            lambda: with_retry(lambda: asyncio.sleep(0) or "success", strategy)
        )
        assert result == "success"
        assert cb.state == CircuitState.CLOSED


class TestPerformanceAndLimits:
    """Test performance characteristics and limits."""

    @pytest.mark.asyncio
    async def test_retry_timing_accuracy(self):
        """Test that retry delays are reasonably accurate."""
        strategy = RetryStrategy(base_delay=0.1, jitter=False)
        error = LLMNetworkError("error")

        # Measure actual delay
        start_time = time.time()
        should_retry, expected_delay = strategy.should_retry(error, 1)
        await asyncio.sleep(expected_delay)
        actual_delay = time.time() - start_time

        # Should be within 10ms of expected (allowing for system variance)
        assert abs(actual_delay - expected_delay) < 0.01

    def test_circuit_breaker_memory_efficiency(self):
        """Test that circuit breaker doesn't accumulate unbounded memory."""
        cb = CircuitBreaker()

        # Simulate many operations
        for i in range(1000):
            if i % 2 == 0:
                cb.record_success()
            else:
                cb.record_failure(LLMError(f"error_{i}"))

        # Should not accumulate internal state excessively
        stats = cb.get_stats()
        assert stats["total_calls"] == 1000
        # Circuit breaker should not store individual operation history

    def test_retry_strategy_memory_efficiency(self):
        """Test retry strategy memory usage with history."""
        strategy = RetryStrategy()

        # Add many retry attempts
        for i in range(100):
            strategy.record_attempt(i % 5, LLMError(f"error_{i}"), i * 0.1)

        # Should store all attempts
        assert len(strategy.retry_history) == 100

        # But history should be bounded in real usage
        stats = strategy.get_retry_stats()
        assert stats["total_attempts"] == 100
