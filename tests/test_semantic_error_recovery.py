import pytest
import asyncio
import time
from unittest.mock import AsyncMock

from codedocsync.matcher.semantic_error_recovery import (
    SemanticErrorRecovery,
    ErrorType,
    RecoveryAction,
    ErrorAnalysis,
    CircuitBreaker,
    CircuitBreakerOpenError,
    with_api_fallback,
    with_resilient_retry,
)
from codedocsync.parser import ParsedFunction, FunctionSignature
from codedocsync.matcher.semantic_models import FunctionEmbedding


class TestSemanticErrorRecovery:
    """Test suite for SemanticErrorRecovery class."""

    @pytest.fixture
    def error_recovery(self):
        """Create SemanticErrorRecovery instance for testing."""
        return SemanticErrorRecovery(max_retries=2, base_wait_time=0.1)

    @pytest.fixture
    def sample_function(self):
        """Create sample ParsedFunction for testing."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="test_function", parameters=[], return_type="None"
            ),
            docstring=None,
            file_path="test.py",
            line_number=1,
            end_line_number=5,
            source_code="def test_function(): pass",
        )

    @pytest.fixture
    def sample_embedding(self):
        """Create sample FunctionEmbedding for testing."""
        return FunctionEmbedding(
            function_id="test.test_function",
            embedding=[0.1] * 384,
            model="test-model",
            text_embedded="def test_function(): pass",
            timestamp=time.time(),
            signature_hash="test_hash",
        )

    def test_error_analysis_creation(self):
        """Test ErrorAnalysis dataclass creation and validation."""
        analysis = ErrorAnalysis(
            error_type=ErrorType.RATE_LIMIT,
            action=RecoveryAction.WAIT,
            wait_seconds=60.0,
            message="Rate limit exceeded",
        )

        assert analysis.error_type == ErrorType.RATE_LIMIT
        assert analysis.action == RecoveryAction.WAIT
        assert analysis.wait_seconds == 60.0
        assert analysis.message == "Rate limit exceeded"
        assert analysis.metadata == {}

    def test_analyze_rate_limit_error(self, error_recovery):
        """Test analysis of rate limit errors."""
        error = Exception("Rate limit exceeded - too many requests")
        analysis = error_recovery.analyze_error(error)

        assert analysis.error_type == ErrorType.RATE_LIMIT
        assert analysis.action == RecoveryAction.WAIT
        assert analysis.wait_seconds == 60.0
        assert "rate limit" in analysis.message.lower()

    def test_analyze_auth_error(self, error_recovery):
        """Test analysis of authentication errors."""
        error = Exception("Invalid API key provided")
        analysis = error_recovery.analyze_error(error)

        assert analysis.error_type == ErrorType.AUTH_ERROR
        assert analysis.action == RecoveryAction.SKIP
        assert "authentication" in analysis.message.lower()

    def test_analyze_timeout_error(self, error_recovery):
        """Test analysis of timeout errors."""
        error = Exception("Request timed out after 30 seconds")
        analysis = error_recovery.analyze_error(error)

        assert analysis.error_type == ErrorType.TIMEOUT
        assert analysis.action == RecoveryAction.RETRY
        assert analysis.wait_seconds == 5.0

    def test_analyze_connection_error(self, error_recovery):
        """Test analysis of connection errors."""
        error = Exception("Connection failed - network unreachable")
        analysis = error_recovery.analyze_error(error)

        assert analysis.error_type == ErrorType.CONNECTION_ERROR
        assert analysis.action == RecoveryAction.RETRY
        assert analysis.wait_seconds == 3.0

    def test_analyze_memory_error(self, error_recovery):
        """Test analysis of memory errors."""
        error = MemoryError("Out of memory")
        analysis = error_recovery.analyze_error(error)

        assert analysis.error_type == ErrorType.MEMORY_ERROR
        assert analysis.action == RecoveryAction.ABORT

    def test_analyze_unknown_error(self, error_recovery):
        """Test analysis of unknown errors."""
        error = Exception("Some unexpected error")
        analysis = error_recovery.analyze_error(error)

        assert analysis.error_type == ErrorType.UNKNOWN
        assert analysis.action == RecoveryAction.SKIP

    @pytest.mark.asyncio
    async def test_with_embedding_fallback_success_primary(
        self, error_recovery, sample_function, sample_embedding
    ):
        """Test successful embedding generation with primary function."""
        primary_func = AsyncMock(return_value=sample_embedding)
        fallback_funcs = [AsyncMock()]

        result = await error_recovery.with_embedding_fallback(
            primary_func, fallback_funcs, sample_function
        )

        assert result == sample_embedding
        primary_func.assert_called_once_with(sample_function)
        # Fallbacks should not be called
        for fallback in fallback_funcs:
            fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_with_embedding_fallback_success_fallback(
        self, error_recovery, sample_function, sample_embedding
    ):
        """Test successful embedding generation with fallback function."""
        primary_func = AsyncMock(side_effect=Exception("Primary failed"))
        fallback1 = AsyncMock(side_effect=Exception("Fallback 1 failed"))
        fallback2 = AsyncMock(return_value=sample_embedding)
        fallback_funcs = [fallback1, fallback2]

        result = await error_recovery.with_embedding_fallback(
            primary_func, fallback_funcs, sample_function
        )

        assert result == sample_embedding
        primary_func.assert_called_once()
        fallback1.assert_called_once()
        fallback2.assert_called_once()
        assert error_recovery.error_stats["recoveries_successful"] == 1

    @pytest.mark.asyncio
    async def test_with_embedding_fallback_all_fail(
        self, error_recovery, sample_function
    ):
        """Test embedding generation when all functions fail."""
        primary_func = AsyncMock(side_effect=Exception("Primary failed"))
        fallback_funcs = [
            AsyncMock(side_effect=Exception("Fallback 1 failed")),
            AsyncMock(side_effect=Exception("Fallback 2 failed")),
        ]

        result = await error_recovery.with_embedding_fallback(
            primary_func, fallback_funcs, sample_function
        )

        assert result is None
        assert error_recovery.error_stats["operations_aborted"] == 1
        assert error_recovery.error_stats["total_errors"] == 3

    @pytest.mark.asyncio
    async def test_with_retry_success_first_attempt(self, error_recovery):
        """Test successful operation on first attempt."""
        operation = AsyncMock(return_value="success")

        result = await error_recovery.with_retry(operation, "test_operation")

        assert result == "success"
        operation.assert_called_once()
        assert error_recovery.error_stats["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_with_retry_success_after_retries(self, error_recovery):
        """Test successful operation after retries."""
        operation = AsyncMock(
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
                "success",
            ]
        )

        result = await error_recovery.with_retry(operation, "test_operation")

        assert result == "success"
        assert operation.call_count == 3
        assert error_recovery.error_stats["recoveries_successful"] == 1

    @pytest.mark.asyncio
    async def test_with_retry_all_attempts_fail(self, error_recovery):
        """Test operation failure after all retry attempts."""
        operation = AsyncMock(side_effect=Exception("Persistent failure"))

        with pytest.raises(Exception, match="Persistent failure"):
            await error_recovery.with_retry(operation, "test_operation")

        # max_retries=2, so 3 total attempts (initial + 2 retries)
        assert operation.call_count == 3
        assert error_recovery.error_stats["operations_aborted"] == 1

    @pytest.mark.asyncio
    async def test_with_retry_non_retryable_error(self, error_recovery):
        """Test that non-retryable errors are not retried."""
        # Memory error should not be retried
        operation = AsyncMock(side_effect=MemoryError("Out of memory"))

        with pytest.raises(MemoryError):
            await error_recovery.with_retry(operation, "test_operation")

        # Should only be called once (no retries)
        operation.assert_called_once()

    def test_error_stats_tracking(self, error_recovery):
        """Test error statistics tracking."""
        # Simulate some errors
        error_recovery.analyze_error(Exception("Rate limit exceeded"))
        error_recovery.analyze_error(Exception("Invalid API key"))
        error_recovery.analyze_error(Exception("Connection failed"))

        # Check stats are updated
        assert error_recovery.error_stats["total_errors"] == 3
        assert error_recovery.error_stats["errors_by_type"]["rate_limit"] == 1
        assert error_recovery.error_stats["errors_by_type"]["auth_error"] == 1
        assert error_recovery.error_stats["errors_by_type"]["connection_error"] == 1

    def test_get_error_summary(self, error_recovery):
        """Test error summary generation."""
        # Simulate some operations
        error_recovery.error_stats["recoveries_successful"] = 8
        error_recovery.error_stats["operations_aborted"] = 2
        error_recovery.error_stats["total_errors"] = 5
        error_recovery.error_stats["errors_by_type"]["rate_limit"] = 3
        error_recovery.error_stats["errors_by_type"]["timeout"] = 2

        summary = error_recovery.get_error_summary()

        assert summary["total_operations"] == 10
        assert summary["success_rate"] == 0.8
        assert summary["recoveries_successful"] == 8
        assert summary["operations_aborted"] == 2
        assert len(summary["most_common_errors"]) == 2
        assert summary["most_common_errors"][0]["type"] == "rate_limit"
        assert summary["most_common_errors"][0]["count"] == 3


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create CircuitBreaker instance for testing."""
        return CircuitBreaker(failure_threshold=3, recovery_time=1.0)

    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_operation(self, circuit_breaker):
        """Test circuit breaker with successful operations."""
        operation = AsyncMock(return_value="success")

        for _ in range(5):
            result = await circuit_breaker.call(operation)
            assert result == "success"

        assert not circuit_breaker.is_open
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.stats["successful_requests"] == 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""
        operation = AsyncMock(side_effect=Exception("Operation failed"))

        # First 3 failures should open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(operation)

            if i < 2:
                assert not circuit_breaker.is_open
            else:
                assert circuit_breaker.is_open

        # Next call should be rejected with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(operation)

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_after_recovery_time(self, circuit_breaker):
        """Test circuit breaker resets after recovery time."""
        operation = AsyncMock(
            side_effect=[
                Exception("Fail 1"),
                Exception("Fail 2"),
                Exception("Fail 3"),  # Opens circuit
                "success",  # After recovery time
            ]
        )

        # Cause circuit to open
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(operation)

        assert circuit_breaker.is_open

        # Wait for recovery time
        await asyncio.sleep(1.1)

        # Should succeed and close circuit
        result = await circuit_breaker.call(operation)
        assert result == "success"
        assert not circuit_breaker.is_open
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, circuit_breaker):
        """Test circuit breaker resets failure count on success."""
        operation = AsyncMock(
            side_effect=[
                Exception("Fail 1"),
                Exception("Fail 2"),
                "success",  # Should reset failure count
                Exception("Fail 3"),
                Exception("Fail 4"),
            ]
        )

        # Two failures
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(operation)

        assert circuit_breaker.failure_count == 2

        # Success should reset
        result = await circuit_breaker.call(operation)
        assert result == "success"
        assert circuit_breaker.failure_count == 0

        # Two more failures shouldn't open circuit yet
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(operation)

        assert not circuit_breaker.is_open  # Should still be closed

    def test_circuit_breaker_manual_control(self, circuit_breaker):
        """Test manual circuit breaker control."""
        # Force open
        circuit_breaker.force_open()
        assert circuit_breaker.is_open
        assert circuit_breaker.stats["circuit_opens"] == 1

        # Force close
        circuit_breaker.force_close()
        assert not circuit_breaker.is_open
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.stats["circuit_closes"] == 1

    def test_circuit_breaker_stats(self, circuit_breaker):
        """Test circuit breaker statistics."""
        initial_stats = circuit_breaker.get_stats()

        assert not initial_stats["is_open"]
        assert initial_stats["failure_count"] == 0
        assert initial_stats["total_requests"] == 0
        assert initial_stats["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_error_recovery(self):
        """Test circuit breaker integration with error recovery."""
        error_recovery = SemanticErrorRecovery()
        circuit_breaker = error_recovery.create_circuit_breaker(
            failure_threshold=2, recovery_time=0.5
        )

        operation = AsyncMock(side_effect=Exception("Test failure"))

        # Should fail twice then open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(operation)

        # Circuit should be open now
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(operation)

        assert circuit_breaker.is_open


class TestConvenienceFunctions:
    """Test convenience functions for error recovery."""

    @pytest.mark.asyncio
    async def test_with_api_fallback(self, sample_function, sample_embedding):
        """Test with_api_fallback convenience function."""
        primary_api = AsyncMock(side_effect=Exception("Primary failed"))
        fallback_api = AsyncMock(return_value=sample_embedding)

        result = await with_api_fallback(primary_api, [fallback_api], sample_function)

        assert result == sample_embedding

    @pytest.mark.asyncio
    async def test_with_resilient_retry(self):
        """Test with_resilient_retry convenience function."""
        operation = AsyncMock(side_effect=[Exception("First failure"), "success"])

        result = await with_resilient_retry(operation, "test_operation", max_retries=2)

        assert result == "success"
        assert operation.call_count == 2


class TestIntegrationScenarios:
    """Integration tests for complex error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_complex_fallback_with_different_errors(self, sample_function):
        """Test complex fallback scenario with different error types."""
        error_recovery = SemanticErrorRecovery()

        # Primary: Rate limit
        # Fallback 1: Auth error
        # Fallback 2: Success
        primary_func = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        fallback1 = AsyncMock(side_effect=Exception("Invalid API key"))
        fallback2 = AsyncMock(return_value="success")

        result = await error_recovery.with_embedding_fallback(
            primary_func, [fallback1, fallback2], sample_function
        )

        assert result == "success"
        assert error_recovery.error_stats["errors_by_type"]["rate_limit"] == 1
        assert error_recovery.error_stats["errors_by_type"]["auth_error"] == 1

    @pytest.mark.asyncio
    async def test_cascading_failures_with_circuit_breaker(self):
        """Test cascading failures protection with circuit breaker."""
        error_recovery = SemanticErrorRecovery()
        circuit_breaker = error_recovery.create_circuit_breaker(
            failure_threshold=2, recovery_time=0.1
        )

        failing_operation = AsyncMock(side_effect=Exception("Service down"))

        # Cause circuit to open
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)

        # Should now reject requests immediately
        start_time = time.time()
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_operation)
        duration = time.time() - start_time

        # Should be very fast (not waiting for operation)
        assert duration < 0.01

    @pytest.mark.asyncio
    async def test_memory_error_abort_behavior(self, sample_function):
        """Test that memory errors cause immediate abort."""
        error_recovery = SemanticErrorRecovery()

        primary_func = AsyncMock(side_effect=MemoryError("Out of memory"))
        fallback_funcs = [AsyncMock(return_value="should_not_be_called")]

        result = await error_recovery.with_embedding_fallback(
            primary_func, fallback_funcs, sample_function
        )

        assert result is None
        # Fallbacks should not be called due to abort
        for fallback in fallback_funcs:
            fallback.assert_not_called()

    @pytest.fixture
    def sample_function(self):
        """Create sample ParsedFunction for testing."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="test_function", parameters=[], return_type="None"
            ),
            docstring=None,
            file_path="test.py",
            line_number=1,
            end_line_number=5,
            source_code="def test_function(): pass",
        )

    @pytest.fixture
    def sample_embedding(self):
        """Create sample FunctionEmbedding for testing."""
        return FunctionEmbedding(
            function_id="test.test_function",
            embedding=[0.1] * 384,
            model="test-model",
            text_embedded="def test_function(): pass",
            timestamp=time.time(),
            signature_hash="test_hash",
        )
