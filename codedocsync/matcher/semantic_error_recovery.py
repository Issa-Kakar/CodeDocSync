import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..parser import ParsedFunction
from .semantic_models import FunctionEmbedding

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during semantic matching."""

    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    MODEL_ERROR = "model_error"
    VECTOR_STORE_ERROR = "vector_store_error"
    MEMORY_ERROR = "memory_error"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions for different error types."""

    WAIT = "wait"
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"


@dataclass
class ErrorAnalysis:
    """Analysis of an error with recommended recovery action."""

    error_type: ErrorType
    action: RecoveryAction
    wait_seconds: float = 0.0
    max_retries: int = 3
    message: str = ""
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticErrorRecovery:
    """
    Comprehensive error recovery for semantic matching operations.

    Handles API failures, connection issues, rate limits, and other
    external service problems with intelligent fallback strategies.
    """

    def __init__(self, max_retries: int = 3, base_wait_time: float = 1.0):
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time

        # Error tracking statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {error_type.value: 0 for error_type in ErrorType},
            "recoveries_successful": 0,
            "operations_aborted": 0,
            "total_wait_time": 0.0,
        }

    async def with_embedding_fallback(
        self,
        primary_func: Callable,
        fallback_funcs: list[Callable],
        function: ParsedFunction,
        *args,
        **kwargs,
    ) -> FunctionEmbedding | None:
        """
        Try multiple embedding providers with intelligent fallback.

        Args:
            primary_func: Primary embedding generation function
            fallback_funcs: List of fallback functions to try in order
            function: Function to generate embedding for
            *args, **kwargs: Additional arguments to pass to functions

        Returns:
            FunctionEmbedding if successful, None if all methods fail
        """
        all_funcs = [primary_func] + fallback_funcs

        for i, func in enumerate(all_funcs):
            is_primary = i == 0
            func_name = f"{'primary' if is_primary else f'fallback_{i}'}"

            try:
                logger.debug(
                    f"Trying {func_name} embedding generation for {function.signature.name}"
                )
                result = await func(function, *args, **kwargs)

                if result:
                    if not is_primary:
                        self.error_stats["recoveries_successful"] += 1
                        logger.info(
                            f"Successfully recovered using {func_name} for {function.signature.name}"
                        )
                    return result

            except Exception as error:
                self.error_stats["total_errors"] += 1

                # Analyze the error
                analysis = self.analyze_error(error)
                self.error_stats["errors_by_type"][analysis.error_type.value] += 1

                logger.warning(
                    f"{func_name} failed for {function.signature.name}: {analysis.message}"
                )

                # Apply recovery strategy
                if analysis.action == RecoveryAction.WAIT and analysis.wait_seconds > 0:
                    logger.info(f"Waiting {analysis.wait_seconds}s before continuing")
                    await asyncio.sleep(analysis.wait_seconds)
                    self.error_stats["total_wait_time"] += analysis.wait_seconds

                # Continue to next fallback unless this is critical
                if analysis.action == RecoveryAction.ABORT:
                    logger.error(
                        f"Critical error, aborting embedding generation: {analysis.message}"
                    )
                    break

        # All methods failed
        self.error_stats["operations_aborted"] += 1
        logger.error(f"All embedding methods failed for {function.signature.name}")
        return None

    def analyze_error(self, error: Exception) -> ErrorAnalysis:
        """
        Analyze an error and determine the appropriate recovery strategy.

        Args:
            error: The exception that occurred

        Returns:
            ErrorAnalysis with recommended recovery action
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Rate limiting errors
        if any(
            term in error_str
            for term in ["rate limit", "too many requests", "quota exceeded"]
        ):
            return ErrorAnalysis(
                error_type=ErrorType.RATE_LIMIT,
                action=RecoveryAction.WAIT,
                wait_seconds=60.0,  # Wait 1 minute for rate limits
                message="Rate limit exceeded, waiting before retry",
            )

        # Authentication errors
        if any(
            term in error_str
            for term in ["api key", "authentication", "unauthorized", "403", "401"]
        ):
            return ErrorAnalysis(
                error_type=ErrorType.AUTH_ERROR,
                action=RecoveryAction.SKIP,
                message="API authentication failed, check credentials",
            )

        # Timeout errors
        if any(term in error_str for term in ["timeout", "timed out", "read timeout"]):
            return ErrorAnalysis(
                error_type=ErrorType.TIMEOUT,
                action=RecoveryAction.RETRY,
                wait_seconds=5.0,
                message="Request timed out, retrying with backoff",
            )

        # Connection errors
        if any(
            term in error_str for term in ["connection", "network", "dns", "resolve"]
        ):
            return ErrorAnalysis(
                error_type=ErrorType.CONNECTION_ERROR,
                action=RecoveryAction.RETRY,
                wait_seconds=3.0,
                message="Connection error, retrying",
            )

        # Model-specific errors
        if any(term in error_str for term in ["model", "invalid", "not found", "404"]):
            return ErrorAnalysis(
                error_type=ErrorType.MODEL_ERROR,
                action=RecoveryAction.FALLBACK,
                message="Model error, trying fallback model",
            )

        # Vector store errors
        if any(
            term in error_str for term in ["chroma", "vector", "index", "embedding"]
        ):
            return ErrorAnalysis(
                error_type=ErrorType.VECTOR_STORE_ERROR,
                action=RecoveryAction.RETRY,
                wait_seconds=2.0,
                message="Vector store error, retrying",
            )

        # Memory errors - critical
        if "memory" in error_str or "memoryerror" in error_type_name:
            return ErrorAnalysis(
                error_type=ErrorType.MEMORY_ERROR,
                action=RecoveryAction.ABORT,
                message="Memory error, operation aborted",
            )

        # Unknown errors
        return ErrorAnalysis(
            error_type=ErrorType.UNKNOWN,
            action=RecoveryAction.SKIP,
            message=f"Unknown error: {error}",
        )

    async def with_retry(
        self, operation: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """
        Execute an operation with intelligent retry logic.

        Args:
            operation: Function to execute
            operation_name: Name for logging
            *args, **kwargs: Arguments to pass to operation

        Returns:
            Result of operation if successful

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying {operation_name} (attempt {attempt + 1}/{self.max_retries + 1})"
                    )

                result = await operation(*args, **kwargs)

                if attempt > 0:
                    self.error_stats["recoveries_successful"] += 1
                    logger.info(
                        f"Successfully recovered {operation_name} after {attempt} retries"
                    )

                return result

            except Exception as error:
                last_exception = error
                self.error_stats["total_errors"] += 1

                # Analyze error for retry strategy
                analysis = self.analyze_error(error)
                self.error_stats["errors_by_type"][analysis.error_type.value] += 1

                # Don't retry if action says to skip or abort
                if analysis.action in [RecoveryAction.SKIP, RecoveryAction.ABORT]:
                    logger.error(
                        f"{operation_name} failed with non-retryable error: {analysis.message}"
                    )
                    break

                # Don't retry on final attempt
                if attempt >= self.max_retries:
                    break

                # Wait before retry
                wait_time = self._calculate_backoff_time(attempt, analysis.wait_seconds)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    self.error_stats["total_wait_time"] += wait_time

        # All retries failed
        self.error_stats["operations_aborted"] += 1
        logger.error(f"{operation_name} failed after {self.max_retries} retries")
        raise last_exception

    def _calculate_backoff_time(self, attempt: int, base_wait: float = None) -> float:
        """Calculate exponential backoff time."""
        if base_wait is None:
            base_wait = self.base_wait_time

        # Exponential backoff with jitter
        backoff = base_wait * (2**attempt)

        # Add some randomness to avoid thundering herd
        import random

        jitter = random.uniform(0.5, 1.5)

        return min(backoff * jitter, 60.0)  # Cap at 60 seconds

    def create_circuit_breaker(
        self, failure_threshold: int = 5, recovery_time: float = 60.0
    ) -> "CircuitBreaker":
        """
        Create a circuit breaker for protecting against cascading failures.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_time: Time to wait before allowing requests again

        Returns:
            CircuitBreaker instance
        """
        return CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_time=recovery_time,
            error_recovery=self,
        )

    def get_error_summary(self) -> dict[str, Any]:
        """Get comprehensive error statistics."""
        total_operations = (
            self.error_stats["recoveries_successful"]
            + self.error_stats["operations_aborted"]
        )

        success_rate = (
            self.error_stats["recoveries_successful"] / total_operations
            if total_operations > 0
            else 0.0
        )

        return {
            "total_errors": self.error_stats["total_errors"],
            "total_operations": total_operations,
            "success_rate": success_rate,
            "error_breakdown": self.error_stats["errors_by_type"].copy(),
            "recoveries_successful": self.error_stats["recoveries_successful"],
            "operations_aborted": self.error_stats["operations_aborted"],
            "total_wait_time_seconds": self.error_stats["total_wait_time"],
            "most_common_errors": self._get_most_common_errors(),
        }

    def _get_most_common_errors(self) -> list[dict[str, Any]]:
        """Get the most common error types."""
        errors = [
            {"type": error_type, "count": count}
            for error_type, count in self.error_stats["errors_by_type"].items()
            if count > 0
        ]

        # Sort by count, descending
        errors.sort(key=lambda x: x["count"], reverse=True)

        return errors[:5]  # Top 5

    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {error_type.value: 0 for error_type in ErrorType},
            "recoveries_successful": 0,
            "operations_aborted": 0,
            "total_wait_time": 0.0,
        }


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for semantic matching operations.

    Prevents cascading failures by temporarily stopping requests to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_time: float = 60.0,
        error_recovery: SemanticErrorRecovery = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.error_recovery = error_recovery

        # Circuit state
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.is_open = False

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_opens": 0,
            "circuit_closes": 0,
        }

    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation through circuit breaker.

        Args:
            operation: Function to execute
            *args, **kwargs: Arguments for operation

        Returns:
            Result of operation

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If operation fails
        """
        self.stats["total_requests"] += 1

        # Check if circuit should close
        if self.is_open and self._should_attempt_reset():
            logger.info("Circuit breaker attempting reset")
            self.is_open = False
            self.failure_count = 0

        # Reject requests if circuit is open
        if self.is_open:
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open. "
                f"Failures: {self.failure_count}/{self.failure_threshold}. "
                f"Try again after {self.recovery_time}s."
            )

        try:
            # Execute operation
            result = await operation(*args, **kwargs)

            # Success - reset failure count
            if self.failure_count > 0:
                logger.info(
                    "Circuit breaker: Operation succeeded, resetting failure count"
                )
                self.failure_count = 0

            self.stats["successful_requests"] += 1
            return result

        except Exception:
            # Record failure
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.stats["failed_requests"] += 1

            # Open circuit if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                self.stats["circuit_opens"] += 1
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures. "
                    f"Will retry after {self.recovery_time}s."
                )

            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        return time.time() - self.last_failure_time >= self.recovery_time

    def force_open(self) -> None:
        """Manually open the circuit breaker."""
        self.is_open = True
        self.stats["circuit_opens"] += 1
        logger.warning("Circuit breaker manually opened")

    def force_close(self) -> None:
        """Manually close the circuit breaker."""
        self.is_open = False
        self.failure_count = 0
        self.stats["circuit_closes"] += 1
        logger.info("Circuit breaker manually closed")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = (
            self.stats["successful_requests"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0
            else 0.0
        )

        return {
            "is_open": self.is_open,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": success_rate,
            "circuit_opens": self.stats["circuit_opens"],
            "circuit_closes": self.stats["circuit_closes"],
            "time_since_last_failure": (
                time.time() - self.last_failure_time
                if self.last_failure_time > 0
                else 0
            ),
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    pass


# Convenience functions for common error recovery patterns


async def with_api_fallback(
    primary_api_call: Callable, fallback_api_calls: list[Callable], *args, **kwargs
) -> Any:
    """Convenience function for API calls with fallback."""
    recovery = SemanticErrorRecovery()
    return await recovery.with_embedding_fallback(
        primary_api_call, fallback_api_calls, *args, **kwargs
    )


async def with_resilient_retry(
    operation: Callable, operation_name: str, max_retries: int = 3, *args, **kwargs
) -> Any:
    """Convenience function for resilient operation retry."""
    recovery = SemanticErrorRecovery(max_retries=max_retries)
    return await recovery.with_retry(operation, operation_name, *args, **kwargs)
