"""Error handling and retry logic for LLM analyzer."""

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar


class LLMError(Exception):
    """Base class for LLM-related errors."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, retry_after)


class LLMTimeoutError(LLMError):
    """Request timed out."""

    pass


class LLMInvalidResponseError(LLMError):
    """Response doesn't match expected format."""

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


class LLMAPIKeyError(LLMError):
    """Invalid or missing API key."""

    pass


class LLMNetworkError(LLMError):
    """Network-related error."""

    pass


class LLMQuotaExceededError(LLMError):
    """API quota exceeded."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, retry_after)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""

    attempt: int
    error: Exception
    delay: float
    timestamp: float


class RetryStrategy:
    """Configurable retry logic for LLM calls."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry parameters.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_history: list[RetryAttempt] = []

    def should_retry(self, error: Exception, attempt: int) -> tuple[bool, float]:
        """Determine if should retry and calculate delay.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-based)

        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        if attempt >= self.max_retries:
            return False, 0.0

        # Determine retry eligibility based on error type
        if isinstance(error, LLMAPIKeyError):
            # Never retry invalid API key
            return False, 0.0

        if isinstance(error, LLMInvalidResponseError):
            # Don't retry malformed responses (likely prompt issue)
            return False, 0.0

        if isinstance(error, LLMRateLimitError):
            # Always retry rate limits with exponential backoff
            delay = self._calculate_delay(attempt, error.retry_after)
            return True, delay

        if isinstance(error, LLMTimeoutError):
            # Retry timeout once with no delay
            return attempt == 0, 0.0

        if isinstance(error, LLMNetworkError | LLMError):
            # Retry general LLM and network errors with backoff
            delay = self._calculate_delay(attempt)
            return True, delay

        # Unknown error - be conservative and retry
        delay = self._calculate_delay(attempt)
        return True, delay

    def _calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate delay for retry attempt.

        Args:
            attempt: Current attempt number (0-based)
            retry_after: Explicit retry delay from error (e.g., Retry-After header)

        Returns:
            Delay in seconds
        """
        if retry_after is not None:
            # Respect explicit retry delay
            delay = retry_after
        else:
            # Exponential backoff
            delay = self.base_delay * (self.exponential_base**attempt)

        # Cap at maximum delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)

    def record_attempt(self, attempt: int, error: Exception, delay: float) -> None:
        """Record a retry attempt for analysis."""
        self.retry_history.append(
            RetryAttempt(
                attempt=attempt, error=error, delay=delay, timestamp=time.time()
            )
        )

    def get_retry_stats(self) -> dict[str, Any]:
        """Get statistics about retry attempts."""
        if not self.retry_history:
            return {"total_attempts": 0}

        error_types: dict[str, int] = {}
        total_delay = 0.0

        for retry in self.retry_history:
            error_type = type(retry.error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            total_delay += retry.delay

        return {
            "total_attempts": len(self.retry_history),
            "error_types": error_types,
            "total_delay_seconds": total_delay,
            "average_delay": total_delay / len(self.retry_history),
        }


T = TypeVar("T")


class CircuitBreaker:
    """Prevent cascading failures using circuit breaker pattern."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] = LLMError,
        half_open_max_calls: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            expected_exception: Exception type to consider as failure
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.success_count = 0
        self.total_calls = 0

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        current_time = time.time()

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful execution."""
        self.total_calls += 1
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0

        if self.state == CircuitState.CLOSED and self.failure_count > 0:
            # Gradually recover from failures
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self, exception: Exception) -> None:
        """Record a failed execution."""
        self.total_calls += 1

        if isinstance(exception, self.expected_exception):
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.half_open_calls = 0
            elif (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.failure_threshold
            ):
                self.state = CircuitState.OPEN

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            LLMError: If circuit is open or function fails
        """
        if not self.can_execute():
            raise LLMError(f"Circuit breaker is {self.state.value}, rejecting request")

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self.record_success()
            return result

        except Exception as e:
            self.record_failure(e)
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "failure_rate": self.failure_rate,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0


async def with_retry(
    func: Callable[..., T],
    retry_strategy: RetryStrategy | None = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute function with retry logic.

    Args:
        func: Function to execute
        retry_strategy: Retry strategy to use
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        Function result

    Raises:
        Exception: Last exception if all retries fail
    """
    if retry_strategy is None:
        retry_strategy = RetryStrategy()

    last_exception = None

    for attempt in range(retry_strategy.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            should_retry, delay = retry_strategy.should_retry(e, attempt)
            retry_strategy.record_attempt(attempt, e, delay)

            if not should_retry:
                break

            if delay > 0:
                await asyncio.sleep(delay)

    # All retries exhausted
    if last_exception:
        raise last_exception
    else:
        raise LLMError("All retry attempts failed with no exception recorded")


def create_default_retry_strategy() -> RetryStrategy:
    """Create default retry strategy for LLM operations."""
    return RetryStrategy(
        max_retries=3, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=True
    )


def create_aggressive_retry_strategy() -> RetryStrategy:
    """Create aggressive retry strategy for critical operations."""
    return RetryStrategy(
        max_retries=5, base_delay=0.5, max_delay=30.0, exponential_base=1.5, jitter=True
    )


def create_conservative_retry_strategy() -> RetryStrategy:
    """Create conservative retry strategy to minimize load."""
    return RetryStrategy(
        max_retries=2,
        base_delay=2.0,
        max_delay=120.0,
        exponential_base=3.0,
        jitter=True,
    )
