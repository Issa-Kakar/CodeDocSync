"""
Performance monitoring for suggestion generation.

Tracks generation times, identifies bottlenecks, and provides
performance optimization recommendations.
"""

import logging
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# Performance thresholds (in seconds)
PERFORMANCE_THRESHOLDS = {
    "simple_suggestion": 0.05,  # 50ms
    "complex_suggestion": 0.2,  # 200ms
    "style_conversion": 0.1,  # 100ms
    "full_file_suggestions": 1.0,  # 1s
    "batch_processing": 5.0,  # 5s
    "template_render": 0.02,  # 20ms
    "style_detection": 0.01,  # 10ms
    "example_generation": 0.1,  # 100ms
}


@dataclass
class PerformanceMetric:
    """Single performance measurement."""

    operation: str
    duration: float
    timestamp: datetime
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exceeded_threshold(self) -> bool:
        """Check if operation exceeded its threshold."""
        threshold = PERFORMANCE_THRESHOLDS.get(self.operation, 1.0)
        return self.duration > threshold


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    operation: str
    count: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    p50_time: float  # Median
    p95_time: float  # 95th percentile
    p99_time: float  # 99th percentile
    success_rate: float
    threshold_violations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "count": self.count,
            "total_time": round(self.total_time, 3),
            "min_time": round(self.min_time, 3),
            "max_time": round(self.max_time, 3),
            "avg_time": round(self.avg_time, 3),
            "p50_time": round(self.p50_time, 3),
            "p95_time": round(self.p95_time, 3),
            "p99_time": round(self.p99_time, 3),
            "success_rate": round(self.success_rate, 2),
            "threshold_violations": self.threshold_violations,
        }


class SuggestionPerformanceMonitor:
    """Monitor suggestion generation performance."""

    def __init__(
        self,
        max_history: int = 1000,
        enable_detailed_logging: bool = False,
    ):
        """
        Initialize performance monitor.

        Args:
            max_history: Maximum metrics to keep per operation
            enable_detailed_logging: Enable detailed performance logging
        """
        self.max_history = max_history
        self.enable_detailed_logging = enable_detailed_logging
        self.metrics: dict[str, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._active_timers: dict[str, float] = {}

    @contextmanager
    def measure(self, operation: str, **metadata):
        """
        Measure operation performance.

        Args:
            operation: Operation name
            **metadata: Additional metadata to store

        Yields:
            None

        Example:
            with monitor.measure("parameter_suggestion", issue_type="missing"):
                # Generate suggestion
        """
        start_time = time.perf_counter()
        success = True

        # Track nested operations
        operation_key = f"{operation}:{id(metadata)}"
        self._active_timers[operation_key] = start_time

        try:
            yield
        except Exception as e:
            success = False
            metadata["error"] = str(e)
            raise
        finally:
            duration = time.perf_counter() - start_time
            del self._active_timers[operation_key]

            # Record metric
            metric = PerformanceMetric(
                operation=operation,
                duration=duration,
                timestamp=datetime.now(),
                success=success,
                metadata=metadata,
            )
            self.metrics[operation].append(metric)

            # Log if threshold exceeded or detailed logging enabled
            if metric.exceeded_threshold:
                logger.warning(
                    f"Performance threshold exceeded: {operation} "
                    f"took {duration:.2f}s (threshold: "
                    f"{PERFORMANCE_THRESHOLDS.get(operation, 1.0):.2f}s)"
                )
            elif self.enable_detailed_logging:
                logger.debug(f"Performance: {operation} took {duration:.3f}s")

    def get_stats(self, operation: str | None = None) -> dict[str, PerformanceStats]:
        """
        Get performance statistics.

        Args:
            operation: Specific operation to get stats for, or None for all

        Returns:
            Dictionary of operation names to performance stats
        """
        operations = [operation] if operation else list(self.metrics.keys())
        stats = {}

        for op in operations:
            metrics = list(self.metrics[op])
            if not metrics:
                continue

            # Calculate statistics
            durations = [m.duration for m in metrics]
            durations.sort()
            success_count = sum(1 for m in metrics if m.success)
            threshold_violations = sum(1 for m in metrics if m.exceeded_threshold)

            stats[op] = PerformanceStats(
                operation=op,
                count=len(metrics),
                total_time=sum(durations),
                min_time=min(durations),
                max_time=max(durations),
                avg_time=sum(durations) / len(durations),
                p50_time=self._percentile(durations, 50),
                p95_time=self._percentile(durations, 95),
                p99_time=self._percentile(durations, 99),
                success_rate=success_count / len(metrics) if metrics else 0,
                threshold_violations=threshold_violations,
            )

        return stats

    def get_slow_operations(
        self, limit: int = 10
    ) -> list[tuple[str, PerformanceMetric]]:
        """
        Get slowest operations.

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of (operation_name, metric) tuples
        """
        all_metrics = []
        for operation, metrics in self.metrics.items():
            for metric in metrics:
                all_metrics.append((operation, metric))

        # Sort by duration descending
        all_metrics.sort(key=lambda x: x[1].duration, reverse=True)
        return all_metrics[:limit]

    def get_bottlenecks(self) -> dict[str, str]:
        """
        Identify performance bottlenecks.

        Returns:
            Dictionary of bottleneck descriptions
        """
        bottlenecks = {}
        stats = self.get_stats()

        for operation, stat in stats.items():
            # High average time
            threshold = PERFORMANCE_THRESHOLDS.get(operation, 1.0)
            if stat.avg_time > threshold * 1.5:
                bottlenecks[operation] = (
                    f"Average time ({stat.avg_time:.2f}s) exceeds threshold "
                    f"by {((stat.avg_time / threshold) - 1) * 100:.0f}%"
                )

            # High variance
            elif stat.max_time > stat.avg_time * 5:
                bottlenecks[operation] = (
                    f"High variance: max time ({stat.max_time:.2f}s) is "
                    f"{stat.max_time / stat.avg_time:.1f}x average"
                )

            # Many threshold violations
            elif stat.threshold_violations > stat.count * 0.1:
                bottlenecks[operation] = (
                    f"{stat.threshold_violations} operations "
                    f"({(stat.threshold_violations / stat.count) * 100:.0f}%) "
                    f"exceeded threshold"
                )

        return bottlenecks

    def get_recommendations(self) -> list[str]:
        """
        Get performance optimization recommendations.

        Returns:
            List of recommendation strings
        """
        recommendations = []
        stats = self.get_stats()
        bottlenecks = self.get_bottlenecks()

        # Check for slow operations
        for operation, bottleneck in bottlenecks.items():
            if "style_detection" in operation:
                recommendations.append(
                    f"Style detection is slow ({bottleneck}). "
                    "Consider caching detected styles per file."
                )
            elif "template_render" in operation:
                recommendations.append(
                    f"Template rendering is slow ({bottleneck}). "
                    "Consider pre-compiling templates."
                )
            elif "example_generation" in operation:
                recommendations.append(
                    f"Example generation is slow ({bottleneck}). "
                    "Consider using simpler examples or caching."
                )

        # Check for operations with low success rates
        for operation, stat in stats.items():
            if stat.success_rate < 0.9:
                recommendations.append(
                    f"{operation} has low success rate ({stat.success_rate:.0%}). "
                    "Review error handling and input validation."
                )

        # Check for operations with high p99 times
        for operation, stat in stats.items():
            if stat.p99_time > stat.p50_time * 10:
                recommendations.append(
                    f"{operation} has high tail latency (p99={stat.p99_time:.2f}s, "
                    f"p50={stat.p50_time:.2f}s). Consider timeout or caching."
                )

        # General recommendations based on patterns
        if len(self.metrics) > 5 and all(
            stats[op].avg_time > 0.1 for op in stats if "simple" in op
        ):
            recommendations.append(
                "All operations are slow. Consider profiling for system-wide issues."
            )

        return recommendations

    def reset(self, operation: str | None = None) -> None:
        """
        Reset performance metrics.

        Args:
            operation: Specific operation to reset, or None for all
        """
        if operation:
            self.metrics[operation].clear()
        else:
            self.metrics.clear()

    def export_metrics(self) -> dict[str, Any]:
        """
        Export all metrics for analysis.

        Returns:
            Dictionary with metrics and statistics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "stats": {op: stat.to_dict() for op, stat in self.get_stats().items()},
            "bottlenecks": self.get_bottlenecks(),
            "recommendations": self.get_recommendations(),
            "slow_operations": [
                {
                    "operation": op,
                    "duration": metric.duration,
                    "timestamp": metric.timestamp.isoformat(),
                    "metadata": metric.metadata,
                }
                for op, metric in self.get_slow_operations()
            ],
        }

    @staticmethod
    def _percentile(sorted_list: list[float], percentile: int) -> float:
        """Calculate percentile from sorted list."""
        if not sorted_list:
            return 0.0

        index = (len(sorted_list) - 1) * percentile / 100
        lower = int(index)
        upper = lower + 1

        if upper >= len(sorted_list):
            return sorted_list[lower]

        weight = index - lower
        return sorted_list[lower] * (1 - weight) + sorted_list[upper] * weight


class PerformanceOptimizer:
    """Optimize suggestion generation based on performance data."""

    def __init__(self, monitor: SuggestionPerformanceMonitor):
        """
        Initialize optimizer.

        Args:
            monitor: Performance monitor to analyze
        """
        self.monitor = monitor

    def should_use_cache(self, operation: str) -> bool:
        """
        Determine if operation should use caching.

        Args:
            operation: Operation to check

        Returns:
            True if caching recommended
        """
        stats = self.monitor.get_stats(operation)
        if not stats or operation not in stats:
            return False

        stat = stats[operation]

        # Cache if slow or high variance
        return (
            stat.avg_time > 0.1  # Slower than 100ms
            or stat.max_time > stat.avg_time * 3  # High variance
            or stat.threshold_violations > stat.count * 0.2  # Many violations
        )

    def get_timeout(self, operation: str) -> float:
        """
        Get recommended timeout for operation.

        Args:
            operation: Operation to get timeout for

        Returns:
            Timeout in seconds
        """
        stats = self.monitor.get_stats(operation)
        if not stats or operation not in stats:
            # Default timeout
            return PERFORMANCE_THRESHOLDS.get(operation, 1.0) * 3

        stat = stats[operation]

        # Use p99 + buffer as timeout
        return min(stat.p99_time * 1.5, 10.0)  # Cap at 10 seconds

    def should_batch(self, operation: str, item_count: int) -> bool:
        """
        Determine if operations should be batched.

        Args:
            operation: Operation type
            item_count: Number of items to process

        Returns:
            True if batching recommended
        """
        stats = self.monitor.get_stats(operation)
        if not stats or operation not in stats:
            return item_count > 10

        stat = stats[operation]

        # Batch if total time would exceed threshold
        estimated_time = stat.avg_time * item_count
        return estimated_time > 1.0  # Batch if > 1 second


# Global performance monitor instance
_performance_monitor = SuggestionPerformanceMonitor()


def get_performance_monitor() -> SuggestionPerformanceMonitor:
    """Get global performance monitor instance."""
    return _performance_monitor


@contextmanager
def measure_performance(operation: str, **metadata):
    """
    Measure performance using global monitor.

    Args:
        operation: Operation name
        **metadata: Additional metadata

    Yields:
        None
    """
    with _performance_monitor.measure(operation, **metadata):
        yield
