"""
LLM Performance Monitoring - Chunk 4 Implementation

This module provides comprehensive performance monitoring for LLM operations:
- Response time percentiles (p50, p95, p99)
- Cache hit rate by analysis type
- Token usage tracking and cost estimation
- Error rate monitoring by error type
- Rate limiter queue depth tracking
- Performance alerting and recommendations

Performance Requirements:
- Monitoring overhead should be <1ms per operation
- Real-time metrics aggregation
- Alert if p95 response time > 2000ms
- Track cost metrics for budget management
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert notification."""

    level: AlertLevel
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics snapshot."""

    # Response time metrics
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    # Cache metrics
    overall_cache_hit_rate: float
    cache_hit_rate_by_type: dict[str, float]
    cache_efficiency_score: float

    # Token and cost metrics
    total_tokens_used: int
    avg_tokens_per_request: float
    estimated_cost_usd: float
    cost_per_analysis: float

    # Error metrics
    error_rate: float
    error_rate_by_type: dict[str, float]

    # Rate limiting metrics
    avg_queue_depth: float
    max_queue_depth: int
    rate_limit_hits: int

    # Throughput metrics
    requests_per_second: float
    successful_requests: int
    failed_requests: int

    # Performance scores
    overall_performance_score: float
    recommendations: list[str]


class LLMPerformanceMonitor:
    """
    Track LLM performance metrics with real-time monitoring and alerting.

    Provides comprehensive monitoring of LLM operations including:
    - Response time tracking with percentile calculations
    - Cache performance monitoring by analysis type
    - Token usage and cost tracking
    - Error rate monitoring with categorization
    - Rate limiter performance tracking
    - Automated alerting for performance degradation
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of recent requests to keep for metrics calculation
            alert_thresholds: Custom alert thresholds for different metrics
        """
        self.window_size = window_size
        self._lock = threading.Lock()

        # Response time tracking (sliding window)
        self.response_times = deque(maxlen=window_size)

        # Cache performance tracking
        self.cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0})
        self.total_cache_hits = 0
        self.total_cache_misses = 0

        # Token usage tracking
        self.token_usage = deque(maxlen=window_size)
        self.total_tokens = 0

        # Error tracking
        self.error_counts = defaultdict(int)
        self.total_requests = 0
        self.successful_requests = 0

        # Rate limiter tracking
        self.queue_depths = deque(maxlen=window_size)
        self.rate_limit_hits = 0

        # Cost tracking (OpenAI pricing as of 2024)
        self.token_costs = {
            "gpt-4o-mini": {
                "input": 0.00015 / 1000,
                "output": 0.0006 / 1000,
            },  # per token
            "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
            "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
        }

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "p95_response_time_ms": 2000.0,
            "error_rate": 0.05,  # 5%
            "cache_hit_rate": 0.8,  # 80%
            "avg_queue_depth": 10.0,
            "cost_per_analysis": 0.01,  # $0.01 per analysis
        }

        # Alert tracking
        self.alerts = deque(maxlen=100)
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 300  # 5 minutes

        # Start time for throughput calculation
        self.start_time = time.time()

        # Performance recommendations
        self.recommendations = []

    def record_request(
        self,
        analysis_type: str,
        response_time_ms: float,
        tokens_used: int,
        cache_hit: bool,
        model: str = "gpt-4o-mini",
        error: str | None = None,
        queue_depth: int = 0,
    ) -> None:
        """
        Record performance metrics for a single request.

        Args:
            analysis_type: Type of analysis performed
            response_time_ms: Total response time in milliseconds
            tokens_used: Number of tokens consumed
            cache_hit: Whether the request was served from cache
            model: LLM model used
            error: Error type if request failed
            queue_depth: Current rate limiter queue depth
        """
        with self._lock:
            self.total_requests += 1

            # Record response time
            self.response_times.append(response_time_ms)

            # Record cache performance
            if cache_hit:
                self.cache_stats[analysis_type]["hits"] += 1
                self.total_cache_hits += 1
            else:
                self.cache_stats[analysis_type]["misses"] += 1
                self.total_cache_misses += 1

            # Record token usage
            if tokens_used > 0:
                self.token_usage.append(
                    {
                        "tokens": tokens_used,
                        "model": model,
                        "analysis_type": analysis_type,
                        "cache_hit": cache_hit,
                    }
                )
                self.total_tokens += tokens_used

            # Record errors
            if error:
                self.error_counts[error] += 1
            else:
                self.successful_requests += 1

            # Record queue depth
            self.queue_depths.append(queue_depth)

            # Check for alerts
            self._check_alerts()

    def record_rate_limit_hit(self) -> None:
        """Record a rate limit hit."""
        with self._lock:
            self.rate_limit_hits += 1

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics.

        Returns:
            Current performance metrics snapshot
        """
        with self._lock:
            # Calculate response time percentiles
            response_times_list = list(self.response_times)
            if response_times_list:
                avg_response_time = statistics.mean(response_times_list)
                p50_response_time = statistics.median(response_times_list)
                p95_response_time = self._percentile(response_times_list, 95)
                p99_response_time = self._percentile(response_times_list, 99)
            else:
                avg_response_time = p50_response_time = p95_response_time = (
                    p99_response_time
                ) = 0.0

            # Calculate cache hit rates
            total_cache_requests = self.total_cache_hits + self.total_cache_misses
            overall_cache_hit_rate = (
                self.total_cache_hits / total_cache_requests
                if total_cache_requests > 0
                else 0.0
            )

            cache_hit_rate_by_type = {}
            for analysis_type, stats in self.cache_stats.items():
                total = stats["hits"] + stats["misses"]
                hit_rate = stats["hits"] / total if total > 0 else 0.0
                cache_hit_rate_by_type[analysis_type] = hit_rate

            # Calculate token metrics
            token_usage_list = list(self.token_usage)
            avg_tokens_per_request = (
                statistics.mean([t["tokens"] for t in token_usage_list])
                if token_usage_list
                else 0.0
            )

            # Estimate cost
            estimated_cost = self._calculate_estimated_cost()
            cost_per_analysis = estimated_cost / max(self.total_requests, 1)

            # Calculate error rate
            error_rate = (
                (self.total_requests - self.successful_requests) / self.total_requests
                if self.total_requests > 0
                else 0.0
            )

            error_rate_by_type = {}
            for error_type, count in self.error_counts.items():
                error_rate_by_type[error_type] = count / self.total_requests

            # Calculate queue metrics
            queue_depths_list = list(self.queue_depths)
            avg_queue_depth = (
                statistics.mean(queue_depths_list) if queue_depths_list else 0.0
            )
            max_queue_depth = max(queue_depths_list) if queue_depths_list else 0

            # Calculate throughput
            elapsed_time = time.time() - self.start_time
            requests_per_second = (
                self.total_requests / elapsed_time if elapsed_time > 0 else 0.0
            )

            # Calculate performance scores
            cache_efficiency_score = self._calculate_cache_efficiency()
            overall_performance_score = self._calculate_overall_performance_score(
                avg_response_time, overall_cache_hit_rate, error_rate
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                p95_response_time, overall_cache_hit_rate, error_rate, avg_queue_depth
            )

            return PerformanceMetrics(
                avg_response_time_ms=avg_response_time,
                p50_response_time_ms=p50_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                overall_cache_hit_rate=overall_cache_hit_rate,
                cache_hit_rate_by_type=cache_hit_rate_by_type,
                cache_efficiency_score=cache_efficiency_score,
                total_tokens_used=self.total_tokens,
                avg_tokens_per_request=avg_tokens_per_request,
                estimated_cost_usd=estimated_cost,
                cost_per_analysis=cost_per_analysis,
                error_rate=error_rate,
                error_rate_by_type=error_rate_by_type,
                avg_queue_depth=avg_queue_depth,
                max_queue_depth=max_queue_depth,
                rate_limit_hits=self.rate_limit_hits,
                requests_per_second=requests_per_second,
                successful_requests=self.successful_requests,
                failed_requests=self.total_requests - self.successful_requests,
                overall_performance_score=overall_performance_score,
                recommendations=recommendations,
            )

    def get_alerts(self, since: float | None = None) -> list[PerformanceAlert]:
        """
        Get performance alerts.

        Args:
            since: Unix timestamp to get alerts since (optional)

        Returns:
            List of performance alerts
        """
        if since is None:
            return list(self.alerts)

        return [alert for alert in self.alerts if alert.timestamp >= since]

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self._lock:
            self.response_times.clear()
            self.cache_stats.clear()
            self.token_usage.clear()
            self.error_counts.clear()
            self.queue_depths.clear()

            self.total_cache_hits = 0
            self.total_cache_misses = 0
            self.total_tokens = 0
            self.total_requests = 0
            self.successful_requests = 0
            self.rate_limit_hits = 0

            self.alerts.clear()
            self.last_alert_time.clear()
            self.start_time = time.time()

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile from sorted data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]

        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        weight = index - lower_index

        return (
            sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
        )

    def _calculate_estimated_cost(self) -> float:
        """Calculate estimated cost based on token usage."""
        total_cost = 0.0

        for usage in self.token_usage:
            model = usage["model"]
            tokens = usage["tokens"]

            if model in self.token_costs:
                # Assume input/output split is roughly 70/30
                input_tokens = int(tokens * 0.7)
                output_tokens = int(tokens * 0.3)

                cost = (
                    input_tokens * self.token_costs[model]["input"]
                    + output_tokens * self.token_costs[model]["output"]
                )
                total_cost += cost

        return total_cost

    def _calculate_cache_efficiency(self) -> float:
        """Calculate cache efficiency score (0.0 to 1.0)."""
        if self.total_cache_hits + self.total_cache_misses == 0:
            return 0.0

        hit_rate = self.total_cache_hits / (
            self.total_cache_hits + self.total_cache_misses
        )

        # Factor in response time improvement from cache
        cache_response_times = [
            rt
            for rt, usage in zip(self.response_times, self.token_usage, strict=False)
            if usage.get("cache_hit", False)
        ]
        non_cache_response_times = [
            rt
            for rt, usage in zip(self.response_times, self.token_usage, strict=False)
            if not usage.get("cache_hit", False)
        ]

        if cache_response_times and non_cache_response_times:
            cache_avg = statistics.mean(cache_response_times)
            non_cache_avg = statistics.mean(non_cache_response_times)
            speed_improvement = max(0, (non_cache_avg - cache_avg) / non_cache_avg)
        else:
            speed_improvement = 0.5  # Assume 50% improvement

        return hit_rate * 0.7 + speed_improvement * 0.3

    def _calculate_overall_performance_score(
        self, avg_response_time: float, cache_hit_rate: float, error_rate: float
    ) -> float:
        """Calculate overall performance score (0.0 to 1.0)."""
        # Response time score (inverse relationship)
        response_score = max(0, 1.0 - avg_response_time / 2000)  # 2s baseline

        # Cache score
        cache_score = cache_hit_rate

        # Error score (inverse relationship)
        error_score = max(0, 1.0 - error_rate / 0.1)  # 10% baseline

        # Weighted average
        return response_score * 0.4 + cache_score * 0.3 + error_score * 0.3

    def _generate_recommendations(
        self,
        p95_response_time: float,
        cache_hit_rate: float,
        error_rate: float,
        avg_queue_depth: float,
    ) -> list[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        if p95_response_time > 2000:
            recommendations.append(
                "High response times detected. Consider enabling cache warming "
                "or using a faster model like gpt-3.5-turbo for simple analyses."
            )

        if cache_hit_rate < 0.8:
            recommendations.append(
                "Low cache hit rate. Consider implementing cache warming for "
                "frequently analyzed functions or increasing cache TTL."
            )

        if error_rate > 0.05:
            recommendations.append(
                "High error rate detected. Check API key validity, network "
                "connectivity, and implement better retry logic."
            )

        if avg_queue_depth > 5:
            recommendations.append(
                "High queue depth suggests rate limiting. Consider increasing "
                "requests_per_second or implementing smarter batching."
            )

        if not recommendations:
            recommendations.append("Performance is within acceptable thresholds.")

        return recommendations

    def _check_alerts(self) -> None:
        """Check if any metrics exceed alert thresholds."""
        current_time = time.time()

        # Check p95 response time
        if len(self.response_times) >= 20:  # Need enough data
            p95_time = self._percentile(list(self.response_times), 95)
            self._check_threshold_alert(
                "p95_response_time_ms", p95_time, AlertLevel.WARNING, current_time
            )

        # Check error rate
        if self.total_requests >= 10:
            error_rate = (
                self.total_requests - self.successful_requests
            ) / self.total_requests
            self._check_threshold_alert(
                "error_rate", error_rate, AlertLevel.CRITICAL, current_time
            )

        # Check cache hit rate
        total_cache_requests = self.total_cache_hits + self.total_cache_misses
        if total_cache_requests >= 10:
            hit_rate = self.total_cache_hits / total_cache_requests
            if hit_rate < self.alert_thresholds["cache_hit_rate"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    "cache_hit_rate",
                    hit_rate,
                    self.alert_thresholds["cache_hit_rate"],
                    f"Cache hit rate {hit_rate:.1%} below threshold",
                    current_time,
                )

    def _check_threshold_alert(
        self, metric: str, value: float, level: AlertLevel, current_time: float
    ) -> None:
        """Check if a metric exceeds its threshold and create alert if needed."""
        if metric not in self.alert_thresholds:
            return

        threshold = self.alert_thresholds[metric]

        # Check cooldown period
        if current_time - self.last_alert_time[metric] < self.alert_cooldown:
            return

        if value > threshold:
            message = f"{metric} {value:.2f} exceeds threshold {threshold:.2f}"
            self._create_alert(level, metric, value, threshold, message, current_time)

    def _create_alert(
        self,
        level: AlertLevel,
        metric: str,
        value: float,
        threshold: float,
        message: str,
        timestamp: float,
    ) -> None:
        """Create and store a performance alert."""
        alert = PerformanceAlert(
            level=level,
            metric=metric,
            current_value=value,
            threshold=threshold,
            message=message,
            timestamp=timestamp,
        )

        self.alerts.append(alert)
        self.last_alert_time[metric] = timestamp

        # Log the alert
        log_level = logging.WARNING if level == AlertLevel.WARNING else logging.ERROR
        logger.log(log_level, f"Performance Alert: {message}")


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> LLMPerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = LLMPerformanceMonitor()
    return _performance_monitor


def reset_performance_monitor() -> None:
    """Reset the global performance monitor."""
    global _performance_monitor
    _performance_monitor = None
