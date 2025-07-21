"""
Tests for LLM Performance Monitoring - Chunk 4 Implementation

Comprehensive test coverage for:
- Performance metrics tracking and calculation
- Response time percentiles (p50, p95, p99)
- Cache hit rate monitoring by analysis type
- Token usage and cost estimation
- Error rate monitoring with categorization
- Performance alerting and recommendations
"""

import time

import pytest

from codedocsync.analyzer.llm_performance import (
    AlertLevel,
    LLMPerformanceMonitor,
    PerformanceAlert,
    PerformanceMetrics,
    get_performance_monitor,
    reset_performance_monitor,
)


@pytest.fixture
def performance_monitor():
    """Create a fresh performance monitor for testing."""
    return LLMPerformanceMonitor(window_size=100)


@pytest.fixture
def monitor_with_data(performance_monitor):
    """Create a performance monitor with sample data."""
    # Add sample response times
    for i in range(50):
        response_time = 100 + (i * 10)  # 100ms to 590ms
        performance_monitor.record_request(
            analysis_type="behavior",
            response_time_ms=response_time,
            tokens_used=50 + i,
            cache_hit=(i % 3 == 0),  # Every 3rd request is cache hit
            model="gpt-4o-mini",
            error=None if i < 45 else "timeout",  # 5 errors
            queue_depth=i % 10,
        )

    return performance_monitor


class TestLLMPerformanceMonitor:
    """Test LLM performance monitor functionality."""

    def test_monitor_initialization(self):
        """Test performance monitor initializes correctly."""
        monitor = LLMPerformanceMonitor(window_size=500)

        assert monitor.window_size == 500
        assert len(monitor.response_times) == 0
        assert monitor.total_cache_hits == 0
        assert monitor.total_cache_misses == 0
        assert monitor.total_tokens == 0
        assert monitor.total_requests == 0
        assert monitor.successful_requests == 0

    def test_record_successful_request(self, performance_monitor):
        """Test recording a successful request."""
        performance_monitor.record_request(
            analysis_type="behavior",
            response_time_ms=150.0,
            tokens_used=75,
            cache_hit=False,
            model="gpt-4o-mini",
            error=None,
            queue_depth=2,
        )

        assert performance_monitor.total_requests == 1
        assert performance_monitor.successful_requests == 1
        assert performance_monitor.total_cache_misses == 1
        assert performance_monitor.total_tokens == 75
        assert len(performance_monitor.response_times) == 1
        assert performance_monitor.response_times[0] == 150.0

    def test_record_cache_hit(self, performance_monitor):
        """Test recording a cache hit."""
        performance_monitor.record_request(
            analysis_type="examples",
            response_time_ms=25.0,
            tokens_used=0,  # No tokens used for cache hit
            cache_hit=True,
            model="gpt-4o-mini",
        )

        assert performance_monitor.total_cache_hits == 1
        assert performance_monitor.cache_stats["examples"]["hits"] == 1
        assert performance_monitor.total_tokens == 0

    def test_record_error_request(self, performance_monitor):
        """Test recording a request with error."""
        performance_monitor.record_request(
            analysis_type="behavior",
            response_time_ms=5000.0,  # Long timeout
            tokens_used=0,
            cache_hit=False,
            error="timeout",
        )

        assert performance_monitor.total_requests == 1
        assert performance_monitor.successful_requests == 0
        assert performance_monitor.error_counts["timeout"] == 1

    def test_record_rate_limit_hit(self, performance_monitor):
        """Test recording rate limit hits."""
        performance_monitor.record_rate_limit_hit()
        performance_monitor.record_rate_limit_hit()

        assert performance_monitor.rate_limit_hits == 2

    def test_get_metrics_empty(self, performance_monitor):
        """Test getting metrics with no data."""
        metrics = performance_monitor.get_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.overall_cache_hit_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.total_tokens_used == 0
        assert metrics.successful_requests == 0

    def test_get_metrics_with_data(self, monitor_with_data):
        """Test getting metrics with sample data."""
        metrics = monitor_with_data.get_metrics()

        # Check response time metrics
        assert metrics.avg_response_time_ms > 0
        assert metrics.p50_response_time_ms > 0
        assert metrics.p95_response_time_ms > metrics.p50_response_time_ms
        assert metrics.p99_response_time_ms >= metrics.p95_response_time_ms

        # Check cache metrics
        assert 0 < metrics.overall_cache_hit_rate < 1
        assert "behavior" in metrics.cache_hit_rate_by_type

        # Check token metrics
        assert metrics.total_tokens_used > 0
        assert metrics.avg_tokens_per_request > 0

        # Check error metrics
        assert metrics.error_rate > 0  # We added some errors
        assert "timeout" in metrics.error_rate_by_type

        # Check cost estimation
        assert metrics.estimated_cost_usd > 0
        assert metrics.cost_per_analysis > 0

    def test_percentile_calculation(self, performance_monitor):
        """Test percentile calculation accuracy."""
        # Add known data points
        response_times = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for rt in response_times:
            performance_monitor.record_request(
                analysis_type="behavior",
                response_time_ms=rt,
                tokens_used=50,
                cache_hit=False,
            )

        metrics = performance_monitor.get_metrics()

        # Test percentile calculations
        assert metrics.p50_response_time_ms == 550.0  # Median of 10 values
        assert metrics.p95_response_time_ms == 950.0  # 95th percentile
        assert metrics.p99_response_time_ms == 990.0  # 99th percentile

    def test_cache_hit_rate_by_type(self, performance_monitor):
        """Test cache hit rate calculation by analysis type."""
        # Add data for different analysis types
        for _ in range(5):
            performance_monitor.record_request("behavior", 100, 50, cache_hit=True)
            performance_monitor.record_request("behavior", 100, 50, cache_hit=False)
            performance_monitor.record_request("examples", 100, 50, cache_hit=True)

        metrics = performance_monitor.get_metrics()

        assert metrics.cache_hit_rate_by_type["behavior"] == 0.5  # 50% hit rate
        assert metrics.cache_hit_rate_by_type["examples"] == 1.0  # 100% hit rate

    def test_cost_calculation(self, performance_monitor):
        """Test token cost calculation."""
        # Add requests with different models
        performance_monitor.record_request(
            "behavior", 100, 100, False, model="gpt-4o-mini"
        )
        performance_monitor.record_request("behavior", 100, 100, False, model="gpt-4o")

        metrics = performance_monitor.get_metrics()

        # Should have calculated costs for both models
        assert metrics.estimated_cost_usd > 0
        assert metrics.cost_per_analysis > 0

    def test_window_size_limit(self):
        """Test that window size is respected."""
        monitor = LLMPerformanceMonitor(window_size=5)

        # Add more than window size
        for i in range(10):
            monitor.record_request("behavior", 100 + i, 50, False)

        # Should only keep last 5 response times
        assert len(monitor.response_times) == 5
        assert list(monitor.response_times) == [105, 106, 107, 108, 109]

    def test_reset_metrics(self, monitor_with_data):
        """Test metrics reset functionality."""
        # Verify we have data
        assert monitor_with_data.total_requests > 0
        assert len(monitor_with_data.response_times) > 0

        # Reset metrics
        monitor_with_data.reset_metrics()

        # Verify reset
        assert monitor_with_data.total_requests == 0
        assert monitor_with_data.successful_requests == 0
        assert len(monitor_with_data.response_times) == 0
        assert monitor_with_data.total_cache_hits == 0
        assert monitor_with_data.total_tokens == 0


class TestPerformanceAlerts:
    """Test performance alerting functionality."""

    def test_alert_creation(self):
        """Test performance alert creation."""
        alert = PerformanceAlert(
            level=AlertLevel.WARNING,
            metric="p95_response_time_ms",
            current_value=2500.0,
            threshold=2000.0,
            message="Response time exceeded threshold",
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.metric == "p95_response_time_ms"
        assert alert.current_value == 2500.0
        assert alert.threshold == 2000.0
        assert "threshold" in alert.message

    def test_p95_response_time_alert(self):
        """Test alert generation for high p95 response time."""
        monitor = LLMPerformanceMonitor(
            alert_thresholds={"p95_response_time_ms": 500.0}
        )

        # Add data that will trigger alert
        for i in range(25):  # Need enough data for p95 calculation
            response_time = 600.0 if i >= 20 else 100.0  # Last 5 are high
            monitor.record_request("behavior", response_time, 50, False)

        alerts = monitor.get_alerts()

        # Should have generated alert for high p95 response time
        p95_alerts = [a for a in alerts if a.metric == "p95_response_time_ms"]
        assert len(p95_alerts) > 0
        assert p95_alerts[0].level == AlertLevel.WARNING

    def test_error_rate_alert(self):
        """Test alert generation for high error rate."""
        monitor = LLMPerformanceMonitor(
            alert_thresholds={"error_rate": 0.1}  # 10% threshold
        )

        # Add requests with high error rate
        for i in range(20):
            error = "api_error" if i >= 17 else None  # 15% error rate
            monitor.record_request("behavior", 100, 50, False, error=error)

        alerts = monitor.get_alerts()

        # Should have generated error rate alert
        error_alerts = [a for a in alerts if a.metric == "error_rate"]
        assert len(error_alerts) > 0
        assert error_alerts[0].level == AlertLevel.CRITICAL

    def test_cache_hit_rate_alert(self):
        """Test alert generation for low cache hit rate."""
        monitor = LLMPerformanceMonitor(
            alert_thresholds={"cache_hit_rate": 0.8}  # 80% threshold
        )

        # Add requests with low cache hit rate
        for i in range(20):
            cache_hit = i < 10  # 50% hit rate
            monitor.record_request("behavior", 100, 50, cache_hit)

        alerts = monitor.get_alerts()

        # Should have generated cache hit rate alert
        cache_alerts = [a for a in alerts if a.metric == "cache_hit_rate"]
        assert len(cache_alerts) > 0
        assert cache_alerts[0].level == AlertLevel.WARNING

    def test_alert_cooldown(self):
        """Test alert cooldown prevents spam."""
        monitor = LLMPerformanceMonitor(alert_thresholds={"error_rate": 0.1})
        monitor.alert_cooldown = 1  # 1 second cooldown

        # Trigger alert
        for _i in range(15):
            monitor.record_request("behavior", 100, 50, False, error="test_error")

        initial_alert_count = len(monitor.get_alerts())

        # Trigger more errors immediately (should not generate new alerts due to cooldown)
        for _i in range(5):
            monitor.record_request("behavior", 100, 50, False, error="test_error")

        # Should not have new alerts due to cooldown
        assert len(monitor.get_alerts()) == initial_alert_count

    def test_get_alerts_since(self, performance_monitor):
        """Test getting alerts since a specific timestamp."""
        start_time = time.time()

        # Generate some alerts
        for _i in range(15):
            performance_monitor.record_request(
                "behavior", 100, 50, False, error="test_error"
            )

        # Get alerts since start time
        recent_alerts = performance_monitor.get_alerts(since=start_time)
        all_alerts = performance_monitor.get_alerts()

        assert len(recent_alerts) <= len(all_alerts)
        for alert in recent_alerts:
            assert alert.timestamp >= start_time


class TestPerformanceRecommendations:
    """Test performance recommendation generation."""

    def test_high_response_time_recommendation(self):
        """Test recommendation for high response times."""
        monitor = LLMPerformanceMonitor()

        # Add data with high response times
        for _i in range(20):
            monitor.record_request("behavior", 3000, 100, False)  # 3s response time

        metrics = monitor.get_metrics()

        # Should include response time recommendation
        recommendations = metrics.recommendations
        assert any("response times" in rec.lower() for rec in recommendations)
        assert any(
            "cache warming" in rec.lower() or "faster model" in rec.lower()
            for rec in recommendations
        )

    def test_low_cache_hit_rate_recommendation(self):
        """Test recommendation for low cache hit rate."""
        monitor = LLMPerformanceMonitor()

        # Add data with low cache hit rate
        for _i in range(20):
            monitor.record_request("behavior", 100, 50, cache_hit=False)

        metrics = monitor.get_metrics()

        # Should include cache recommendation
        recommendations = metrics.recommendations
        assert any("cache" in rec.lower() for rec in recommendations)

    def test_high_error_rate_recommendation(self):
        """Test recommendation for high error rate."""
        monitor = LLMPerformanceMonitor()

        # Add data with high error rate
        for i in range(20):
            error = "api_error" if i >= 15 else None
            monitor.record_request("behavior", 100, 50, False, error=error)

        metrics = monitor.get_metrics()

        # Should include error recommendation
        recommendations = metrics.recommendations
        assert any("error" in rec.lower() for rec in recommendations)
        assert any(
            "api key" in rec.lower() or "retry" in rec.lower()
            for rec in recommendations
        )

    def test_good_performance_recommendation(self):
        """Test recommendation when performance is good."""
        monitor = LLMPerformanceMonitor()

        # Add data with good performance
        for _i in range(20):
            monitor.record_request("behavior", 100, 50, cache_hit=True)  # Fast, cached

        metrics = monitor.get_metrics()

        # Should indicate good performance
        recommendations = metrics.recommendations
        assert any(
            "acceptable" in rec.lower() or "good" in rec.lower()
            for rec in recommendations
        )


class TestPerformanceScoring:
    """Test performance scoring calculations."""

    def test_cache_efficiency_score(self, performance_monitor):
        """Test cache efficiency score calculation."""
        # Add mixed cache performance data
        for i in range(20):
            cache_hit = i % 2 == 0  # 50% hit rate
            response_time = 50 if cache_hit else 200  # Cache hits are faster
            performance_monitor.record_request("behavior", response_time, 50, cache_hit)

        metrics = performance_monitor.get_metrics()

        # Should have reasonable efficiency score
        assert 0 <= metrics.cache_efficiency_score <= 1
        assert (
            metrics.cache_efficiency_score > 0.3
        )  # Should be decent with 50% hit rate

    def test_overall_performance_score(self, performance_monitor):
        """Test overall performance score calculation."""
        # Add good performance data
        for _i in range(20):
            performance_monitor.record_request("behavior", 150, 50, cache_hit=True)

        metrics = performance_monitor.get_metrics()

        # Should have high performance score
        assert 0 <= metrics.overall_performance_score <= 1
        assert metrics.overall_performance_score > 0.7  # Good performance


class TestGlobalMonitor:
    """Test global performance monitor functionality."""

    def test_get_global_monitor(self):
        """Test getting global performance monitor instance."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        # Should return same instance
        assert monitor1 is monitor2

    def test_reset_global_monitor(self):
        """Test resetting global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor1.record_request("behavior", 100, 50, False)

        # Reset global monitor
        reset_performance_monitor()

        monitor2 = get_performance_monitor()

        # Should be new instance
        assert monitor1 is not monitor2
        assert monitor2.total_requests == 0


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance monitoring."""

    def test_performance_monitor_thread_safety(self):
        """Test performance monitor is thread-safe."""
        import threading
        import time

        monitor = LLMPerformanceMonitor()
        results = []

        def record_requests():
            for i in range(100):
                monitor.record_request("behavior", 100 + i, 50, False)
                time.sleep(0.001)  # Small delay
            results.append("done")

        # Run concurrent threads
        threads = [threading.Thread(target=record_requests) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete
        assert len(results) == 3
        assert monitor.total_requests == 300

    def test_performance_monitor_memory_usage(self):
        """Test performance monitor doesn't leak memory."""
        monitor = LLMPerformanceMonitor(window_size=100)

        # Add many requests
        for _i in range(1000):
            monitor.record_request("behavior", 100, 50, False)

        # Should not exceed window size
        assert len(monitor.response_times) == 100
        assert len(monitor.token_usage) == 100
        assert len(monitor.queue_depths) == 100

    def test_performance_monitor_with_real_metrics(self):
        """Test performance monitor with realistic metrics."""
        monitor = LLMPerformanceMonitor()

        # Simulate realistic usage pattern
        analysis_types = ["behavior", "examples", "edge_cases", "type_consistency"]
        models = ["gpt-4o-mini", "gpt-4o"]

        for i in range(100):
            analysis_type = analysis_types[i % len(analysis_types)]
            model = models[i % len(models)]

            # Realistic response times and token usage
            if analysis_type == "behavior":
                response_time = 200 + (i % 100)
                tokens = 150 + (i % 50)
            else:
                response_time = 300 + (i % 150)
                tokens = 200 + (i % 100)

            cache_hit = i % 4 == 0  # 25% cache hit rate
            error = "timeout" if i % 50 == 0 else None  # 2% error rate

            monitor.record_request(
                analysis_type=analysis_type,
                response_time_ms=response_time,
                tokens_used=tokens,
                cache_hit=cache_hit,
                model=model,
                error=error,
                queue_depth=i % 5,
            )

        metrics = monitor.get_metrics()

        # Verify realistic metrics
        assert metrics.total_tokens_used > 0
        assert metrics.avg_response_time_ms > 200
        assert metrics.overall_cache_hit_rate == 0.25
        assert metrics.error_rate == 0.02
        assert len(metrics.cache_hit_rate_by_type) == 4
        assert metrics.estimated_cost_usd > 0
