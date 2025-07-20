import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ContextManager

import psutil

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""

    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float  # MB
    memory_after: float  # MB
    memory_peak: float  # MB
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_used(self) -> float:
        """Memory used during operation (MB)."""
        return self.memory_after - self.memory_before

    @property
    def memory_peak_delta(self) -> float:
        """Peak memory increase during operation (MB)."""
        return self.memory_peak - self.memory_before


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting."""

    max_operation_time: float = 30.0  # seconds
    max_memory_usage: float = 500.0  # MB
    max_memory_per_operation: float = 100.0  # MB
    min_success_rate: float = 0.85  # 85%
    max_error_rate: float = 0.15  # 15%


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for semantic matching operations.

    Tracks timing, memory usage, success rates, and provides alerting
    capabilities for production monitoring.
    """

    def __init__(
        self,
        thresholds: PerformanceThresholds | None = None,
        history_size: int = 1000,
        enable_real_time_alerts: bool = True,
    ):
        self.thresholds = thresholds or PerformanceThresholds()
        self.history_size = history_size
        self.enable_real_time_alerts = enable_real_time_alerts

        # Thread-safe operation tracking
        self._lock = threading.RLock()
        self._operation_history = deque(maxlen=history_size)
        self._active_operations = {}  # operation_id -> start_metrics

        # Aggregated statistics
        self._stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_duration": 0.0,
            "total_memory_used": 0.0,
            "peak_memory_usage": 0.0,
            "operations_by_type": defaultdict(int),
            "errors_by_type": defaultdict(int),
        }

        # System resource monitoring
        self.process = psutil.Process()
        self._initial_memory = self._get_memory_usage()

        # Alert callbacks
        self._alert_callbacks: list[Callable[[str, dict[str, Any]], None]] = []

    def add_alert_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Add a callback function for performance alerts."""
        self._alert_callbacks.append(callback)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _get_system_metrics(self) -> dict[str, Any]:
        """Get current system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return {
            "cpu_percent": cpu_percent,
            "memory_total_mb": memory.total / 1024 / 1024,
            "memory_available_mb": memory.available / 1024 / 1024,
            "memory_used_percent": memory.percent,
            "process_memory_mb": self._get_memory_usage(),
            "process_memory_percent": self.process.memory_percent(),
            "cpu_count": psutil.cpu_count(),
            "disk_usage": self._get_disk_usage(),
        }

    def _get_disk_usage(self) -> dict[str, float]:
        """Get disk usage for current working directory."""
        try:
            disk = psutil.disk_usage(Path.cwd())
            return {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_percent": (disk.used / disk.total) * 100,
            }
        except Exception as e:
            logger.warning(f"Could not get disk usage: {e}")
            return {"error": str(e)}

    @contextmanager
    def track_operation(
        self, operation_name: str, metadata: dict[str, Any] | None = None
    ) -> ContextManager["OperationTracker"]:
        """
        Context manager for tracking an operation.

        Usage:
            with monitor.track_operation("embedding_generation") as tracker:
                result = generate_embedding()
                tracker.set_metadata({"functions_processed": 10})
        """
        operation_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"
        tracker = OperationTracker(self, operation_id, operation_name, metadata or {})

        try:
            yield tracker
        except Exception as e:
            tracker._mark_error(str(e))
            raise
        finally:
            tracker._finalize()

    def record_operation(
        self,
        operation_name: str,
        duration: float,
        success: bool,
        memory_used: float = 0.0,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed operation manually."""
        current_memory = self._get_memory_usage()

        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time() - duration,
            end_time=time.time(),
            duration=duration,
            memory_before=current_memory - memory_used,
            memory_after=current_memory,
            memory_peak=current_memory,  # Approximation
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )

        self._record_metrics(metrics)

    def _record_metrics(self, metrics: OperationMetrics) -> None:
        """Record operation metrics and update statistics."""
        with self._lock:
            # Add to history
            self._operation_history.append(metrics)

            # Update aggregated stats
            self._stats["total_operations"] += 1
            if metrics.success:
                self._stats["successful_operations"] += 1
            else:
                self._stats["failed_operations"] += 1
                if metrics.error_message:
                    self._stats["errors_by_type"][metrics.error_message] += 1

            self._stats["total_duration"] += metrics.duration
            self._stats["total_memory_used"] += metrics.memory_used
            self._stats["peak_memory_usage"] = max(
                self._stats["peak_memory_usage"], metrics.memory_peak
            )
            self._stats["operations_by_type"][metrics.operation_name] += 1

            # Check thresholds and alert if necessary
            if self.enable_real_time_alerts:
                self._check_thresholds(metrics)

    def _check_thresholds(self, metrics: OperationMetrics) -> None:
        """Check if operation exceeded performance thresholds."""
        alerts = []

        # Duration threshold
        if metrics.duration > self.thresholds.max_operation_time:
            alerts.append(
                {
                    "type": "slow_operation",
                    "message": f"Operation '{metrics.operation_name}' took {metrics.duration:.2f}s (threshold: {self.thresholds.max_operation_time}s)",
                    "metrics": metrics,
                }
            )

        # Memory threshold
        if metrics.memory_used > self.thresholds.max_memory_per_operation:
            alerts.append(
                {
                    "type": "high_memory_usage",
                    "message": f"Operation '{metrics.operation_name}' used {metrics.memory_used:.1f}MB (threshold: {self.thresholds.max_memory_per_operation}MB)",
                    "metrics": metrics,
                }
            )

        # Overall memory threshold
        if metrics.memory_after > self.thresholds.max_memory_usage:
            alerts.append(
                {
                    "type": "memory_limit_exceeded",
                    "message": f"Total memory usage {metrics.memory_after:.1f}MB exceeds threshold {self.thresholds.max_memory_usage}MB",
                    "metrics": metrics,
                }
            )

        # Success rate threshold (check recent operations)
        recent_ops = list(self._operation_history)[-50:]  # Last 50 operations
        if len(recent_ops) >= 10:
            success_rate = sum(1 for op in recent_ops if op.success) / len(recent_ops)
            if success_rate < self.thresholds.min_success_rate:
                alerts.append(
                    {
                        "type": "low_success_rate",
                        "message": f"Recent success rate {success_rate:.1%} below threshold {self.thresholds.min_success_rate:.1%}",
                        "metrics": {
                            "success_rate": success_rate,
                            "sample_size": len(recent_ops),
                        },
                    }
                )

        # Send alerts
        for alert in alerts:
            self._send_alert(alert["type"], alert)

    def _send_alert(self, alert_type: str, alert_data: dict[str, Any]) -> None:
        """Send alert to registered callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            # Calculate derived metrics
            total_ops = self._stats["total_operations"]
            success_rate = (
                self._stats["successful_operations"] / total_ops
                if total_ops > 0
                else 0.0
            )
            avg_duration = (
                self._stats["total_duration"] / total_ops if total_ops > 0 else 0.0
            )
            avg_memory = (
                self._stats["total_memory_used"] / total_ops if total_ops > 0 else 0.0
            )

            return {
                "summary": {
                    "total_operations": total_ops,
                    "success_rate": success_rate,
                    "error_rate": 1.0 - success_rate,
                    "average_duration_seconds": avg_duration,
                    "average_memory_mb": avg_memory,
                    "peak_memory_mb": self._stats["peak_memory_usage"],
                    "total_runtime_seconds": self._stats["total_duration"],
                },
                "operations_by_type": dict(self._stats["operations_by_type"]),
                "errors_by_type": dict(self._stats["errors_by_type"]),
                "system_metrics": self._get_system_metrics(),
                "thresholds": {
                    "max_operation_time": self.thresholds.max_operation_time,
                    "max_memory_usage": self.thresholds.max_memory_usage,
                    "max_memory_per_operation": self.thresholds.max_memory_per_operation,
                    "min_success_rate": self.thresholds.min_success_rate,
                },
            }

    def get_operation_history(
        self, operation_name: str | None = None, limit: int | None = None
    ) -> list[OperationMetrics]:
        """Get operation history, optionally filtered by operation name."""
        with self._lock:
            history = list(self._operation_history)

            if operation_name:
                history = [op for op in history if op.operation_name == operation_name]

            if limit:
                history = history[-limit:]

            return history

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        current_metrics = self.get_current_metrics()

        # Recent performance trends
        recent_ops = self.get_operation_history(limit=100)

        if recent_ops:
            recent_durations = [op.duration for op in recent_ops]
            recent_memory = [op.memory_used for op in recent_ops]

            trends = {
                "recent_avg_duration": sum(recent_durations) / len(recent_durations),
                "recent_min_duration": min(recent_durations),
                "recent_max_duration": max(recent_durations),
                "recent_avg_memory": sum(recent_memory) / len(recent_memory),
                "recent_max_memory": max(recent_memory) if recent_memory else 0,
                "recent_operations": len(recent_ops),
            }
        else:
            trends = {}

        # Performance issues
        issues = self._identify_performance_issues()

        # Recommendations
        recommendations = self._generate_recommendations(
            current_metrics, trends, issues
        )

        return {
            "timestamp": time.time(),
            "current_metrics": current_metrics,
            "recent_trends": trends,
            "performance_issues": issues,
            "recommendations": recommendations,
            "monitoring_config": {
                "history_size": self.history_size,
                "real_time_alerts": self.enable_real_time_alerts,
                "thresholds": current_metrics["thresholds"],
            },
        }

    def _identify_performance_issues(self) -> list[dict[str, Any]]:
        """Identify current performance issues."""
        issues = []
        metrics = self.get_current_metrics()

        # High error rate
        if metrics["summary"]["error_rate"] > self.thresholds.max_error_rate:
            issues.append(
                {
                    "type": "high_error_rate",
                    "severity": "high",
                    "description": f"Error rate {metrics['summary']['error_rate']:.1%} exceeds threshold {self.thresholds.max_error_rate:.1%}",
                    "current_value": metrics["summary"]["error_rate"],
                    "threshold": self.thresholds.max_error_rate,
                }
            )

        # High memory usage
        system_memory = metrics["system_metrics"]["process_memory_mb"]
        if system_memory > self.thresholds.max_memory_usage:
            issues.append(
                {
                    "type": "high_memory_usage",
                    "severity": "medium",
                    "description": f"Memory usage {system_memory:.1f}MB exceeds threshold {self.thresholds.max_memory_usage}MB",
                    "current_value": system_memory,
                    "threshold": self.thresholds.max_memory_usage,
                }
            )

        # Slow operations
        if (
            metrics["summary"]["average_duration_seconds"]
            > self.thresholds.max_operation_time / 2
        ):
            issues.append(
                {
                    "type": "slow_operations",
                    "severity": "medium",
                    "description": f"Average operation time {metrics['summary']['average_duration_seconds']:.2f}s is high",
                    "current_value": metrics["summary"]["average_duration_seconds"],
                    "threshold": self.thresholds.max_operation_time,
                }
            )

        return issues

    def _generate_recommendations(
        self,
        current_metrics: dict[str, Any],
        trends: dict[str, Any],
        issues: list[dict[str, Any]],
    ) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Memory recommendations
        memory_usage = current_metrics["system_metrics"]["process_memory_mb"]
        if memory_usage > 300:
            recommendations.append(
                "Consider reducing batch sizes or enabling more aggressive garbage collection "
                f"(current memory usage: {memory_usage:.1f}MB)"
            )

        # Duration recommendations
        avg_duration = current_metrics["summary"]["average_duration_seconds"]
        if avg_duration > 10:
            recommendations.append(
                "Operations are running slowly. Consider enabling caching, reducing batch sizes, "
                f"or optimizing API calls (average: {avg_duration:.2f}s)"
            )

        # Error rate recommendations
        error_rate = current_metrics["summary"]["error_rate"]
        if error_rate > 0.1:
            recommendations.append(
                f"High error rate ({error_rate:.1%}). Check network connectivity, API keys, "
                "and consider implementing more robust retry logic"
            )

        # System resource recommendations
        cpu_percent = current_metrics["system_metrics"]["cpu_percent"]
        if cpu_percent > 80:
            recommendations.append(
                f"High CPU usage ({cpu_percent:.1f}%). Consider reducing concurrent operations "
                "or optimizing processing algorithms"
            )

        # Disk space recommendations
        disk_usage = current_metrics["system_metrics"]["disk_usage"]
        if isinstance(disk_usage, dict) and disk_usage.get("used_percent", 0) > 85:
            recommendations.append(
                f"Disk usage is high ({disk_usage['used_percent']:.1f}%). "
                "Consider cleaning up cache files or increasing storage"
            )

        return recommendations

    def export_metrics(self, file_path: str) -> None:
        """Export current metrics to a JSON file."""
        report = self.get_performance_report()

        with open(file_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance metrics exported to {file_path}")

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._operation_history.clear()
            self._stats = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "total_duration": 0.0,
                "total_memory_used": 0.0,
                "peak_memory_usage": 0.0,
                "operations_by_type": defaultdict(int),
                "errors_by_type": defaultdict(int),
            }

        logger.info("Performance metrics reset")


class OperationTracker:
    """Tracks a single operation's performance metrics."""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_id: str,
        operation_name: str,
        metadata: dict[str, Any],
    ):
        self.monitor = monitor
        self.operation_id = operation_id
        self.operation_name = operation_name
        self.metadata = metadata.copy()

        # Start tracking
        self.start_time = time.time()
        self.memory_before = monitor._get_memory_usage()
        self.memory_peak = self.memory_before
        self.success = True
        self.error_message = None

        # Monitor peak memory during operation
        self._monitoring = True
        self._peak_monitor_thread = threading.Thread(
            target=self._monitor_peak_memory, daemon=True
        )
        self._peak_monitor_thread.start()

    def _monitor_peak_memory(self) -> None:
        """Monitor peak memory usage during operation."""
        while self._monitoring:
            try:
                current_memory = self.monitor._get_memory_usage()
                self.memory_peak = max(self.memory_peak, current_memory)
                time.sleep(0.1)  # Check every 100ms
            except Exception:
                break  # Exit on any error

    def set_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the operation."""
        self.metadata[key] = value

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update operation metadata."""
        self.metadata.update(metadata)

    def _mark_error(self, error_message: str) -> None:
        """Mark operation as failed."""
        self.success = False
        self.error_message = error_message

    def _finalize(self) -> None:
        """Finalize operation tracking."""
        self._monitoring = False

        end_time = time.time()
        memory_after = self.monitor._get_memory_usage()

        metrics = OperationMetrics(
            operation_name=self.operation_name,
            start_time=self.start_time,
            end_time=end_time,
            duration=end_time - self.start_time,
            memory_before=self.memory_before,
            memory_after=memory_after,
            memory_peak=self.memory_peak,
            success=self.success,
            error_message=self.error_message,
            metadata=self.metadata,
        )

        self.monitor._record_metrics(metrics)


# Convenience functions for common monitoring patterns


def create_console_alert_handler() -> Callable[[str, dict[str, Any]], None]:
    """Create a console alert handler for debugging."""

    def handle_alert(alert_type: str, alert_data: dict[str, Any]) -> None:
        logger.warning(
            f"PERFORMANCE ALERT [{alert_type}]: {alert_data.get('message', 'Unknown alert')}"
        )

    return handle_alert


def create_file_alert_handler(file_path: str) -> Callable[[str, dict[str, Any]], None]:
    """Create a file-based alert handler."""

    def handle_alert(alert_type: str, alert_data: dict[str, Any]) -> None:
        alert_record = {
            "timestamp": time.time(),
            "alert_type": alert_type,
            "data": alert_data,
        }

        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(alert_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert to {file_path}: {e}")

    return handle_alert
