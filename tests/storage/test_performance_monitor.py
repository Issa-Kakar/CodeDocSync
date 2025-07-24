"""Tests for performance monitoring functionality."""

import json
import logging
import time
from unittest.mock import MagicMock

import pytest

from codedocsync.storage.performance_monitor import (
    PerformanceMonitor,
    PerformanceThresholds,
    create_console_alert_handler,
    create_file_alert_handler,
)


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""

    def test_init_default_thresholds(self):
        """Test initialization with default thresholds."""
        monitor = PerformanceMonitor()

        assert monitor.thresholds.max_operation_time == 30.0
        assert monitor.thresholds.max_memory_usage == 500.0
        assert monitor.thresholds.max_memory_per_operation == 100.0
        assert monitor.thresholds.min_success_rate == 0.85
        assert monitor.enable_real_time_alerts is True

    def test_init_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        thresholds = PerformanceThresholds(
            max_operation_time=60.0,
            max_memory_usage=1000.0,
            max_memory_per_operation=200.0,
            min_success_rate=0.95,
        )
        monitor = PerformanceMonitor(thresholds=thresholds, history_size=500)

        assert monitor.thresholds.max_operation_time == 60.0
        assert monitor.thresholds.max_memory_usage == 1000.0
        assert monitor.history_size == 500

    def test_track_operation_success(self, mock_psutil):
        """Test tracking a successful operation."""
        monitor = PerformanceMonitor()

        with monitor.track_operation("test_op", {"key": "value"}) as tracker:
            time.sleep(0.01)  # Simulate work
            tracker.set_metadata("result", "success")

        # Check operation was recorded
        history = monitor.get_operation_history()
        assert len(history) == 1

        op = history[0]
        assert op.operation_name == "test_op"
        assert op.success is True
        assert op.duration > 0
        assert op.metadata["key"] == "value"
        assert op.metadata["result"] == "success"

    def test_track_operation_failure(self, mock_psutil):
        """Test tracking a failed operation."""
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError):
            with monitor.track_operation("failing_op"):
                raise ValueError("Test error")

        # Check operation was recorded as failed
        history = monitor.get_operation_history()
        assert len(history) == 1

        op = history[0]
        assert op.operation_name == "failing_op"
        assert op.success is False
        assert op.error_message == "Test error"

    def test_track_operation_metadata(self, mock_psutil):
        """Test setting operation metadata."""
        monitor = PerformanceMonitor()

        with monitor.track_operation("metadata_test") as tracker:
            tracker.set_metadata("items_processed", 100)
            tracker.update_metadata({"status": "complete", "errors": 0})

        op = monitor.get_operation_history()[0]
        assert op.metadata["items_processed"] == 100
        assert op.metadata["status"] == "complete"
        assert op.metadata["errors"] == 0

    def test_record_operation_manual(self, mock_psutil):
        """Test manually recording an operation."""
        monitor = PerformanceMonitor()

        monitor.record_operation(
            operation_name="manual_op",
            duration=5.0,
            success=True,
            memory_used=10.0,
            metadata={"batch_size": 50},
        )

        op = monitor.get_operation_history()[0]
        assert op.operation_name == "manual_op"
        assert op.duration == 5.0
        assert op.success is True
        assert op.metadata["batch_size"] == 50

    def test_memory_usage_tracking(self, mock_psutil):
        """Test memory usage tracking with mocked values."""
        monitor = PerformanceMonitor()

        # Mock changing memory values
        memory_values = [100 * 1024 * 1024, 120 * 1024 * 1024, 110 * 1024 * 1024]
        mock_psutil.Process.return_value.memory_info.side_effect = [
            MagicMock(rss=val) for val in memory_values
        ]

        with monitor.track_operation("memory_test"):
            time.sleep(0.05)  # Let peak monitor thread run

        op = monitor.get_operation_history()[0]
        assert op.memory_before == 100.0  # MB
        assert op.memory_after == 110.0  # MB
        # Peak tracking happens in background thread, so we can't guarantee exact value

    def test_memory_peak_tracking(self, mock_psutil):
        """Test peak memory tracking."""
        monitor = PerformanceMonitor()

        # Simulate memory spike during operation
        memory_values = [
            100 * 1024 * 1024,  # Before
            150 * 1024 * 1024,  # Peak during operation
            120 * 1024 * 1024,  # After
        ]
        mock_psutil.Process.return_value.memory_info.side_effect = [
            MagicMock(rss=val) for val in memory_values
        ]

        with monitor.track_operation("peak_test"):
            time.sleep(0.15)  # Let peak monitor detect spike

        # Peak should be tracked (but exact value depends on thread timing)
        metrics = monitor.get_current_metrics()
        assert metrics["summary"]["peak_memory_mb"] > 0

    def test_get_current_metrics(self, mock_psutil):
        """Test getting current performance metrics."""
        monitor = PerformanceMonitor()

        # Record some operations
        monitor.record_operation("op1", 1.0, True)
        monitor.record_operation("op2", 2.0, True)
        monitor.record_operation("op3", 1.5, False, error_message="Error")

        metrics = monitor.get_current_metrics()

        assert metrics["summary"]["total_operations"] == 3
        assert metrics["summary"]["success_rate"] == 2 / 3
        assert metrics["summary"]["average_duration_seconds"] == 1.5
        assert metrics["operations_by_type"]["op1"] == 1
        assert metrics["operations_by_type"]["op2"] == 1
        assert metrics["operations_by_type"]["op3"] == 1
        assert metrics["errors_by_type"]["Error"] == 1

    def test_get_operation_history(self, mock_psutil):
        """Test retrieving operation history."""
        monitor = PerformanceMonitor()

        # Record different operations
        monitor.record_operation("type_a", 1.0, True)
        monitor.record_operation("type_b", 2.0, True)
        monitor.record_operation("type_a", 1.5, True)

        # Get all history
        all_history = monitor.get_operation_history()
        assert len(all_history) == 3

        # Get filtered history
        type_a_history = monitor.get_operation_history("type_a")
        assert len(type_a_history) == 2
        assert all(op.operation_name == "type_a" for op in type_a_history)

        # Get limited history
        limited_history = monitor.get_operation_history(limit=2)
        assert len(limited_history) == 2

    def test_get_performance_report(self, mock_psutil):
        """Test generating performance report."""
        monitor = PerformanceMonitor()

        # Record some operations
        for i in range(5):
            monitor.record_operation(f"op_{i}", float(i), i % 2 == 0)

        report = monitor.get_performance_report()

        assert "timestamp" in report
        assert "current_metrics" in report
        assert "recent_trends" in report
        assert "performance_issues" in report
        assert "recommendations" in report
        assert "monitoring_config" in report

    def test_alert_slow_operation(self, mock_psutil):
        """Test alert for slow operations."""
        monitor = PerformanceMonitor(
            thresholds=PerformanceThresholds(max_operation_time=1.0)
        )

        alerts = []
        monitor.add_alert_callback(lambda t, d: alerts.append((t, d)))

        # Record slow operation
        monitor.record_operation("slow_op", 2.0, True)

        assert len(alerts) == 1
        assert alerts[0][0] == "slow_operation"
        assert "took 2.00s" in alerts[0][1]["message"]

    def test_alert_high_memory_usage(self, mock_psutil):
        """Test alert for high memory usage."""
        monitor = PerformanceMonitor(
            thresholds=PerformanceThresholds(max_memory_per_operation=50.0)
        )

        alerts = []
        monitor.add_alert_callback(lambda t, d: alerts.append((t, d)))

        # Record high memory operation
        monitor.record_operation("memory_hog", 1.0, True, memory_used=100.0)

        assert len(alerts) == 1
        assert alerts[0][0] == "high_memory_usage"
        assert "used 100.0MB" in alerts[0][1]["message"]

    def test_alert_low_success_rate(self, mock_psutil):
        """Test alert for low success rate."""
        monitor = PerformanceMonitor(
            thresholds=PerformanceThresholds(min_success_rate=0.8)
        )

        alerts = []
        monitor.add_alert_callback(lambda t, d: alerts.append((t, d)))

        # Record many failures
        for i in range(20):
            monitor.record_operation(f"op_{i}", 0.1, i < 5)  # 25% success rate

        # Should trigger low success rate alert
        assert any(alert[0] == "low_success_rate" for alert in alerts)

    def test_alert_callbacks(self, mock_psutil, caplog):
        """Test alert callback mechanism."""
        monitor = PerformanceMonitor()

        callback_called = []

        def test_callback(alert_type, data):
            callback_called.append((alert_type, data))

        # Test callback that raises exception
        def failing_callback(alert_type, data):
            raise RuntimeError("Callback error")

        monitor.add_alert_callback(test_callback)
        monitor.add_alert_callback(failing_callback)

        # Trigger alert
        monitor._send_alert("test_alert", {"message": "test"})

        # First callback should be called
        assert len(callback_called) == 1
        assert callback_called[0][0] == "test_alert"

        # Error from second callback should be logged
        assert "Alert callback failed" in caplog.text

    def test_export_metrics(self, mock_psutil, tmp_path):
        """Test exporting metrics to file."""
        monitor = PerformanceMonitor()

        # Record some operations
        monitor.record_operation("export_test", 1.0, True)

        # Export to file
        export_path = tmp_path / "metrics.json"
        monitor.export_metrics(str(export_path))

        # Verify file contents
        assert export_path.exists()
        with open(export_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "current_metrics" in data
        assert data["current_metrics"]["summary"]["total_operations"] == 1

    def test_reset_metrics(self, mock_psutil):
        """Test resetting all metrics."""
        monitor = PerformanceMonitor()

        # Record operations
        monitor.record_operation("reset_test", 1.0, True)
        monitor.record_operation("reset_test", 2.0, False)

        # Verify operations recorded
        assert monitor.get_current_metrics()["summary"]["total_operations"] == 2

        # Reset
        monitor.reset_metrics()

        # Verify reset
        metrics = monitor.get_current_metrics()
        assert metrics["summary"]["total_operations"] == 0
        assert len(monitor.get_operation_history()) == 0

    def test_operation_tracker_context(self, mock_psutil):
        """Test OperationTracker context manager."""
        monitor = PerformanceMonitor()

        # Test normal operation
        with monitor.track_operation("context_test") as tracker:
            assert tracker.operation_name == "context_test"
            assert tracker.success is True
            tracker.set_metadata("test", "value")

        # Verify operation recorded
        op = monitor.get_operation_history()[0]
        assert op.success is True
        assert op.metadata["test"] == "value"

    def test_history_size_limit(self, mock_psutil):
        """Test that history respects size limit."""
        monitor = PerformanceMonitor(history_size=3)

        # Record more operations than history size
        for i in range(5):
            monitor.record_operation(f"op_{i}", 1.0, True)

        # Should only keep last 3
        history = monitor.get_operation_history()
        assert len(history) == 3
        assert history[0].operation_name == "op_2"
        assert history[-1].operation_name == "op_4"

    def test_performance_issues_identification(self, mock_psutil):
        """Test identifying performance issues."""
        # Mock high memory usage
        mock_psutil.Process.return_value.memory_info.return_value = MagicMock(
            rss=600 * 1024 * 1024  # 600 MB
        )

        monitor = PerformanceMonitor(
            thresholds=PerformanceThresholds(max_memory_usage=500.0, max_error_rate=0.1)
        )

        # Create high error rate
        for i in range(10):
            monitor.record_operation("op", 1.0, i < 3)  # 30% success = 70% error

        issues = monitor._identify_performance_issues()

        # Should identify both issues
        issue_types = [issue["type"] for issue in issues]
        assert "high_error_rate" in issue_types
        assert "high_memory_usage" in issue_types


def test_create_console_alert_handler(caplog):
    """Test console alert handler creation."""
    handler = create_console_alert_handler()

    with caplog.at_level(logging.WARNING):
        handler("test_alert", {"message": "Test message"})

    assert "PERFORMANCE ALERT [test_alert]: Test message" in caplog.text


def test_create_file_alert_handler(tmp_path):
    """Test file alert handler creation."""
    alert_file = tmp_path / "alerts.log"
    handler = create_file_alert_handler(str(alert_file))

    # Send alert
    handler("file_alert", {"message": "File test", "severity": "high"})

    # Verify file contents
    assert alert_file.exists()
    with open(alert_file) as f:
        line = f.readline()
        data = json.loads(line)

    assert data["alert_type"] == "file_alert"
    assert data["data"]["message"] == "File test"
    assert "timestamp" in data
