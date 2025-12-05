"""
Comprehensive tests for MetricsCollector.
Tests sliding window aggregation, trigger detection, and performance.
"""

import asyncio
import json
import os
import pickle
import statistics
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from monitoring.metrics import (
    Anomaly,
    CorrelationResult,
    CustomMetric,
    MetricsCollector,
    MetricsData,
    MetricWindow,
)


class TestMetricsData:
    """Test MetricsData dataclass."""

    def test_metrics_data_creation(self):
        """Test creating MetricsData instance."""
        data = MetricsData(
            task_id="test-123",
            timestamp=datetime.now(),
            response_time=1.5,
            success=True,
            token_usage=100,
            error_type=None,
            metadata={"extra": "data"},
        )

        assert data.task_id == "test-123"
        assert data.response_time == 1.5
        assert data.success is True
        assert data.token_usage == 100
        assert data.metadata["extra"] == "data"


class TestMetricWindow:
    """Test MetricWindow dataclass and properties."""

    def test_metric_window_creation(self):
        """Test creating MetricWindow instance."""
        start = datetime.now()
        end = start + timedelta(hours=1)

        window = MetricWindow(
            start_time=start,
            end_time=end,
            total_requests=100,
            success_count=85,
            avg_response_time=1.2,
            total_tokens=5000,
            error_counts={"timeout": 10, "server_error": 5},
        )

        assert window.total_requests == 100
        assert window.success_count == 85
        assert window.success_rate == 0.85
        assert window.avg_tokens_per_request == 50.0
        assert window.error_counts["timeout"] == 10

    def test_success_rate_with_zero_requests(self):
        """Test success rate calculation with zero requests."""
        window = MetricWindow(
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_requests=0,
            success_count=0,
            avg_response_time=0.0,
            total_tokens=0,
        )

        assert window.success_rate == 0.0
        assert window.avg_tokens_per_request == 0.0


class TestMetricsCollector:
    """Test MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create MetricsCollector instance for testing."""
        return MetricsCollector(
            window_1hr=timedelta(hours=1),
            window_24hr=timedelta(hours=24),
            max_data_points=1000,
            enable_langfuse=False,
        )

    def test_collector_initialization(self, collector):
        """Test MetricsCollector initialization."""
        assert collector.window_1hr == timedelta(hours=1)
        assert collector.window_24hr == timedelta(hours=24)
        assert collector.max_data_points == 1000
        assert collector.enable_langfuse is False
        assert len(collector.data_points) == 0
        assert len(collector.thresholds) > 0

    def test_collect_basic_metrics(self, collector):
        """Test basic metrics collection."""
        metrics = {
            "response_time": 1.5,
            "success": True,
            "token_usage": 100,
            "error_type": None,
        }

        collector.collect("task-123", metrics)

        assert len(collector.data_points) == 1
        data_point = collector.data_points[0]
        assert data_point.task_id == "task-123"
        assert data_point.response_time == 1.5
        assert data_point.success is True
        assert data_point.token_usage == 100

    def test_collect_with_error(self, collector):
        """Test collecting metrics with error information."""
        metrics = {
            "response_time": 0.0,
            "success": False,
            "token_usage": 0,
            "error_type": "timeout",
            "extra_info": "connection failed",
        }

        collector.collect("error-task", metrics)

        data_point = collector.data_points[0]
        assert data_point.success is False
        assert data_point.error_type == "timeout"
        assert data_point.metadata["extra_info"] == "connection failed"

    def test_collect_invalid_metrics(self, collector):
        """Test collecting invalid metrics doesn't crash."""
        # Missing required fields
        collector.collect("task-1", {})

        # Invalid types - this should fail gracefully and not add invalid data
        collector.collect(
            "task-2",
            {"response_time": "invalid", "success": "maybe", "token_usage": "lots"},
        )

        # Should have 1 data point (only the first one with defaults applied)
        assert len(collector.data_points) == 1

    def test_get_window_metrics_empty(self, collector):
        """Test getting window metrics with no data."""
        window = collector.get_window_metrics(timedelta(hours=1))

        assert window.total_requests == 0
        assert window.success_count == 0
        assert window.success_rate == 0.0
        assert window.avg_response_time == 0.0
        assert window.total_tokens == 0

    def test_get_window_metrics_with_data(self, collector):
        """Test getting window metrics with data."""
        # Add test data
        now = datetime.now()

        # Add data points in the last hour
        test_data = [
            {"response_time": 1.0, "success": True, "token_usage": 100},
            {"response_time": 2.0, "success": True, "token_usage": 150},
            {
                "response_time": 0.5,
                "success": False,
                "token_usage": 50,
                "error_type": "timeout",
            },
        ]

        for i, metrics in enumerate(test_data):
            collector.collect(f"task-{i}", metrics)

        window = collector.get_window_metrics(timedelta(hours=1))

        assert window.total_requests == 3
        assert window.success_count == 2
        assert window.success_rate == 2 / 3
        assert (
            abs(window.avg_response_time - 1.167) < 0.01
        )  # approximately (1.0+2.0+0.5)/3
        assert window.total_tokens == 300
        assert window.error_counts.get("timeout", 0) == 1

    def test_window_metrics_caching(self, collector):
        """Test that window metrics are cached for performance."""
        # Add some data
        collector.collect(
            "task-1", {"response_time": 1.0, "success": True, "token_usage": 100}
        )

        # First call should compute and cache
        window1 = collector.get_window_metrics(timedelta(hours=1))

        # Second call should return cached result
        window2 = collector.get_window_metrics(timedelta(hours=1))

        assert window1 is window2  # Same object (cached)
        assert len(collector._cached_windows) > 0

    def test_check_evolution_triggers_insufficient_data(self, collector):
        """Test trigger checking with insufficient data."""
        # Add only a few data points
        for i in range(5):  # Less than min_requests_1hr threshold
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        trigger = collector.check_evolution_triggers()
        assert trigger is None

    def test_check_evolution_triggers_low_success_rate(self, collector):
        """Test trigger detection for low success rate."""
        # Add data with low success rate
        for i in range(15):  # Above min_requests_1hr threshold
            success = i < 5  # Only first 5 are successful (33% success rate)
            collector.collect(
                f"task-{i}",
                {
                    "response_time": 1.0,
                    "success": success,
                    "token_usage": 100,
                    "error_type": "server_error" if not success else None,
                },
            )

        trigger = collector.check_evolution_triggers()

        assert trigger is not None
        assert trigger.trigger_type == "threshold"
        assert "Low success rate" in str(trigger.trigger_reasons or [])

    def test_check_evolution_triggers_high_response_time(self, collector):
        """Test trigger detection for high response time."""
        # Add data with high response times
        for i in range(15):
            collector.collect(
                f"task-{i}",
                {
                    "response_time": 3.0,  # Above max_response_time_1hr threshold
                    "success": True,
                    "token_usage": 100,
                },
            )

        trigger = collector.check_evolution_triggers()

        assert trigger is not None
        assert "High response time" in str(trigger.trigger_reasons or [])

    def test_update_thresholds(self, collector):
        """Test updating trigger thresholds."""
        original_threshold = collector.thresholds["min_success_rate_1hr"]

        new_thresholds = {"min_success_rate_1hr": 0.95}
        collector.update_thresholds(new_thresholds)

        assert collector.thresholds["min_success_rate_1hr"] == 0.95
        assert collector.thresholds["min_success_rate_1hr"] != original_threshold

    def test_export_to_langfuse_disabled(self, collector):
        """Test Langfuse export when disabled."""
        result = collector.export_to_langfuse()
        assert result == {}

    def test_export_to_langfuse_enabled(self):
        """Test Langfuse export when enabled."""
        collector = MetricsCollector(enable_langfuse=True)

        # Add some test data
        collector.collect(
            "task-1", {"response_time": 1.0, "success": True, "token_usage": 100}
        )

        result = collector.export_to_langfuse()

        assert "timestamp" in result
        assert "metrics" in result
        assert "thresholds" in result
        assert result["metrics"]["total_requests"] == 1
        assert result["metrics"]["success_rate"] == 1.0

    def test_get_stats(self, collector):
        """Test getting collector statistics."""
        # Add some test data
        collector.collect(
            "task-1", {"response_time": 1.0, "success": True, "token_usage": 100}
        )

        stats = collector.get_stats()

        assert "data_points_stored" in stats
        assert "cache_entries" in stats
        assert "thresholds" in stats
        assert "current_metrics" in stats
        assert stats["data_points_stored"] == 1
        assert "1hr_window" in stats["current_metrics"]
        assert "24hr_window" in stats["current_metrics"]

    def test_clear_data(self, collector):
        """Test clearing all data."""
        # Add some test data
        collector.collect(
            "task-1", {"response_time": 1.0, "success": True, "token_usage": 100}
        )
        collector.get_window_metrics(timedelta(hours=1))  # Create cache entry

        assert len(collector.data_points) > 0
        assert len(collector._cached_windows) > 0

        collector.clear_data()

        assert len(collector.data_points) == 0
        assert len(collector._cached_windows) == 0

    def test_max_data_points_limit(self):
        """Test that data points are limited to max_data_points."""
        collector = MetricsCollector(max_data_points=5)  # Small limit for testing

        # Add more data points than the limit
        for i in range(10):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        # Should only keep the last 5 data points
        assert len(collector.data_points) == 5
        assert collector.data_points[0].task_id == "task-5"  # First kept item
        assert collector.data_points[-1].task_id == "task-9"  # Last item


class TestMetricsCollectorPerformance:
    """Test MetricsCollector performance characteristics."""

    def test_collect_performance(self):
        """Test that collect() operation is fast."""
        collector = MetricsCollector()

        # Time multiple collect operations
        start_time = time.time()

        for i in range(1000):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete 1000 operations in under 0.1 seconds
        assert total_time < 0.1, f"Collect operations took {total_time:.3f}s, too slow"

        # Calculate overhead per operation
        overhead_per_op = total_time / 1000
        assert (
            overhead_per_op < 0.0001
        ), f"Per-operation overhead {overhead_per_op:.6f}s too high"

    def test_window_metrics_performance(self):
        """Test that window metrics calculation is fast."""
        collector = MetricsCollector()

        # Add substantial amount of data
        for i in range(1000):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        # Time window metrics calculation
        start_time = time.time()

        for _ in range(100):  # Multiple calculations
            window = collector.get_window_metrics(timedelta(hours=1))

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete 100 window calculations in under 0.05 seconds (with caching)
        assert (
            total_time < 0.05
        ), f"Window calculations took {total_time:.3f}s, too slow"

    def test_memory_usage_bounded(self):
        """Test that memory usage stays bounded."""
        collector = MetricsCollector(max_data_points=100)

        # Add many data points
        for i in range(1000):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        # Memory should be bounded by max_data_points
        assert len(collector.data_points) <= 100

        # Cache should not grow unbounded either
        for hours in range(50):
            collector.get_window_metrics(timedelta(hours=hours + 1))

        # Trigger cleanup
        collector._cleanup_old_data()

        # Cache should be reasonable size
        assert len(collector._cached_windows) < 100


@pytest.mark.asyncio
async def test_async_compatibility():
    """Test that the collector works well in async context."""
    collector = MetricsCollector()

    # Simulate async collection
    async def collect_metrics():
        collector.collect(
            "async-task", {"response_time": 1.0, "success": True, "token_usage": 100}
        )

    # Run multiple async collections
    await asyncio.gather(*[collect_metrics() for _ in range(10)])

    assert len(collector.data_points) == 10


class TestProductionReadyFeatures:
    """Test production-ready features added to MetricsCollector."""

    @pytest.fixture
    def collector(self):
        """Create MetricsCollector instance for testing."""
        return MetricsCollector(
            window_1hr=timedelta(hours=1),
            window_24hr=timedelta(hours=24),
            max_data_points=1000,
            enable_langfuse=False,
            data_retention_days=7,
        )

    @pytest.fixture
    def collector_with_data(self, collector):
        """Collector with test data."""
        # Add diverse test data
        test_cases = [
            {
                "response_time": 1.0,
                "success": True,
                "token_usage": 100,
                "agent_id": "agent1",
            },
            {
                "response_time": 2.5,
                "success": True,
                "token_usage": 150,
                "agent_id": "agent1",
            },
            {
                "response_time": 0.8,
                "success": False,
                "token_usage": 50,
                "error_type": "timeout",
                "agent_id": "agent2",
            },
            {
                "response_time": 3.2,
                "success": True,
                "token_usage": 200,
                "agent_id": "agent2",
            },
            {
                "response_time": 1.8,
                "success": True,
                "token_usage": 120,
                "agent_id": "agent1",
            },
            {
                "response_time": 0.5,
                "success": False,
                "token_usage": 30,
                "error_type": "server_error",
                "agent_id": "agent2",
            },
        ]

        for i, metrics in enumerate(test_cases):
            collector.collect(f"task-{i}", metrics, metrics.get("agent_id"))

        return collector

    # Data Persistence Tests

    @pytest.mark.asyncio
    async def test_persist_and_restore_data(self, collector_with_data):
        """Test data persistence and restoration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = os.path.join(temp_dir, "metrics_backup")

            # Save current state
            original_count = len(collector_with_data.data_points)
            original_agent_count = len(collector_with_data.agent_metrics)

            await collector_with_data.persist_to_disk(persist_path)

            # Verify files were created
            assert os.path.exists(f"{persist_path}.pkl")
            assert os.path.exists(f"{persist_path}.json")

            # Clear data and restore
            collector_with_data.clear_data()
            assert len(collector_with_data.data_points) == 0

            await collector_with_data.restore_from_disk(persist_path)

            # Verify restoration
            assert len(collector_with_data.data_points) == original_count
            assert len(collector_with_data.agent_metrics) == original_agent_count

    @pytest.mark.asyncio
    async def test_restore_from_json_fallback(self, collector_with_data):
        """Test restoration from JSON when pickle is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = os.path.join(temp_dir, "metrics_backup")

            await collector_with_data.persist_to_disk(persist_path)

            # Remove pickle file to force JSON fallback
            os.remove(f"{persist_path}.pkl")

            collector_with_data.clear_data()
            await collector_with_data.restore_from_disk(persist_path)

            # Should still work with JSON
            assert len(collector_with_data.data_points) > 0

    @pytest.mark.asyncio
    async def test_restore_nonexistent_file(self, collector):
        """Test handling of nonexistent backup file."""
        with pytest.raises(FileNotFoundError):
            await collector.restore_from_disk("/nonexistent/path")

    # Multi-Agent Support Tests

    def test_collect_with_agent_id(self, collector):
        """Test collecting metrics with agent ID."""
        metrics = {"response_time": 1.5, "success": True, "token_usage": 100}

        collector.collect("task-1", metrics, "agent-alpha")

        # Check global storage
        assert len(collector.data_points) == 1
        assert collector.data_points[0].agent_id == "agent-alpha"

        # Check agent-specific storage
        assert "agent-alpha" in collector.agent_metrics
        assert len(collector.agent_metrics["agent-alpha"]) == 1

    def test_get_agent_metrics(self, collector_with_data):
        """Test getting metrics for specific agent."""
        agent1_metrics = collector_with_data.get_agent_metrics("agent1")
        agent2_metrics = collector_with_data.get_agent_metrics("agent2")

        # Verify agent separation
        assert agent1_metrics.total_requests > 0
        assert agent2_metrics.total_requests > 0
        # Both agents have 3 requests each, so check other metrics differ
        assert agent1_metrics.success_count != agent2_metrics.success_count

    def test_get_agent_metrics_nonexistent(self, collector):
        """Test getting metrics for nonexistent agent."""
        metrics = collector.get_agent_metrics("nonexistent-agent")

        assert metrics.total_requests == 0
        assert metrics.success_rate == 0.0

    # Custom Metrics Tests

    def test_register_custom_metric(self, collector):
        """Test registering custom metrics."""

        def max_aggregator(values):
            return max(values) if values else 0.0

        collector.register_custom_metric(
            "max_response_time", max_aggregator, "Maximum response time", "seconds"
        )

        assert "max_response_time" in collector.custom_metrics
        assert (
            collector.custom_metrics["max_response_time"].description
            == "Maximum response time"
        )

    def test_get_custom_metric_value(self, collector_with_data):
        """Test calculating custom metric values."""
        # Test built-in p95 metric
        p95_value = collector_with_data.get_custom_metric_value("p95_response_time")
        assert p95_value is not None
        assert p95_value > 0

        # Test nonexistent metric
        invalid_value = collector_with_data.get_custom_metric_value(
            "nonexistent_metric"
        )
        assert invalid_value is None

    def test_default_custom_metrics(self, collector):
        """Test that default custom metrics are registered."""
        expected_metrics = [
            "p95_response_time",
            "p99_response_time",
            "median_response_time",
        ]

        for metric_name in expected_metrics:
            assert metric_name in collector.custom_metrics

    # Anomaly Detection Tests

    def test_detect_anomalies_insufficient_data(self, collector):
        """Test anomaly detection with insufficient data."""
        # Add only a few data points
        for i in range(5):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        anomalies = collector.detect_anomalies()
        assert len(anomalies) == 0

    def test_detect_anomalies_response_time(self, collector):
        """Test anomaly detection for response time outliers."""
        # Add normal data
        for i in range(20):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        # Add clear outlier
        collector.collect(
            "outlier-task", {"response_time": 10.0, "success": True, "token_usage": 100}
        )

        anomalies = collector.detect_anomalies(sensitivity=2.0)

        # Should detect the outlier
        response_time_anomalies = [
            a for a in anomalies if a.metric_name == "response_time"
        ]
        assert len(response_time_anomalies) > 0
        assert response_time_anomalies[0].value == 10.0
        assert response_time_anomalies[0].severity in ["medium", "high"]

    def test_detect_anomalies_token_usage(self, collector):
        """Test anomaly detection for token usage outliers."""
        # Add normal data
        for i in range(20):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        # Add token usage outlier
        collector.collect(
            "token-outlier",
            {"response_time": 1.0, "success": True, "token_usage": 1000},
        )

        anomalies = collector.detect_anomalies(sensitivity=2.0)

        token_anomalies = [a for a in anomalies if a.metric_name == "token_usage"]
        assert len(token_anomalies) > 0
        assert token_anomalies[0].value == 1000.0

    # Correlation Analysis Tests

    def test_analyze_metric_correlation_insufficient_data(self, collector):
        """Test correlation analysis with insufficient data."""
        # Add minimal data
        for i in range(5):
            collector.collect(
                f"task-{i}", {"response_time": 1.0, "success": True, "token_usage": 100}
            )

        result = collector.analyze_metric_correlation("response_time", "token_usage")

        assert result.strength == "insufficient_data"
        assert result.direction == "none"

    def test_analyze_metric_correlation_valid(self, collector):
        """Test correlation analysis with valid data."""
        # Add data with positive correlation (higher token usage = higher response time)
        for i in range(20):
            token_usage = 100 + i * 10  # Increasing token usage
            response_time = 1.0 + i * 0.1  # Increasing response time
            collector.collect(
                f"task-{i}",
                {
                    "response_time": response_time,
                    "success": True,
                    "token_usage": token_usage,
                },
            )

        result = collector.analyze_metric_correlation("response_time", "token_usage")

        assert result.correlation_coefficient > 0.5  # Should be strongly positive
        assert result.strength in ["moderate", "strong"]
        assert result.direction == "positive"

    def test_analyze_metric_correlation_invalid_metrics(self, collector_with_data):
        """Test correlation analysis with invalid metric names."""
        result = collector_with_data.analyze_metric_correlation(
            "invalid_metric", "response_time"
        )

        # With invalid metric, we get empty values list, so insufficient data
        assert result.strength in ["invalid_metrics", "insufficient_data"]
        assert result.direction == "none"

    # Export Format Tests

    def test_export_to_csv(self, collector_with_data):
        """Test CSV export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "metrics_export.csv")

            collector_with_data.export_to_csv(csv_path)

            # Verify file creation and content
            assert os.path.exists(csv_path)

            with open(csv_path, "r") as f:
                content = f.read()
                assert "timestamp" in content
                assert "task_id" in content
                assert "agent_id" in content
                assert "response_time" in content

    def test_export_to_json(self, collector_with_data):
        """Test JSON export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "metrics_export.json")

            export_data = collector_with_data.export_to_json(json_path)

            # Verify file creation
            assert os.path.exists(json_path)

            # Verify export data structure
            assert "export_timestamp" in export_data
            assert "summary" in export_data
            assert "agent_breakdown" in export_data
            assert "custom_metrics" in export_data

            # Verify file content
            with open(json_path, "r") as f:
                file_data = json.load(f)
                assert file_data["summary"]["total_requests"] > 0

    def test_export_prometheus(self, collector_with_data):
        """Test Prometheus format export."""
        prometheus_output = collector_with_data.export_prometheus()

        # Verify Prometheus format
        assert "# HELP goose_requests_total" in prometheus_output
        assert "# TYPE goose_requests_total counter" in prometheus_output
        assert 'goose_requests_total{window="1h"}' in prometheus_output
        assert 'goose_success_rate{window="1h"}' in prometheus_output
        assert 'goose_response_time_seconds{window="1h"}' in prometheus_output

        # Should include agent-specific metrics
        assert 'agent="agent1"' in prometheus_output
        assert 'agent="agent2"' in prometheus_output

    # Data Retention Policy Tests

    def test_set_retention_policy(self, collector):
        """Test setting data retention policy."""
        original_retention = collector.data_retention_days

        collector.set_retention_policy(14)

        assert collector.data_retention_days == 14
        assert collector.data_retention_days != original_retention

    def test_retention_policy_cleanup(self, collector):
        """Test that retention policy actually cleans up old data."""
        # Set very short retention period
        collector.set_retention_policy(0)  # 0 days = immediate cleanup

        # Add data point
        collector.collect(
            "task-1", {"response_time": 1.0, "success": True, "token_usage": 100}
        )
        assert len(collector.data_points) == 1

        # Trigger cleanup (normally happens automatically)
        collector._cleanup_old_data()

        # Old data should be removed
        assert len(collector.data_points) == 0

    # Real-time Streaming Tests

    def test_subscribe_to_stream(self, collector):
        """Test subscribing to real-time metrics stream."""
        received_data = []

        def callback(data_point):
            received_data.append(data_point)

        collector.subscribe_to_stream(callback)

        # Collect some metrics
        collector.collect(
            "stream-task", {"response_time": 1.0, "success": True, "token_usage": 100}
        )

        # Callback should have been called
        assert len(received_data) == 1
        assert received_data[0].task_id == "stream-task"

    def test_unsubscribe_from_stream(self, collector):
        """Test unsubscribing from metrics stream."""
        received_data = []

        def callback(data_point):
            received_data.append(data_point)

        collector.subscribe_to_stream(callback)
        collector.collect(
            "task-1", {"response_time": 1.0, "success": True, "token_usage": 100}
        )
        assert len(received_data) == 1

        # Unsubscribe and collect more data
        collector.unsubscribe_from_stream(callback)
        collector.collect(
            "task-2", {"response_time": 1.0, "success": True, "token_usage": 100}
        )

        # Should not receive new data
        assert len(received_data) == 1

    def test_stream_error_handling(self, collector):
        """Test that streaming errors don't break the collector."""

        def failing_callback(data_point):
            raise Exception("Callback error")

        collector.subscribe_to_stream(failing_callback)

        # Should not raise exception despite callback failure
        collector.collect(
            "task-1", {"response_time": 1.0, "success": True, "token_usage": 100}
        )

        # Data should still be collected
        assert len(collector.data_points) == 1

    # Integration Tests

    def test_enhanced_cleanup_with_agents(self, collector):
        """Test cleanup properly handles agent metrics."""
        # Add data for multiple agents
        collector.collect(
            "task-1",
            {"response_time": 1.0, "success": True, "token_usage": 100},
            "agent1",
        )
        collector.collect(
            "task-2",
            {"response_time": 1.0, "success": True, "token_usage": 100},
            "agent2",
        )

        assert len(collector.agent_metrics) == 2

        # Set immediate cleanup
        collector.set_retention_policy(0)
        collector._cleanup_old_data()

        # Agent metrics should be cleaned up too
        assert len(collector.agent_metrics) == 0

    def test_comprehensive_stats_with_new_features(self, collector_with_data):
        """Test that stats include information about new features."""
        stats = collector_with_data.get_stats()

        # Should include basic stats
        assert "data_points_stored" in stats
        assert "current_metrics" in stats

        # Test agent metrics
        agent1_metrics = collector_with_data.get_agent_metrics("agent1")
        agent2_metrics = collector_with_data.get_agent_metrics("agent2")

        assert agent1_metrics.total_requests > 0
        assert agent2_metrics.total_requests > 0

    # Performance Tests for New Features

    def test_persistence_performance(self, collector_with_data):
        """Test that persistence operations are reasonably fast."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = os.path.join(temp_dir, "perf_test")

            start_time = time.time()

            # Should complete in reasonable time
            asyncio.run(collector_with_data.persist_to_disk(persist_path))

            persist_time = time.time() - start_time
            assert persist_time < 1.0, f"Persistence took {persist_time:.3f}s, too slow"

    def test_anomaly_detection_performance(self, collector):
        """Test anomaly detection performance with larger dataset."""
        # Add substantial data
        for i in range(100):
            collector.collect(
                f"task-{i}",
                {
                    "response_time": 1.0 + (i % 10) * 0.1,
                    "success": True,
                    "token_usage": 100,
                },
            )

        start_time = time.time()

        anomalies = collector.detect_anomalies()

        detection_time = time.time() - start_time
        assert (
            detection_time < 0.1
        ), f"Anomaly detection took {detection_time:.3f}s, too slow"

    def test_correlation_analysis_performance(self, collector):
        """Test correlation analysis performance."""
        # Add data
        for i in range(100):
            collector.collect(
                f"task-{i}",
                {
                    "response_time": 1.0 + i * 0.01,
                    "success": True,
                    "token_usage": 100 + i,
                },
            )

        start_time = time.time()

        result = collector.analyze_metric_correlation("response_time", "token_usage")

        analysis_time = time.time() - start_time
        assert (
            analysis_time < 0.1
        ), f"Correlation analysis took {analysis_time:.3f}s, too slow"

    # Edge Cases and Error Handling

    def test_collect_with_agent_id_metadata_separation(self, collector):
        """Test that agent_id doesn't interfere with metadata."""
        metrics = {
            "response_time": 1.5,
            "success": True,
            "token_usage": 100,
            "agent_id": "should_not_be_in_metadata",  # This should be filtered out
            "custom_field": "should_be_in_metadata",
        }

        collector.collect("test-task", metrics, "actual-agent-id")

        data_point = collector.data_points[0]
        assert data_point.agent_id == "actual-agent-id"
        assert "agent_id" not in data_point.metadata
        assert data_point.metadata["custom_field"] == "should_be_in_metadata"

    def test_empty_agent_metrics_handling(self, collector):
        """Test handling of empty agent metrics after cleanup."""
        collector.collect(
            "task-1",
            {"response_time": 1.0, "success": True, "token_usage": 100},
            "temp-agent",
        )

        assert "temp-agent" in collector.agent_metrics

        # Force cleanup that removes all data
        collector.set_retention_policy(0)
        collector._cleanup_old_data()

        # Empty agent metrics should be removed
        assert "temp-agent" not in collector.agent_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
