"""
Comprehensive tests for MetricsCollector.
Tests sliding window aggregation, trigger detection, and performance.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from monitoring.metrics import MetricsCollector, MetricsData, MetricWindow


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
