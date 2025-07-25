"""
Metrics collection and analysis for Goose Evolve.
Provides sliding window aggregation and evolution trigger detection.
"""

import asyncio
import logging
import os

# Import from docs/architecture/interfaces.py
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "docs", "architecture"))
from evolution.interfaces import EvolutionTriggerEvent

logger = logging.getLogger(__name__)


@dataclass
class MetricsData:
    """Individual metrics data point."""

    task_id: str
    timestamp: datetime
    response_time: float  # seconds
    success: bool
    token_usage: int
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricWindow:
    """Aggregated metrics for a time window."""

    start_time: datetime
    end_time: datetime
    total_requests: int
    success_count: int
    avg_response_time: float
    total_tokens: int
    error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.success_count / max(self.total_requests, 1)

    @property
    def avg_tokens_per_request(self) -> float:
        """Calculate average tokens per request."""
        return self.total_tokens / max(self.total_requests, 1)


class MetricsCollector:
    """
    Collects and analyzes metrics for evolution triggering.

    Features:
    - In-memory sliding window storage
    - Configurable time windows (1hr, 24hr)
    - Threshold-based trigger detection
    - Optional Langfuse integration
    - Low overhead design (<5% performance impact)
    """

    def __init__(
        self,
        window_1hr: timedelta = timedelta(hours=1),
        window_24hr: timedelta = timedelta(hours=24),
        max_data_points: int = 10000,
        enable_langfuse: bool = False,
    ):
        self.window_1hr = window_1hr
        self.window_24hr = window_24hr
        self.max_data_points = max_data_points
        self.enable_langfuse = enable_langfuse

        # In-memory storage with deque for efficient operations
        self.data_points: Deque[MetricsData] = deque(maxlen=max_data_points)

        # Trigger thresholds (configurable)
        self.thresholds = {
            "min_success_rate_1hr": 0.8,
            "max_response_time_1hr": 2.0,  # seconds
            "min_requests_1hr": 10,  # minimum requests to trigger
            "success_rate_drop_24hr": 0.1,  # 10% drop triggers evolution
            "response_time_increase_24hr": 0.2,  # 20% increase triggers evolution
        }

        # Cache for performance
        self._cached_windows: Dict[str, tuple] = {}
        self._cache_ttl = 60  # seconds
        self._last_cleanup = time.time()

        logger.info(
            f"MetricsCollector initialized with {window_1hr} and {window_24hr} windows"
        )

    def collect(self, task_id: str, metrics: Dict[str, Any]) -> None:
        """
        Collect metrics for a task execution.

        Args:
            task_id: Unique identifier for the task
            metrics: Dictionary containing:
                - response_time: float (seconds)
                - success: bool
                - token_usage: int
                - error_type: Optional[str]
                - Additional metadata
        """
        try:
            # Extract required metrics
            response_time = float(metrics.get("response_time", 0.0))
            success = bool(metrics.get("success", False))
            token_usage = int(metrics.get("token_usage", 0))
            error_type = metrics.get("error_type")

            # Create data point
            data_point = MetricsData(
                task_id=task_id,
                timestamp=datetime.now(),
                response_time=response_time,
                success=success,
                token_usage=token_usage,
                error_type=error_type,
                metadata={
                    k: v
                    for k, v in metrics.items()
                    if k
                    not in ["response_time", "success", "token_usage", "error_type"]
                },
            )

            # Add to storage (thread-safe with deque)
            self.data_points.append(data_point)

            # Periodic cleanup to maintain performance
            if time.time() - self._last_cleanup > 300:  # 5 minutes
                self._cleanup_old_data()
                self._last_cleanup = time.time()

            logger.debug(
                f"Collected metrics for task {task_id}: {response_time:.3f}s, "
                f"success={success}, tokens={token_usage}"
            )

        except Exception as e:
            logger.error(f"Error collecting metrics for task {task_id}: {e}")

    def get_window_metrics(self, window_duration: timedelta) -> MetricWindow:
        """
        Get aggregated metrics for a specific time window.

        Args:
            window_duration: Time window to analyze

        Returns:
            MetricWindow with aggregated data
        """
        cache_key = f"{window_duration.total_seconds()}"
        now = time.time()

        # Check cache first
        if cache_key in self._cached_windows:
            cached_data, cache_time = self._cached_windows[cache_key]
            if now - cache_time < self._cache_ttl:
                return cached_data

        # Calculate window
        end_time = datetime.now()
        start_time = end_time - window_duration

        # Filter data points in window
        window_data = [
            dp for dp in self.data_points if start_time <= dp.timestamp <= end_time
        ]

        if not window_data:
            window = MetricWindow(
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                success_count=0,
                avg_response_time=0.0,
                total_tokens=0,
            )
        else:
            # Aggregate metrics
            total_requests = len(window_data)
            success_count = sum(1 for dp in window_data if dp.success)
            total_response_time = sum(dp.response_time for dp in window_data)
            total_tokens = sum(dp.token_usage for dp in window_data)

            # Count errors by type
            error_counts: Dict[str, int] = defaultdict(int)
            for dp in window_data:
                if not dp.success and dp.error_type:
                    error_counts[dp.error_type] += 1

            window = MetricWindow(
                start_time=start_time,
                end_time=end_time,
                total_requests=total_requests,
                success_count=success_count,
                avg_response_time=total_response_time / total_requests,
                total_tokens=total_tokens,
                error_counts=dict(error_counts),
            )

        # Cache result
        self._cached_windows[cache_key] = (window, now)
        return window

    def check_evolution_triggers(self) -> Optional[EvolutionTriggerEvent]:
        """
        Check if current metrics indicate evolution should be triggered.

        Returns:
            EvolutionTriggerEvent if triggers are met, None otherwise
        """
        try:
            # Get current window metrics
            window_1hr = self.get_window_metrics(self.window_1hr)
            window_24hr = self.get_window_metrics(self.window_24hr)

            # Check minimum requests threshold
            if window_1hr.total_requests < self.thresholds["min_requests_1hr"]:
                logger.debug(
                    f"Insufficient requests in 1hr window: {window_1hr.total_requests}"
                )
                return None

            trigger_reasons = []

            # Check 1-hour thresholds
            if window_1hr.success_rate < self.thresholds["min_success_rate_1hr"]:
                trigger_reasons.append(
                    f"Low success rate: {window_1hr.success_rate:.2f}"
                )

            if window_1hr.avg_response_time > self.thresholds["max_response_time_1hr"]:
                trigger_reasons.append(
                    f"High response time: {window_1hr.avg_response_time:.2f}s"
                )

            # Check 24-hour trends (if we have enough data)
            if window_24hr.total_requests >= self.thresholds["min_requests_1hr"]:
                # Calculate baseline from earlier 24hr period
                baseline_end = datetime.now() - self.window_1hr
                baseline_start = baseline_end - self.window_24hr
                baseline_data = [
                    dp
                    for dp in self.data_points
                    if baseline_start <= dp.timestamp <= baseline_end
                ]

                if baseline_data:
                    baseline_success_rate = sum(
                        1 for dp in baseline_data if dp.success
                    ) / len(baseline_data)
                    baseline_response_time = sum(
                        dp.response_time for dp in baseline_data
                    ) / len(baseline_data)

                    # Check for degradation trends
                    success_drop = baseline_success_rate - window_1hr.success_rate
                    if success_drop > self.thresholds["success_rate_drop_24hr"]:
                        trigger_reasons.append(
                            f"Success rate dropped {success_drop:.2f}"
                        )

                    response_increase = (
                        window_1hr.avg_response_time - baseline_response_time
                    ) / baseline_response_time
                    if (
                        response_increase
                        > self.thresholds["response_time_increase_24hr"]
                    ):
                        trigger_reasons.append(
                            f"Response time increased {response_increase:.1%}"
                        )

            if trigger_reasons:
                logger.info(
                    f"Evolution triggers detected: {', '.join(trigger_reasons)}"
                )

                # Create metrics snapshot
                metrics_snapshot = {
                    "success_rate_1hr": window_1hr.success_rate,
                    "avg_response_time_1hr": window_1hr.avg_response_time,
                    "total_requests_1hr": float(window_1hr.total_requests),
                    "success_rate_24hr": window_24hr.success_rate,
                    "avg_response_time_24hr": window_24hr.avg_response_time,
                    "total_requests_24hr": float(window_24hr.total_requests),
                    "trigger_count": float(len(trigger_reasons)),
                }

                return EvolutionTriggerEvent(
                    trigger_type="threshold",
                    metrics_snapshot=metrics_snapshot,
                    timestamp=datetime.now(),
                    trigger_reasons=trigger_reasons,
                )

            return None

        except Exception as e:
            logger.error(f"Error checking evolution triggers: {e}")
            return None

    def export_to_langfuse(
        self, window_duration: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Export metrics to Langfuse for observability.

        Args:
            window_duration: Time window to export (default: 1hr)

        Returns:
            Dictionary suitable for Langfuse export
        """
        if not self.enable_langfuse:
            logger.warning("Langfuse export called but not enabled")
            return {}

        try:
            window_duration = window_duration or self.window_1hr
            window = self.get_window_metrics(window_duration)

            export_data = {
                "timestamp": datetime.now().isoformat(),
                "window_duration_hours": window_duration.total_seconds() / 3600,
                "metrics": {
                    "total_requests": window.total_requests,
                    "success_rate": window.success_rate,
                    "avg_response_time": window.avg_response_time,
                    "total_tokens": window.total_tokens,
                    "avg_tokens_per_request": window.avg_tokens_per_request,
                    "error_counts": window.error_counts,
                },
                "thresholds": self.thresholds,
                "system_info": {
                    "data_points_stored": len(self.data_points),
                    "cache_entries": len(self._cached_windows),
                },
            }

            logger.info(f"Exported {window.total_requests} metrics to Langfuse")
            return export_data

        except Exception as e:
            logger.error(f"Error exporting to Langfuse: {e}")
            return {}

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update trigger thresholds.

        Args:
            new_thresholds: Dictionary of threshold updates
        """
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated thresholds: {new_thresholds}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current collector statistics.

        Returns:
            Dictionary with collector stats and current metrics
        """
        window_1hr = self.get_window_metrics(self.window_1hr)
        window_24hr = self.get_window_metrics(self.window_24hr)

        return {
            "data_points_stored": len(self.data_points),
            "cache_entries": len(self._cached_windows),
            "thresholds": self.thresholds.copy(),
            "current_metrics": {
                "1hr_window": {
                    "total_requests": window_1hr.total_requests,
                    "success_rate": window_1hr.success_rate,
                    "avg_response_time": window_1hr.avg_response_time,
                    "total_tokens": window_1hr.total_tokens,
                },
                "24hr_window": {
                    "total_requests": window_24hr.total_requests,
                    "success_rate": window_24hr.success_rate,
                    "avg_response_time": window_24hr.avg_response_time,
                    "total_tokens": window_24hr.total_tokens,
                },
            },
        }

    def _cleanup_old_data(self) -> None:
        """Clean up old cached data to maintain performance."""
        # Clear expired cache entries
        now = time.time()
        expired_keys = [
            key
            for key, (_, cache_time) in self._cached_windows.items()
            if now - cache_time > self._cache_ttl * 2
        ]
        for key in expired_keys:
            del self._cached_windows[key]

        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear_data(self) -> None:
        """Clear all collected data (for testing/reset)."""
        self.data_points.clear()
        self._cached_windows.clear()
        logger.info("Cleared all metrics data")
