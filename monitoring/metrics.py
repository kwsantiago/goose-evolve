"""
Metrics collection and analysis for Goose Evolve.
Provides sliding window aggregation and evolution trigger detection.
"""

import asyncio
import csv
import json
import logging
import os
import pickle
import statistics

# Import from docs/architecture/interfaces.py
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
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
    agent_id: Optional[str] = None  # For multi-agent support


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


@dataclass
class Anomaly:
    """Detected statistical anomaly in metrics."""

    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # "low", "medium", "high"
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationResult:
    """Correlation analysis between two metrics."""

    metric1: str
    metric2: str
    correlation_coefficient: float
    p_value: float
    strength: str  # "weak", "moderate", "strong"
    direction: str  # "positive", "negative", "none"


@dataclass
class CustomMetric:
    """Custom metric definition."""

    name: str
    aggregator: Callable[[List[float]], float]
    description: str
    unit: str = ""


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
        data_retention_days: int = 30,
    ):
        self.window_1hr = window_1hr
        self.window_24hr = window_24hr
        self.max_data_points = max_data_points
        self.enable_langfuse = enable_langfuse
        self.data_retention_days = data_retention_days

        # In-memory storage with deque for efficient operations
        self.data_points: Deque[MetricsData] = deque(maxlen=max_data_points)

        # Multi-agent support
        self.agent_metrics: Dict[str, Deque[MetricsData]] = defaultdict(
            lambda: deque(maxlen=max_data_points)
        )

        # Custom metrics registry
        self.custom_metrics: Dict[str, CustomMetric] = {}
        self._register_default_custom_metrics()

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

        # Streaming subscribers
        self._streaming_subscribers: Set[Callable] = set()

        logger.info(
            f"MetricsCollector initialized with {window_1hr} and {window_24hr} windows, "
            f"retention={data_retention_days} days"
        )

    def collect(
        self, task_id: str, metrics: Dict[str, Any], agent_id: Optional[str] = None
    ) -> None:
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
            agent_id: Optional agent identifier for multi-agent support
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
                agent_id=agent_id,
                metadata={
                    k: v
                    for k, v in metrics.items()
                    if k
                    not in [
                        "response_time",
                        "success",
                        "token_usage",
                        "error_type",
                        "agent_id",
                    ]
                },
            )

            # Add to storage (thread-safe with deque)
            self.data_points.append(data_point)

            # Add to agent-specific storage if agent_id provided
            if agent_id:
                self.agent_metrics[agent_id].append(data_point)

            # Notify streaming subscribers
            self._notify_subscribers(data_point)

            # Periodic cleanup to maintain performance
            if time.time() - self._last_cleanup > 300:  # 5 minutes
                self._cleanup_old_data()
                self._last_cleanup = time.time()

            logger.debug(
                f"Collected metrics for task {task_id}: {response_time:.3f}s, "
                f"success={success}, tokens={token_usage}, agent={agent_id or 'global'}"
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
        """Clean up old cached data and enforce retention policy."""
        # Clear expired cache entries
        now = time.time()
        expired_keys = [
            key
            for key, (_, cache_time) in self._cached_windows.items()
            if now - cache_time > self._cache_ttl * 2
        ]
        for key in expired_keys:
            del self._cached_windows[key]

        # Enforce data retention policy
        cutoff_time = datetime.now() - timedelta(days=self.data_retention_days)
        original_count = len(self.data_points)

        # Filter out old data points
        self.data_points = deque(
            (dp for dp in self.data_points if dp.timestamp >= cutoff_time),
            maxlen=self.max_data_points,
        )

        # Clean up agent metrics
        for agent_id in list(self.agent_metrics.keys()):
            original_agent_count = len(self.agent_metrics[agent_id])
            self.agent_metrics[agent_id] = deque(
                (
                    dp
                    for dp in self.agent_metrics[agent_id]
                    if dp.timestamp >= cutoff_time
                ),
                maxlen=self.max_data_points,
            )
            # Remove empty agent metrics
            if not self.agent_metrics[agent_id]:
                del self.agent_metrics[agent_id]

        cleaned_count = original_count - len(self.data_points)
        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} old data points (retention: {self.data_retention_days} days)"
            )

        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear_data(self) -> None:
        """Clear all collected data (for testing/reset)."""
        self.data_points.clear()
        self.agent_metrics.clear()
        self._cached_windows.clear()
        logger.info("Cleared all metrics data")

    # Production-Ready Features

    async def persist_to_disk(self, path: str) -> None:
        """
        Save metrics data to disk for recovery.

        Args:
            path: File path to save data to
        """
        try:
            persist_path = Path(path)
            persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            data_to_save = {
                "data_points": [asdict(dp) for dp in self.data_points],
                "agent_metrics": {
                    agent_id: [asdict(dp) for dp in metrics]
                    for agent_id, metrics in self.agent_metrics.items()
                },
                "thresholds": self.thresholds,
                "custom_metrics": {
                    name: {
                        "name": metric.name,
                        "description": metric.description,
                        "unit": metric.unit,
                    }
                    for name, metric in self.custom_metrics.items()
                },
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "window_1hr_seconds": self.window_1hr.total_seconds(),
                    "window_24hr_seconds": self.window_24hr.total_seconds(),
                    "max_data_points": self.max_data_points,
                    "data_retention_days": self.data_retention_days,
                },
            }

            # Save as pickle for performance and JSON for portability
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(data_to_save, f)

            with open(f"{path}.json", "w") as f:
                # Convert datetime objects to strings for JSON
                json_data = json.loads(json.dumps(data_to_save, default=str))
                json.dump(json_data, f, indent=2)

            logger.info(f"Persisted {len(self.data_points)} data points to {path}")

        except Exception as e:
            logger.error(f"Error persisting metrics to {path}: {e}")
            raise

    async def restore_from_disk(self, path: str) -> None:
        """
        Load historical metrics from disk.

        Args:
            path: File path to load data from
        """
        try:
            pkl_path = f"{path}.pkl"
            json_path = f"{path}.json"

            # Prefer pickle for performance, fallback to JSON
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
            elif os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
            else:
                raise FileNotFoundError(f"No metrics file found at {path}")

            # Restore data points
            self.data_points.clear()
            for dp_dict in data["data_points"]:
                # Convert string timestamp back to datetime
                if isinstance(dp_dict["timestamp"], str):
                    dp_dict["timestamp"] = datetime.fromisoformat(dp_dict["timestamp"])
                dp = MetricsData(**dp_dict)
                self.data_points.append(dp)

            # Restore agent metrics
            self.agent_metrics.clear()
            for agent_id, metrics_list in data.get("agent_metrics", {}).items():
                agent_deque: Deque[MetricsData] = deque(maxlen=self.max_data_points)
                for dp_dict in metrics_list:
                    if isinstance(dp_dict["timestamp"], str):
                        dp_dict["timestamp"] = datetime.fromisoformat(
                            dp_dict["timestamp"]
                        )
                    dp = MetricsData(**dp_dict)
                    agent_deque.append(dp)
                self.agent_metrics[agent_id] = agent_deque

            # Restore thresholds
            if "thresholds" in data:
                self.thresholds.update(data["thresholds"])

            # Note: Custom metrics with functions can't be serialized,
            # so we only restore the metadata
            logger.info(f"Restored {len(self.data_points)} data points from {path}")

        except Exception as e:
            logger.error(f"Error restoring metrics from {path}: {e}")
            raise

    def register_custom_metric(
        self,
        name: str,
        aggregator: Callable[[List[float]], float],
        description: str = "",
        unit: str = "",
    ) -> None:
        """
        Add a custom metric type.

        Args:
            name: Unique name for the metric
            aggregator: Function to aggregate a list of values
            description: Human-readable description
            unit: Unit of measurement
        """
        self.custom_metrics[name] = CustomMetric(
            name=name, aggregator=aggregator, description=description, unit=unit
        )
        logger.info(f"Registered custom metric: {name}")

    def detect_anomalies(
        self, sensitivity: float = 2.0, window_duration: Optional[timedelta] = None
    ) -> List[Anomaly]:
        """
        Find statistical outliers in metrics data.

        Args:
            sensitivity: Standard deviations threshold for anomaly detection
            window_duration: Time window to analyze (default: 1hr)

        Returns:
            List of detected anomalies
        """
        window_duration = window_duration or self.window_1hr
        end_time = datetime.now()
        start_time = end_time - window_duration

        # Filter data for the window
        window_data = [
            dp for dp in self.data_points if start_time <= dp.timestamp <= end_time
        ]

        if len(window_data) < 10:  # Need minimum data for statistical analysis
            return []

        anomalies = []

        # Analyze response times
        response_times = [dp.response_time for dp in window_data]
        if len(response_times) > 1:
            mean_rt = statistics.mean(response_times)
            stdev_rt = statistics.stdev(response_times)
            threshold = sensitivity * stdev_rt

            for dp in window_data:
                if abs(dp.response_time - mean_rt) > threshold:
                    severity = (
                        "high"
                        if abs(dp.response_time - mean_rt) > 3 * stdev_rt
                        else "medium"
                    )
                    anomalies.append(
                        Anomaly(
                            metric_name="response_time",
                            value=dp.response_time,
                            expected_range=(mean_rt - threshold, mean_rt + threshold),
                            severity=severity,
                            timestamp=dp.timestamp,
                            context={"task_id": dp.task_id, "agent_id": dp.agent_id},
                        )
                    )

        # Analyze token usage
        token_usage = [dp.token_usage for dp in window_data if dp.token_usage > 0]
        if len(token_usage) > 1:
            mean_tokens = statistics.mean(token_usage)
            stdev_tokens = statistics.stdev(token_usage)
            threshold = sensitivity * stdev_tokens

            for dp in window_data:
                if dp.token_usage > 0 and abs(dp.token_usage - mean_tokens) > threshold:
                    severity = (
                        "high"
                        if abs(dp.token_usage - mean_tokens) > 3 * stdev_tokens
                        else "medium"
                    )
                    anomalies.append(
                        Anomaly(
                            metric_name="token_usage",
                            value=float(dp.token_usage),
                            expected_range=(
                                mean_tokens - threshold,
                                mean_tokens + threshold,
                            ),
                            severity=severity,
                            timestamp=dp.timestamp,
                            context={"task_id": dp.task_id, "agent_id": dp.agent_id},
                        )
                    )

        logger.info(
            f"Detected {len(anomalies)} anomalies with sensitivity {sensitivity}"
        )
        return anomalies

    def analyze_metric_correlation(
        self, metric1: str, metric2: str, window_duration: Optional[timedelta] = None
    ) -> CorrelationResult:
        """
        Analyze correlation between two metrics.

        Args:
            metric1: First metric name ("response_time", "token_usage", "success_rate")
            metric2: Second metric name
            window_duration: Time window to analyze (default: 24hr)

        Returns:
            Correlation analysis result
        """
        window_duration = window_duration or self.window_24hr
        end_time = datetime.now()
        start_time = end_time - window_duration

        # Filter data for the window
        window_data = [
            dp for dp in self.data_points if start_time <= dp.timestamp <= end_time
        ]

        if len(window_data) < 10:
            return CorrelationResult(
                metric1=metric1,
                metric2=metric2,
                correlation_coefficient=0.0,
                p_value=1.0,
                strength="insufficient_data",
                direction="none",
            )

        # Extract metric values
        def get_metric_values(metric_name: str) -> List[float]:
            if metric_name == "response_time":
                return [dp.response_time for dp in window_data]
            elif metric_name == "token_usage":
                return [float(dp.token_usage) for dp in window_data]
            elif metric_name == "success_rate":
                # Calculate success rate in sliding windows
                return [1.0 if dp.success else 0.0 for dp in window_data]
            else:
                return []

        values1 = get_metric_values(metric1)
        values2 = get_metric_values(metric2)

        if not values1 or not values2:
            return CorrelationResult(
                metric1=metric1,
                metric2=metric2,
                correlation_coefficient=0.0,
                p_value=1.0,
                strength="invalid_metrics",
                direction="none",
            )

        if len(values1) != len(values2):
            return CorrelationResult(
                metric1=metric1,
                metric2=metric2,
                correlation_coefficient=0.0,
                p_value=1.0,
                strength="invalid_metrics",
                direction="none",
            )

        # Calculate Pearson correlation (manual implementation for Python < 3.10)
        try:
            if len(values1) < 2:
                corr_coef = 0.0
            else:
                mean1 = statistics.mean(values1)
                mean2 = statistics.mean(values2)

                numerator = sum(
                    (x - mean1) * (y - mean2) for x, y in zip(values1, values2)
                )
                sum_sq1 = sum((x - mean1) ** 2 for x in values1)
                sum_sq2 = sum((y - mean2) ** 2 for y in values2)

                denominator = (sum_sq1 * sum_sq2) ** 0.5

                if denominator == 0:
                    corr_coef = 0.0
                else:
                    corr_coef = numerator / denominator
        except (statistics.StatisticsError, ZeroDivisionError):
            corr_coef = 0.0

        # Determine strength and direction
        abs_corr = abs(corr_coef)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.1:
            strength = "weak"
        else:
            strength = "none"

        direction = (
            "positive"
            if corr_coef > 0.1
            else "negative" if corr_coef < -0.1 else "none"
        )

        # Simple p-value approximation (for production, use scipy.stats)
        n = len(values1)
        t_stat = (
            corr_coef * ((n - 2) / (1 - corr_coef**2)) ** 0.5
            if abs(corr_coef) < 1
            else float("inf")
        )
        p_value = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / (abs(t_stat) + n))))

        return CorrelationResult(
            metric1=metric1,
            metric2=metric2,
            correlation_coefficient=corr_coef,
            p_value=p_value,
            strength=strength,
            direction=direction,
        )

    def export_to_csv(
        self, path: str, window_duration: Optional[timedelta] = None
    ) -> None:
        """Export metrics to CSV format."""
        window_duration = window_duration or self.window_24hr
        end_time = datetime.now()
        start_time = end_time - window_duration

        window_data = [
            dp for dp in self.data_points if start_time <= dp.timestamp <= end_time
        ]

        with open(path, "w", newline="") as csvfile:
            fieldnames = [
                "timestamp",
                "task_id",
                "agent_id",
                "response_time",
                "success",
                "token_usage",
                "error_type",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for dp in window_data:
                writer.writerow(
                    {
                        "timestamp": dp.timestamp.isoformat(),
                        "task_id": dp.task_id,
                        "agent_id": dp.agent_id or "",
                        "response_time": dp.response_time,
                        "success": dp.success,
                        "token_usage": dp.token_usage,
                        "error_type": dp.error_type or "",
                    }
                )

        logger.info(f"Exported {len(window_data)} records to CSV: {path}")

    def export_to_json(
        self, path: str, window_duration: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Export metrics to JSON format."""
        window_duration = window_duration or self.window_24hr
        window = self.get_window_metrics(window_duration)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "window_duration_hours": window_duration.total_seconds() / 3600,
            "summary": {
                "total_requests": window.total_requests,
                "success_rate": window.success_rate,
                "avg_response_time": window.avg_response_time,
                "total_tokens": window.total_tokens,
                "error_counts": window.error_counts,
            },
            "agent_breakdown": self._get_agent_breakdown(window_duration),
            "custom_metrics": self._calculate_custom_metrics(window_duration),
        }

        with open(path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported metrics to JSON: {path}")
        return export_data

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        window_1hr = self.get_window_metrics(self.window_1hr)
        window_24hr = self.get_window_metrics(self.window_24hr)

        prometheus_output = []
        timestamp = int(time.time() * 1000)  # Prometheus uses milliseconds

        # Basic metrics
        prometheus_output.extend(
            [
                f"# HELP goose_requests_total Total number of requests",
                f"# TYPE goose_requests_total counter",
                f'goose_requests_total{{window="1h"}} {window_1hr.total_requests} {timestamp}',
                f'goose_requests_total{{window="24h"}} {window_24hr.total_requests} {timestamp}',
                "",
                f"# HELP goose_success_rate Success rate of requests",
                f"# TYPE goose_success_rate gauge",
                f'goose_success_rate{{window="1h"}} {window_1hr.success_rate} {timestamp}',
                f'goose_success_rate{{window="24h"}} {window_24hr.success_rate} {timestamp}',
                "",
                f"# HELP goose_response_time_seconds Average response time in seconds",
                f"# TYPE goose_response_time_seconds gauge",
                f'goose_response_time_seconds{{window="1h"}} {window_1hr.avg_response_time} {timestamp}',
                f'goose_response_time_seconds{{window="24h"}} {window_24hr.avg_response_time} {timestamp}',
                "",
                f"# HELP goose_tokens_total Total token usage",
                f"# TYPE goose_tokens_total counter",
                f'goose_tokens_total{{window="1h"}} {window_1hr.total_tokens} {timestamp}',
                f'goose_tokens_total{{window="24h"}} {window_24hr.total_tokens} {timestamp}',
            ]
        )

        # Agent-specific metrics
        for agent_id in self.agent_metrics.keys():
            agent_window = self._get_agent_window_metrics(agent_id, self.window_1hr)
            prometheus_output.extend(
                [
                    f'goose_requests_total{{window="1h",agent="{agent_id}"}} {agent_window.total_requests} {timestamp}',
                    f'goose_success_rate{{window="1h",agent="{agent_id}"}} {agent_window.success_rate} {timestamp}',
                    f'goose_response_time_seconds{{window="1h",agent="{agent_id}"}} {agent_window.avg_response_time} {timestamp}',
                ]
            )

        return "\n".join(prometheus_output)

    def set_retention_policy(self, days: int) -> None:
        """
        Configure data retention policy.

        Args:
            days: Number of days to retain data
        """
        self.data_retention_days = days
        logger.info(f"Set data retention policy to {days} days")

        # Trigger immediate cleanup based on new policy
        self._cleanup_old_data()

    def subscribe_to_stream(self, callback: Callable[[MetricsData], None]) -> None:
        """
        Subscribe to real-time metrics stream.

        Args:
            callback: Function to call with each new metric data point
        """
        self._streaming_subscribers.add(callback)
        logger.info(
            f"Added streaming subscriber, total: {len(self._streaming_subscribers)}"
        )

    def unsubscribe_from_stream(self, callback: Callable[[MetricsData], None]) -> None:
        """Remove a streaming subscriber."""
        self._streaming_subscribers.discard(callback)
        logger.info(
            f"Removed streaming subscriber, total: {len(self._streaming_subscribers)}"
        )

    def get_agent_metrics(
        self, agent_id: str, window_duration: Optional[timedelta] = None
    ) -> MetricWindow:
        """
        Get metrics for a specific agent.

        Args:
            agent_id: Agent identifier
            window_duration: Time window (default: 1hr)

        Returns:
            MetricWindow with agent-specific data
        """
        return self._get_agent_window_metrics(
            agent_id, window_duration or self.window_1hr
        )

    def get_custom_metric_value(
        self, metric_name: str, window_duration: Optional[timedelta] = None
    ) -> Optional[float]:
        """
        Calculate value for a custom metric.

        Args:
            metric_name: Name of registered custom metric
            window_duration: Time window (default: 1hr)

        Returns:
            Calculated metric value or None if metric not found
        """
        if metric_name not in self.custom_metrics:
            return None

        window_duration = window_duration or self.window_1hr
        end_time = datetime.now()
        start_time = end_time - window_duration

        window_data = [
            dp for dp in self.data_points if start_time <= dp.timestamp <= end_time
        ]

        # Extract values based on metric type (simplified)
        values = [dp.response_time for dp in window_data]  # Default to response time

        if not values:
            return 0.0

        return self.custom_metrics[metric_name].aggregator(values)

    # Helper methods

    def _register_default_custom_metrics(self) -> None:
        """Register default custom metrics."""
        self.register_custom_metric(
            "p95_response_time",
            lambda values: sorted(values)[int(0.95 * len(values))] if values else 0.0,
            "95th percentile response time",
            "seconds",
        )
        self.register_custom_metric(
            "p99_response_time",
            lambda values: sorted(values)[int(0.99 * len(values))] if values else 0.0,
            "99th percentile response time",
            "seconds",
        )
        self.register_custom_metric(
            "median_response_time",
            lambda values: statistics.median(values) if values else 0.0,
            "Median response time",
            "seconds",
        )

    def _notify_subscribers(self, data_point: MetricsData) -> None:
        """Notify streaming subscribers of new data."""
        for callback in self._streaming_subscribers:
            try:
                callback(data_point)
            except Exception as e:
                logger.error(f"Error notifying streaming subscriber: {e}")

    def _get_agent_window_metrics(
        self, agent_id: str, window_duration: timedelta
    ) -> MetricWindow:
        """Get window metrics for a specific agent."""
        end_time = datetime.now()
        start_time = end_time - window_duration

        agent_data = self.agent_metrics.get(agent_id, deque())
        window_data = [
            dp for dp in agent_data if start_time <= dp.timestamp <= end_time
        ]

        if not window_data:
            return MetricWindow(
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                success_count=0,
                avg_response_time=0.0,
                total_tokens=0,
            )

        total_requests = len(window_data)
        success_count = sum(1 for dp in window_data if dp.success)
        total_response_time = sum(dp.response_time for dp in window_data)
        total_tokens = sum(dp.token_usage for dp in window_data)

        error_counts: Dict[str, int] = defaultdict(int)
        for dp in window_data:
            if not dp.success and dp.error_type:
                error_counts[dp.error_type] += 1

        return MetricWindow(
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            success_count=success_count,
            avg_response_time=total_response_time / total_requests,
            total_tokens=total_tokens,
            error_counts=dict(error_counts),
        )

    def _get_agent_breakdown(
        self, window_duration: timedelta
    ) -> Dict[str, Dict[str, Any]]:
        """Get metrics breakdown by agent."""
        breakdown = {}
        for agent_id in self.agent_metrics.keys():
            window = self._get_agent_window_metrics(agent_id, window_duration)
            breakdown[agent_id] = {
                "total_requests": window.total_requests,
                "success_rate": window.success_rate,
                "avg_response_time": window.avg_response_time,
                "total_tokens": window.total_tokens,
            }
        return breakdown

    def _calculate_custom_metrics(self, window_duration: timedelta) -> Dict[str, float]:
        """Calculate all custom metrics for the window."""
        custom_values = {}
        for name in self.custom_metrics.keys():
            value = self.get_custom_metric_value(name, window_duration)
            if value is not None:
                custom_values[name] = value
        return custom_values
