"""
Metrics collection for RobustSER-Tiny.

Provides a simple metrics collector that can export to JSONL, Prometheus,
or in-memory storage for dashboards.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable


@dataclass(frozen=True)
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    metric_type: str = "gauge"  # gauge, histogram, counter

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "labels": self.labels,
            "unit": self.unit,
            "description": self.description,
            "type": self.metric_type,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class MetricsCollector:
    """
    Thread-safe metrics collector with multiple export formats.

    Supports:
    - In-memory storage for real-time dashboards
    - JSONL file export for persistent logs
    - Prometheus exposition format
    - Callback hooks for custom integrations
    """

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        flush_interval: float = 10.0,
        max_memory_points: int = 10000,
    ):
        self._lock = threading.RLock()
        self._points: list[MetricPoint] = []
        self._jsonl_path = jsonl_path
        self._flush_interval = flush_interval
        self._max_memory_points = max_memory_points
        self._callbacks: list[Callable[[MetricPoint], None]] = []
        self._last_flush = time.time()

        if jsonl_path:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
        unit: str = "",
        description: str = "",
        metric_type: str = "gauge",
        timestamp: Optional[float] = None,
    ) -> MetricPoint:
        """Record a metric point."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp or time.time(),
            labels=labels or {},
            unit=unit,
            description=description,
            metric_type=metric_type,
        )

        with self._lock:
            self._points.append(point)
            # Trim old points if exceeding max
            if len(self._points) > self._max_memory_points:
                self._points = self._points[-self._max_memory_points:]

            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(point)
                except Exception:
                    pass  # Don't let callbacks break metrics

            # Auto-flush to JSONL
            if self._jsonl_path and time.time() - self._last_flush > self._flush_interval:
                self._flush_jsonl()

        return point

    def record_histogram(
        self,
        name: str,
        values: list[float],
        labels: Optional[dict[str, str]] = None,
        unit: str = "",
        description: str = "",
    ) -> list[MetricPoint]:
        """Record multiple values as a histogram (creates summary points)."""
        if not values:
            return []

        points = []
        now = time.time()
        base_labels = labels or {}

        # Record individual values
        for i, value in enumerate(values):
            point_labels = {**base_labels, "bucket": str(i)}
            points.append(self.record(
                name=f"{name}_bucket",
                value=value,
                labels=point_labels,
                unit=unit,
                description=description,
                metric_type="histogram",
                timestamp=now,
            ))

        # Record summary statistics
        points.append(self.record(
            name=f"{name}_count",
            value=len(values),
            labels=base_labels,
            unit="count",
            description=f"Count of {name}",
            metric_type="counter",
            timestamp=now,
        ))
        points.append(self.record(
            name=f"{name}_sum",
            value=sum(values),
            labels=base_labels,
            unit=unit,
            description=f"Sum of {name}",
            metric_type="counter",
            timestamp=now,
        ))
        points.append(self.record(
            name=f"{name}_mean",
            value=sum(values) / len(values),
            labels=base_labels,
            unit=unit,
            description=f"Mean of {name}",
            metric_type="gauge",
            timestamp=now,
        ))
        points.append(self.record(
            name=f"{name}_min",
            value=min(values),
            labels=base_labels,
            unit=unit,
            description=f"Min of {name}",
            metric_type="gauge",
            timestamp=now,
        ))
        points.append(self.record(
            name=f"{name}_max",
            value=max(values),
            labels=base_labels,
            unit=unit,
            description=f"Max of {name}",
            metric_type="gauge",
            timestamp=now,
        ))

        # Percentiles
        sorted_values = sorted(values)
        for p in [0.5, 0.9, 0.95, 0.99]:
            idx = int(len(sorted_values) * p)
            idx = min(idx, len(sorted_values) - 1)
            points.append(self.record(
                name=f"{name}_p{int(p*100)}",
                value=sorted_values[idx],
                labels=base_labels,
                unit=unit,
                description=f"P{int(p*100)} of {name}",
                metric_type="gauge",
                timestamp=now,
            ))

        return points

    def add_callback(self, callback: Callable[[MetricPoint], None]) -> None:
        """Add a callback to be invoked on each metric point."""
        with self._lock:
            self._callbacks.append(callback)

    def get_points(
        self,
        name: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        since: Optional[float] = None,
    ) -> list[MetricPoint]:
        """Get stored metric points with optional filtering."""
        with self._lock:
            points = self._points[:]

        if name:
            points = [p for p in points if p.name == name]
        if labels:
            points = [p for p in points if all(p.labels.get(k) == v for k, v in labels.items())]
        if since:
            points = [p for p in points if p.timestamp >= since]

        return points

    def get_latest(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
    ) -> Optional[MetricPoint]:
        """Get the most recent point for a metric."""
        points = self.get_points(name=name, labels=labels)
        return points[-1] if points else None

    def _flush_jsonl(self) -> None:
        """Flush points to JSONL file."""
        if not self._jsonl_path:
            return

        with self._lock:
            points_to_write = self._points[:]
            self._last_flush = time.time()

        try:
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                for point in points_to_write:
                    f.write(point.to_jsonl() + "\n")
        except Exception:
            pass  # Don't let IO errors break metrics

    def flush(self) -> None:
        """Force flush to JSONL."""
        self._flush_jsonl()

    def export_jsonl(self, path: Path) -> None:
        """Export all points to a JSONL file."""
        with self._lock:
            points = self._points[:]

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for point in points:
                f.write(point.to_jsonl() + "\n")

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        with self._lock:
            points = self._points[:]

        # Group by name and labels
        groups: dict[tuple[str, tuple[tuple[str, str], ...]], list[MetricPoint]] = defaultdict(list)
        for point in points:
            label_items = tuple(sorted(point.labels.items()))
            groups[(point.name, label_items)].append(point)

        lines = []
        for (name, label_items), group_points in groups.items():
            labels_str = ",".join(f'{k}="{v}"' for k, v in label_items) if label_items else ""
            labels_prefix = f"{{{labels_str}}}" if labels_str else ""

            # Get latest value for gauges
            latest = max(group_points, key=lambda p: p.timestamp)

            if latest.metric_type in ("gauge", "counter"):
                lines.append(f"# TYPE {name} {latest.metric_type}")
                lines.append(f"{name}{labels_prefix} {latest.value} {int(latest.timestamp * 1000)}")
            elif latest.metric_type == "histogram":
                # For histograms, export summary metrics
                count = len([p for p in group_points if p.name.endswith("_bucket")])
                sum_val = sum(p.value for p in group_points if p.name.endswith("_bucket"))
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{labels_prefix} {count} {int(latest.timestamp * 1000)}")
                lines.append(f"{name}_sum{labels_prefix} {sum_val} {int(latest.timestamp * 1000)}")

        return "\n".join(lines) + "\n"

    def clear(self) -> None:
        """Clear all stored points."""
        with self._lock:
            self._points.clear()


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector, creating if needed."""
    global _global_collector
    with _collector_lock:
        if _global_collector is None:
            _global_collector = MetricsCollector()
        return _global_collector


def set_global_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector."""
    global _global_collector
    with _collector_lock:
        _global_collector = collector


def record_metric(
    name: str,
    value: float,
    labels: Optional[dict[str, str]] = None,
    unit: str = "",
    description: str = "",
    metric_type: str = "gauge",
) -> MetricPoint:
    """Convenience function to record a metric using the global collector."""
    return get_metrics_collector().record(
        name=name,
        value=value,
        labels=labels,
        unit=unit,
        description=description,
        metric_type=metric_type,
    )


def record_histogram(
    name: str,
    values: list[float],
    labels: Optional[dict[str, str]] = None,
    unit: str = "",
    description: str = "",
) -> list[MetricPoint]:
    """Convenience function to record a histogram using the global collector."""
    return get_metrics_collector().record_histogram(
        name=name,
        values=values,
        labels=labels,
        unit=unit,
        description=description,
    )