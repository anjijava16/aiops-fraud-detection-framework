"""Prometheus metrics exporter for the fraud detection framework.

Exposes prediction counts, latency histograms, live F1 score,
drift event counters, and active model version as Prometheus metrics.
"""

from __future__ import annotations

import logging
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Exports Prometheus metrics for monitoring the fraud detection system.

    Metrics exposed:
        - prediction_count: Total number of predictions made (Counter).
        - prediction_latency_seconds: Prediction latency histogram (Histogram).
        - live_f1_score: Current live F1 score (Gauge).
        - drift_events_total: Total drift events detected (Counter).
        - active_model_version: Currently active model version info (Info).
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        """Initialize the MetricsExporter with Prometheus metric objects.

        Args:
            registry: Optional Prometheus CollectorRegistry. If None, uses a new registry
                      to avoid conflicts in testing.
        """
        self._registry = registry or CollectorRegistry()

        self._prediction_count = Counter(
            "prediction_count",
            "Total number of predictions made",
            registry=self._registry,
        )

        self._prediction_latency = Histogram(
            "prediction_latency_seconds",
            "Prediction latency in seconds",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=self._registry,
        )

        self._live_f1_score = Gauge(
            "live_f1_score",
            "Current live F1 score",
            registry=self._registry,
        )

        self._drift_events_total = Counter(
            "drift_events_total",
            "Total number of drift events detected",
            registry=self._registry,
        )

        self._active_model_version = Info(
            "active_model_version",
            "Currently active model version",
            registry=self._registry,
        )

    @property
    def registry(self) -> CollectorRegistry:
        """Return the Prometheus registry used by this exporter."""
        return self._registry

    def record_prediction(self, latency_seconds: float) -> None:
        """Record a prediction event with its latency.

        Args:
            latency_seconds: Time taken for the prediction in seconds.
        """
        self._prediction_count.inc()
        self._prediction_latency.observe(latency_seconds)

    def update_f1_score(self, f1_score: float) -> None:
        """Update the live F1 score gauge.

        Args:
            f1_score: Current F1 score value.
        """
        self._live_f1_score.set(f1_score)

    def record_drift_event(self) -> None:
        """Increment the drift events counter."""
        self._drift_events_total.inc()

    def set_model_version(self, version: str) -> None:
        """Set the active model version info.

        Args:
            version: The model version string.
        """
        self._active_model_version.info({"version": version})

    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics output for the /metrics endpoint.

        Returns:
            Bytes containing the Prometheus exposition format output.
        """
        return generate_latest(self._registry)

    def export(self) -> bytes:
        """Alias for get_metrics() matching the design interface.

        Returns:
            Bytes containing the Prometheus exposition format output.
        """
        return self.get_metrics()
