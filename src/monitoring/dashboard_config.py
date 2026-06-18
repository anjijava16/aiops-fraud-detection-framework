"""Grafana dashboard JSON configuration generator.

Generates provisioning-ready dashboard JSON for monitoring the fraud
detection system, including panels for predictions, latency, F1 score,
drift events, and model version.
"""

from __future__ import annotations

import json
from typing import Any


class DashboardConfig:
    """Generates Grafana dashboard JSON for the fraud detection monitoring system.

    The generated dashboard includes panels for:
        - Prediction count rate
        - Prediction latency (p50, p95, p99)
        - Live F1 score
        - Drift events over time
        - Active model version
    """

    def __init__(
        self,
        title: str = "AIOps Fraud Detection Monitoring",
        datasource: str = "Prometheus",
        refresh_interval: str = "15s",
    ) -> None:
        """Initialize the DashboardConfig.

        Args:
            title: Dashboard title.
            datasource: Prometheus datasource name.
            refresh_interval: Auto-refresh interval.
        """
        self._title = title
        self._datasource = datasource
        self._refresh_interval = refresh_interval

    def generate(self) -> dict[str, Any]:
        """Generate the complete Grafana dashboard JSON structure.

        Returns:
            Dict representing the Grafana dashboard configuration.
        """
        return {
            "dashboard": {
                "id": None,
                "uid": "aiops-fraud-detection",
                "title": self._title,
                "tags": ["fraud-detection", "aiops", "monitoring"],
                "timezone": "utc",
                "schemaVersion": 38,
                "version": 1,
                "refresh": self._refresh_interval,
                "time": {"from": "now-1h", "to": "now"},
                "panels": self._build_panels(),
                "templating": {"list": []},
                "annotations": {"list": []},
            },
            "overwrite": True,
        }

    def generate_json(self, indent: int = 2) -> str:
        """Generate the dashboard as a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string of the dashboard configuration.
        """
        return json.dumps(self.generate(), indent=indent)

    def _build_panels(self) -> list[dict[str, Any]]:
        """Build all dashboard panels.

        Returns:
            List of Grafana panel configurations.
        """
        panels = [
            self._prediction_count_panel(),
            self._prediction_latency_panel(),
            self._f1_score_panel(),
            self._drift_events_panel(),
            self._model_version_panel(),
        ]
        return panels

    def _prediction_count_panel(self) -> dict[str, Any]:
        """Build the prediction count rate panel."""
        return {
            "id": 1,
            "title": "Prediction Rate",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "datasource": {"type": "prometheus", "uid": self._datasource},
            "targets": [
                {
                    "expr": "rate(prediction_count_total[5m])",
                    "legendFormat": "Predictions/sec",
                    "refId": "A",
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "reqps",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 50},
                            {"color": "red", "value": 100},
                        ],
                    },
                }
            },
        }

    def _prediction_latency_panel(self) -> dict[str, Any]:
        """Build the prediction latency percentiles panel."""
        return {
            "id": 2,
            "title": "Prediction Latency",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "datasource": {"type": "prometheus", "uid": self._datasource},
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(prediction_latency_seconds_bucket[5m]))",
                    "legendFormat": "p50",
                    "refId": "A",
                },
                {
                    "expr": "histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))",
                    "legendFormat": "p95",
                    "refId": "B",
                },
                {
                    "expr": "histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m]))",
                    "legendFormat": "p99",
                    "refId": "C",
                },
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 0.1},
                            {"color": "red", "value": 0.2},
                        ],
                    },
                }
            },
        }

    def _f1_score_panel(self) -> dict[str, Any]:
        """Build the live F1 score panel."""
        return {
            "id": 3,
            "title": "Live F1 Score",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8},
            "datasource": {"type": "prometheus", "uid": self._datasource},
            "targets": [
                {
                    "expr": "live_f1_score",
                    "legendFormat": "F1 Score",
                    "refId": "A",
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 0.85},
                            {"color": "green", "value": 0.90},
                        ],
                    },
                }
            },
        }

    def _drift_events_panel(self) -> dict[str, Any]:
        """Build the drift events panel."""
        return {
            "id": 4,
            "title": "Drift Events",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8},
            "datasource": {"type": "prometheus", "uid": self._datasource},
            "targets": [
                {
                    "expr": "rate(drift_events_total_total[5m])",
                    "legendFormat": "Drift Events/5m",
                    "refId": "A",
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 1},
                            {"color": "red", "value": 5},
                        ],
                    },
                }
            },
        }

    def _model_version_panel(self) -> dict[str, Any]:
        """Build the active model version panel."""
        return {
            "id": 5,
            "title": "Active Model Version",
            "type": "stat",
            "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8},
            "datasource": {"type": "prometheus", "uid": self._datasource},
            "targets": [
                {
                    "expr": "active_model_version_info",
                    "legendFormat": "{{version}}",
                    "refId": "A",
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "blue", "value": None}],
                    },
                }
            },
        }
