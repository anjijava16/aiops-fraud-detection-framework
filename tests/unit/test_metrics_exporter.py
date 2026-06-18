"""Unit tests for MetricsExporter and DashboardConfig."""

import json

import pytest
from prometheus_client import CollectorRegistry

from src.monitoring.dashboard_config import DashboardConfig
from src.monitoring.metrics_exporter import MetricsExporter


@pytest.fixture
def registry():
    """Create a fresh Prometheus registry to avoid conflicts between tests."""
    return CollectorRegistry()


@pytest.fixture
def exporter(registry):
    """Create a MetricsExporter with isolated registry."""
    return MetricsExporter(registry=registry)


class TestMetricsExporterInit:
    """Test MetricsExporter initialization."""

    def test_creates_with_registry(self, registry):
        exporter = MetricsExporter(registry=registry)
        assert exporter.registry is registry

    def test_creates_with_default_registry(self):
        exporter = MetricsExporter()
        assert exporter.registry is not None


class TestPredictionMetrics:
    """Test prediction count and latency metrics."""

    def test_record_prediction_increments_count(self, exporter):
        exporter.record_prediction(0.05)
        exporter.record_prediction(0.08)

        metrics_output = exporter.get_metrics().decode("utf-8")
        assert "prediction_count_total 2.0" in metrics_output

    def test_record_prediction_latency(self, exporter):
        exporter.record_prediction(0.05)
        exporter.record_prediction(0.15)

        metrics_output = exporter.get_metrics().decode("utf-8")
        assert "prediction_latency_seconds" in metrics_output
        # Check that the sum reflects both observations
        assert "prediction_latency_seconds_sum 0.2" in metrics_output

    def test_latency_histogram_buckets(self, exporter):
        exporter.record_prediction(0.005)  # Below 0.01 bucket
        exporter.record_prediction(0.03)  # Between 0.025 and 0.05 buckets

        metrics_output = exporter.get_metrics().decode("utf-8")
        # Verify histogram bucket labels exist
        assert 'prediction_latency_seconds_bucket{le="0.01"}' in metrics_output
        assert 'prediction_latency_seconds_bucket{le="0.1"}' in metrics_output


class TestF1ScoreMetric:
    """Test live F1 score gauge."""

    def test_update_f1_score(self, exporter):
        exporter.update_f1_score(0.92)

        metrics_output = exporter.get_metrics().decode("utf-8")
        assert "live_f1_score 0.92" in metrics_output

    def test_f1_score_updates(self, exporter):
        exporter.update_f1_score(0.92)
        exporter.update_f1_score(0.88)

        metrics_output = exporter.get_metrics().decode("utf-8")
        assert "live_f1_score 0.88" in metrics_output


class TestDriftEventsMetric:
    """Test drift events counter."""

    def test_record_drift_event(self, exporter):
        exporter.record_drift_event()
        exporter.record_drift_event()
        exporter.record_drift_event()

        metrics_output = exporter.get_metrics().decode("utf-8")
        assert "drift_events_total 3.0" in metrics_output


class TestModelVersionMetric:
    """Test active model version info."""

    def test_set_model_version(self, exporter):
        exporter.set_model_version("v1.2.3")

        metrics_output = exporter.get_metrics().decode("utf-8")
        assert "active_model_version_info" in metrics_output
        assert "v1.2.3" in metrics_output

    def test_model_version_update(self, exporter):
        exporter.set_model_version("v1.0.0")
        exporter.set_model_version("v2.0.0")

        metrics_output = exporter.get_metrics().decode("utf-8")
        assert "v2.0.0" in metrics_output


class TestExportAlias:
    """Test export() alias method."""

    def test_export_returns_same_as_get_metrics(self, exporter):
        exporter.record_prediction(0.1)
        assert exporter.export() == exporter.get_metrics()


class TestDashboardConfig:
    """Test Grafana dashboard JSON generation."""

    def test_generate_returns_dict(self):
        config = DashboardConfig()
        dashboard = config.generate()

        assert isinstance(dashboard, dict)
        assert "dashboard" in dashboard
        assert "overwrite" in dashboard

    def test_dashboard_title(self):
        config = DashboardConfig(title="Custom Dashboard")
        dashboard = config.generate()
        assert dashboard["dashboard"]["title"] == "Custom Dashboard"

    def test_dashboard_has_5_panels(self):
        config = DashboardConfig()
        dashboard = config.generate()
        panels = dashboard["dashboard"]["panels"]
        assert len(panels) == 5

    def test_panel_types(self):
        config = DashboardConfig()
        dashboard = config.generate()
        panels = dashboard["dashboard"]["panels"]

        panel_types = [p["type"] for p in panels]
        assert "timeseries" in panel_types
        assert "gauge" in panel_types
        assert "stat" in panel_types

    def test_panel_titles(self):
        config = DashboardConfig()
        dashboard = config.generate()
        panels = dashboard["dashboard"]["panels"]

        titles = [p["title"] for p in panels]
        assert "Prediction Rate" in titles
        assert "Prediction Latency" in titles
        assert "Live F1 Score" in titles
        assert "Drift Events" in titles
        assert "Active Model Version" in titles

    def test_generate_json(self):
        config = DashboardConfig()
        json_str = config.generate_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "dashboard" in parsed

    def test_dashboard_tags(self):
        config = DashboardConfig()
        dashboard = config.generate()
        tags = dashboard["dashboard"]["tags"]
        assert "fraud-detection" in tags
        assert "aiops" in tags
        assert "monitoring" in tags

    def test_dashboard_uid(self):
        config = DashboardConfig()
        dashboard = config.generate()
        assert dashboard["dashboard"]["uid"] == "aiops-fraud-detection"

    def test_custom_datasource(self):
        config = DashboardConfig(datasource="CustomProm")
        dashboard = config.generate()
        panels = dashboard["dashboard"]["panels"]

        for panel in panels:
            assert panel["datasource"]["uid"] == "CustomProm"

    def test_latency_panel_has_percentile_targets(self):
        config = DashboardConfig()
        dashboard = config.generate()
        latency_panel = next(
            p for p in dashboard["dashboard"]["panels"] if p["title"] == "Prediction Latency"
        )
        legends = [t["legendFormat"] for t in latency_panel["targets"]]
        assert "p50" in legends
        assert "p95" in legends
        assert "p99" in legends
