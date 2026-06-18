"""Unit tests for AlertPipeline."""

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.models.monitoring import AlertEvent
from src.monitoring.alert_pipeline import AlertPipeline, DeadLetterEntry, DeliveryStatus


@pytest.fixture
def alert_event():
    """Create a sample drift alert event."""
    return AlertEvent(
        alert_type="drift",
        drifted_features=["V1", "V2", "V3"],
        psi_values={"V1": 0.35, "V2": 0.28, "V3": 0.22},
        current_f1=0.92,
        timestamp="2024-01-15T10:00:00Z",
        dashboard_link="http://grafana:3000/dashboard",
    )


@pytest.fixture
def f1_drop_event():
    """Create a sample F1 drop alert event."""
    return AlertEvent(
        alert_type="f1_drop",
        drifted_features=["V1", "V2"],
        psi_values={"V1": 0.30, "V2": 0.25},
        current_f1=0.82,
        timestamp="2024-01-15T10:05:00Z",
        dashboard_link="http://grafana:3000/dashboard",
    )


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for testing."""
    client = MagicMock(spec=httpx.Client)
    response = MagicMock()
    response.status_code = 200
    client.post.return_value = response
    return client


@pytest.fixture
def pipeline(mock_http_client):
    """Create an AlertPipeline with mock HTTP client."""
    return AlertPipeline(
        slack_webhook_url="https://hooks.slack.com/test",
        pagerduty_routing_key="test-routing-key",
        retry_attempts=3,
        retry_base_interval_seconds=0.01,  # Fast for tests
        suppression_window_minutes=15,
        f1_alert_threshold=0.90,
        http_client=mock_http_client,
    )


class TestAlertPipelineInit:
    """Test AlertPipeline initialization."""

    def test_default_parameters(self):
        pipeline = AlertPipeline()
        assert pipeline.f1_alert_threshold == 0.90
        assert pipeline.suppression_count == 0
        assert pipeline.dead_letter_queue == []

    def test_custom_parameters(self):
        pipeline = AlertPipeline(
            slack_webhook_url="https://slack.example.com",
            pagerduty_routing_key="key-123",
            retry_attempts=5,
            f1_alert_threshold=0.85,
        )
        assert pipeline.f1_alert_threshold == 0.85


class TestSlackAlertDelivery:
    """Test Slack alert delivery."""

    def test_successful_slack_delivery(self, pipeline, mock_http_client, alert_event):
        results = pipeline.alert(alert_event)

        assert len(results) == 1
        assert results[0].delivered is True
        assert results[0].channel == "slack"
        assert results[0].attempts == 1
        mock_http_client.post.assert_called_once()

    def test_slack_delivery_with_retry(self, mock_http_client, alert_event):
        """Alert succeeds on second attempt."""
        response_fail = MagicMock()
        response_fail.status_code = 500
        response_ok = MagicMock()
        response_ok.status_code = 200
        mock_http_client.post.side_effect = [response_fail, response_ok]

        pipeline = AlertPipeline(
            slack_webhook_url="https://hooks.slack.com/test",
            retry_attempts=3,
            retry_base_interval_seconds=0.01,
            http_client=mock_http_client,
        )

        results = pipeline.alert(alert_event)
        assert len(results) == 1
        assert results[0].delivered is True
        assert results[0].attempts == 2

    def test_slack_delivery_exhausted_retries(self, mock_http_client, alert_event):
        """All retries fail — written to dead-letter queue."""
        response_fail = MagicMock()
        response_fail.status_code = 500
        mock_http_client.post.return_value = response_fail

        pipeline = AlertPipeline(
            slack_webhook_url="https://hooks.slack.com/test",
            retry_attempts=3,
            retry_base_interval_seconds=0.01,
            http_client=mock_http_client,
        )

        results = pipeline.alert(alert_event)
        assert len(results) == 1
        assert results[0].delivered is False
        assert results[0].attempts == 3
        assert results[0].error is not None

        # Verify dead-letter queue
        dlq = pipeline.dead_letter_queue
        assert len(dlq) == 1
        assert dlq[0].channel == "slack"
        assert dlq[0].attempts == 3

    def test_slack_delivery_http_error(self, mock_http_client, alert_event):
        """HTTP exception triggers retry and eventually dead-letter."""
        mock_http_client.post.side_effect = httpx.ConnectError("Connection refused")

        pipeline = AlertPipeline(
            slack_webhook_url="https://hooks.slack.com/test",
            retry_attempts=2,
            retry_base_interval_seconds=0.01,
            http_client=mock_http_client,
        )

        results = pipeline.alert(alert_event)
        assert len(results) == 1
        assert results[0].delivered is False
        assert len(pipeline.dead_letter_queue) == 1


class TestPagerDutyAlertDelivery:
    """Test PagerDuty alert delivery."""

    def test_pagerduty_triggered_on_f1_drop(self, mock_http_client, f1_drop_event):
        """PagerDuty is triggered when F1 drops below threshold."""
        response = MagicMock()
        response.status_code = 202
        mock_http_client.post.return_value = response

        pipeline = AlertPipeline(
            slack_webhook_url="https://hooks.slack.com/test",
            pagerduty_routing_key="test-key",
            f1_alert_threshold=0.90,
            retry_base_interval_seconds=0.01,
            http_client=mock_http_client,
        )

        results = pipeline.alert(f1_drop_event)

        # Should have PagerDuty + Slack results
        channels = [r.channel for r in results]
        assert "pagerduty" in channels
        assert "slack" in channels

    def test_pagerduty_not_triggered_above_threshold(self, mock_http_client):
        """PagerDuty NOT triggered when F1 is above threshold."""
        event = AlertEvent(
            alert_type="f1_drop",
            drifted_features=["V1"],
            current_f1=0.95,  # Above 0.90 threshold
            timestamp="2024-01-15T10:00:00Z",
        )

        pipeline = AlertPipeline(
            slack_webhook_url="https://hooks.slack.com/test",
            pagerduty_routing_key="test-key",
            f1_alert_threshold=0.90,
            retry_base_interval_seconds=0.01,
            http_client=mock_http_client,
        )

        results = pipeline.alert(event)
        channels = [r.channel for r in results]
        assert "pagerduty" not in channels


class TestDuplicateSuppression:
    """Test alert duplicate suppression."""

    def test_first_alert_delivered(self, pipeline, alert_event):
        results = pipeline.alert(alert_event)
        assert len(results) > 0
        assert pipeline.suppression_count == 0

    def test_duplicate_suppressed_within_window(self, pipeline, alert_event):
        """Second alert for same features within 15 min is suppressed."""
        pipeline.alert(alert_event)

        # Send same alert again
        results = pipeline.alert(alert_event)
        assert results == []  # Suppressed
        assert pipeline.suppression_count == 1

    def test_different_features_not_suppressed(self, pipeline, mock_http_client, alert_event):
        """Alerts for different features are NOT suppressed."""
        pipeline.alert(alert_event)

        different_event = AlertEvent(
            alert_type="drift",
            drifted_features=["V4", "V5"],
            psi_values={"V4": 0.30, "V5": 0.25},
            current_f1=0.91,
            timestamp="2024-01-15T10:01:00Z",
        )

        results = pipeline.alert(different_event)
        assert len(results) > 0
        assert pipeline.suppression_count == 0

    def test_suppression_counter_increments(self, pipeline, alert_event):
        """Suppression counter increments for each suppressed alert."""
        pipeline.alert(alert_event)

        pipeline.alert(alert_event)
        pipeline.alert(alert_event)
        pipeline.alert(alert_event)

        assert pipeline.suppression_count == 3


class TestNoWebhookConfigured:
    """Test behavior when no webhook URLs are configured."""

    def test_no_slack_url_no_delivery(self, alert_event):
        pipeline = AlertPipeline(slack_webhook_url="", pagerduty_routing_key="")
        results = pipeline.alert(alert_event)
        assert results == []

    def test_no_pagerduty_key_no_delivery(self, f1_drop_event):
        pipeline = AlertPipeline(
            slack_webhook_url="", pagerduty_routing_key=""
        )
        results = pipeline.alert(f1_drop_event)
        assert results == []
