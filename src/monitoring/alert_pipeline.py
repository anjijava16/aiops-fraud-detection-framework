"""Alert pipeline for delivering drift and performance alerts via Slack and PagerDuty.

Implements retry logic with exponential backoff, duplicate suppression,
and dead-letter queue for undelivered alerts.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from src.models.monitoring import AlertEvent

logger = logging.getLogger(__name__)


@dataclass
class DeliveryStatus:
    """Status of an alert delivery attempt.

    Attributes:
        delivered: True if alert was successfully delivered.
        channel: The delivery channel ("slack", "pagerduty").
        attempts: Number of delivery attempts made.
        error: Error message if delivery failed.
    """

    delivered: bool
    channel: str
    attempts: int = 0
    error: Optional[str] = None


@dataclass
class DeadLetterEntry:
    """An undelivered alert stored in the dead-letter queue.

    Attributes:
        alert_event: The original alert event.
        channel: The channel that failed delivery.
        error: The final error message.
        timestamp: ISO 8601 UTC timestamp when queued.
        attempts: Number of delivery attempts made.
    """

    alert_event: AlertEvent
    channel: str
    error: str
    timestamp: str
    attempts: int


class AlertPipeline:
    """Manages alert delivery to Slack and PagerDuty with retry and deduplication.

    Features:
        - Sends Slack alerts within 30s of drift event
        - Triggers PagerDuty incident when live F1 drops below threshold
        - Retries delivery 3x with exponential backoff
        - Writes undelivered alerts to dead-letter queue
        - Suppresses duplicate alerts for same features within 15-minute window
    """

    def __init__(
        self,
        slack_webhook_url: str = "",
        pagerduty_routing_key: str = "",
        retry_attempts: int = 3,
        retry_base_interval_seconds: float = 2.0,
        suppression_window_minutes: float = 15.0,
        f1_alert_threshold: float = 0.90,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize the AlertPipeline.

        Args:
            slack_webhook_url: Slack incoming webhook URL.
            pagerduty_routing_key: PagerDuty Events API v2 routing key.
            retry_attempts: Max delivery attempts (default 3).
            retry_base_interval_seconds: Base interval for exponential backoff (default 2).
            suppression_window_minutes: Window for deduplicating alerts (default 15).
            f1_alert_threshold: F1 score below which PagerDuty is triggered (default 0.90).
            http_client: Optional httpx.Client for dependency injection in tests.
        """
        self._slack_webhook_url = slack_webhook_url
        self._pagerduty_routing_key = pagerduty_routing_key
        self._retry_attempts = retry_attempts
        self._retry_base_interval_seconds = retry_base_interval_seconds
        self._suppression_window_minutes = suppression_window_minutes
        self._f1_alert_threshold = f1_alert_threshold
        self._http_client = http_client

        # Dead-letter queue for undelivered alerts
        self._dead_letter_queue: deque[DeadLetterEntry] = deque()

        # Suppression state: maps frozenset(features) -> last alert timestamp
        self._suppression_state: dict[frozenset[str], float] = {}

        # Suppression counter
        self._suppression_count: int = 0

    @property
    def dead_letter_queue(self) -> list[DeadLetterEntry]:
        """Return a copy of the dead-letter queue contents."""
        return list(self._dead_letter_queue)

    @property
    def suppression_count(self) -> int:
        """Return the number of suppressed duplicate alerts."""
        return self._suppression_count

    @property
    def f1_alert_threshold(self) -> float:
        """Return the F1 alert threshold."""
        return self._f1_alert_threshold

    def alert(self, event: AlertEvent) -> list[DeliveryStatus]:
        """Process an alert event and deliver to appropriate channels.

        Args:
            event: The alert event to deliver.

        Returns:
            List of DeliveryStatus results for each channel attempted.
        """
        # Check duplicate suppression
        if self._is_suppressed(event):
            self._suppression_count += 1
            logger.info(
                "Alert suppressed (duplicate within %d-minute window): %s",
                self._suppression_window_minutes,
                event.drifted_features,
            )
            return []

        # Record this alert for future suppression
        self._record_alert(event)

        results: list[DeliveryStatus] = []

        # Send Slack alert for drift events
        if event.alert_type in ("drift", "latency") and self._slack_webhook_url:
            status = self._send_slack_alert(event)
            results.append(status)

        # Trigger PagerDuty for F1 drops below threshold
        if (
            event.alert_type == "f1_drop"
            and event.current_f1 < self._f1_alert_threshold
            and self._pagerduty_routing_key
        ):
            status = self._send_pagerduty_alert(event)
            results.append(status)

        # Also send Slack for f1_drop
        if event.alert_type == "f1_drop" and self._slack_webhook_url:
            status = self._send_slack_alert(event)
            results.append(status)

        return results

    def _is_suppressed(self, event: AlertEvent) -> bool:
        """Check if an alert should be suppressed (duplicate within window).

        Args:
            event: The alert event to check.

        Returns:
            True if the alert should be suppressed.
        """
        feature_key = frozenset(event.drifted_features)
        if feature_key not in self._suppression_state:
            return False

        last_alert_time = self._suppression_state[feature_key]
        current_time = time.time()
        window_seconds = self._suppression_window_minutes * 60

        return (current_time - last_alert_time) < window_seconds

    def _record_alert(self, event: AlertEvent) -> None:
        """Record an alert timestamp for suppression tracking."""
        feature_key = frozenset(event.drifted_features)
        self._suppression_state[feature_key] = time.time()

    def _send_slack_alert(self, event: AlertEvent) -> DeliveryStatus:
        """Send an alert to Slack with retry logic.

        Args:
            event: The alert event to send.

        Returns:
            DeliveryStatus indicating success or failure.
        """
        payload = self._build_slack_payload(event)

        for attempt in range(1, self._retry_attempts + 1):
            try:
                client = self._http_client or httpx.Client(timeout=30.0)
                response = client.post(
                    self._slack_webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    logger.info("Slack alert delivered successfully (attempt %d)", attempt)
                    return DeliveryStatus(
                        delivered=True, channel="slack", attempts=attempt
                    )

                logger.warning(
                    "Slack delivery failed (attempt %d/%d): HTTP %d",
                    attempt,
                    self._retry_attempts,
                    response.status_code,
                )

            except (httpx.HTTPError, Exception) as e:
                logger.warning(
                    "Slack delivery error (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    str(e),
                )

            # Exponential backoff before retry
            if attempt < self._retry_attempts:
                backoff = self._retry_base_interval_seconds * (2 ** (attempt - 1))
                time.sleep(backoff)

        # All retries exhausted — write to dead-letter queue
        error_msg = f"Slack delivery failed after {self._retry_attempts} attempts"
        self._dead_letter_queue.append(
            DeadLetterEntry(
                alert_event=event,
                channel="slack",
                error=error_msg,
                timestamp=datetime.now(timezone.utc).isoformat(),
                attempts=self._retry_attempts,
            )
        )

        return DeliveryStatus(
            delivered=False,
            channel="slack",
            attempts=self._retry_attempts,
            error=error_msg,
        )

    def _send_pagerduty_alert(self, event: AlertEvent) -> DeliveryStatus:
        """Send a PagerDuty incident trigger with retry logic.

        Args:
            event: The alert event to send.

        Returns:
            DeliveryStatus indicating success or failure.
        """
        payload = self._build_pagerduty_payload(event)

        for attempt in range(1, self._retry_attempts + 1):
            try:
                client = self._http_client or httpx.Client(timeout=30.0)
                response = client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in (200, 202):
                    logger.info(
                        "PagerDuty incident triggered successfully (attempt %d)",
                        attempt,
                    )
                    return DeliveryStatus(
                        delivered=True, channel="pagerduty", attempts=attempt
                    )

                logger.warning(
                    "PagerDuty delivery failed (attempt %d/%d): HTTP %d",
                    attempt,
                    self._retry_attempts,
                    response.status_code,
                )

            except (httpx.HTTPError, Exception) as e:
                logger.warning(
                    "PagerDuty delivery error (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    str(e),
                )

            # Exponential backoff before retry
            if attempt < self._retry_attempts:
                backoff = self._retry_base_interval_seconds * (2 ** (attempt - 1))
                time.sleep(backoff)

        # All retries exhausted — write to dead-letter queue
        error_msg = f"PagerDuty delivery failed after {self._retry_attempts} attempts"
        self._dead_letter_queue.append(
            DeadLetterEntry(
                alert_event=event,
                channel="pagerduty",
                error=error_msg,
                timestamp=datetime.now(timezone.utc).isoformat(),
                attempts=self._retry_attempts,
            )
        )

        return DeliveryStatus(
            delivered=False,
            channel="pagerduty",
            attempts=self._retry_attempts,
            error=error_msg,
        )

    def _build_slack_payload(self, event: AlertEvent) -> dict[str, Any]:
        """Build a Slack webhook payload from an AlertEvent.

        Args:
            event: The alert event.

        Returns:
            Dict formatted for Slack incoming webhook.
        """
        if event.alert_type == "drift":
            text = (
                f"🚨 *Drift Alert*\n"
                f"Features drifted: {', '.join(event.drifted_features)}\n"
                f"PSI values: {json.dumps(event.psi_values, indent=2)}\n"
                f"Current F1: {event.current_f1:.4f}\n"
                f"Timestamp: {event.timestamp}\n"
                f"Dashboard: {event.dashboard_link}"
            )
        elif event.alert_type == "f1_drop":
            text = (
                f"🔴 *F1 Score Drop Alert*\n"
                f"Current F1: {event.current_f1:.4f} "
                f"(threshold: {self._f1_alert_threshold:.4f})\n"
                f"Timestamp: {event.timestamp}\n"
                f"Dashboard: {event.dashboard_link}"
            )
        else:
            text = (
                f"⚠️ *{event.alert_type.upper()} Alert*\n"
                f"Timestamp: {event.timestamp}\n"
                f"Dashboard: {event.dashboard_link}"
            )

        return {"text": text}

    def _build_pagerduty_payload(self, event: AlertEvent) -> dict[str, Any]:
        """Build a PagerDuty Events API v2 payload from an AlertEvent.

        Args:
            event: The alert event.

        Returns:
            Dict formatted for PagerDuty Events API v2.
        """
        return {
            "routing_key": self._pagerduty_routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": (
                    f"Fraud Detection F1 Score Drop - "
                    f"Current: {event.current_f1:.4f}, "
                    f"Threshold: {self._f1_alert_threshold:.4f}"
                ),
                "severity": "critical",
                "source": "aiops-fraud-detection",
                "component": "monitoring",
                "custom_details": {
                    "current_f1": event.current_f1,
                    "threshold": self._f1_alert_threshold,
                    "drifted_features": event.drifted_features,
                    "psi_values": event.psi_values,
                    "timestamp": event.timestamp,
                    "dashboard_link": event.dashboard_link,
                },
            },
        }
