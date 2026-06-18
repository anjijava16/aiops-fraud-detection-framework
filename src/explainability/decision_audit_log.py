"""Decision Audit Log — append-only audit trail for regulatory traceability.

Records every fraud prediction with its explanation for compliance with
financial record retention requirements. Enforces immutability (no updates
or deletes) and supports configurable retention policies.

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.models.errors import ExplainabilityError
from src.models.explainability import AuditEntry

logger = logging.getLogger(__name__)

# Default retention period: 7 years in days
_DEFAULT_RETENTION_YEARS = 7


class DecisionAuditLog:
    """Append-only decision audit log for regulatory traceability.

    Records transaction predictions with SHAP explanations. Enforces:
    - Append-only semantics (no update/delete)
    - 7-year minimum retention (configurable)
    - Query by transaction_id within 200ms
    - Retry with exponential backoff on storage failure

    Uses in-memory dict for storage (production would use a database).
    """

    def __init__(self, retention_years: int = _DEFAULT_RETENTION_YEARS) -> None:
        """Initialize the audit log.

        Args:
            retention_years: Minimum number of years to retain entries.
                             Defaults to 7 per financial regulations.
        """
        if retention_years < 1:
            raise ExplainabilityError(
                "Retention period must be at least 1 year",
                details={"retention_years": retention_years},
            )
        self._retention_years = retention_years
        # Primary storage: transaction_id -> AuditEntry
        self._store: dict[str, AuditEntry] = {}
        # Retry queue for entries that failed to persist
        self._retry_queue: list[AuditEntry] = []
        # Flag to simulate storage failure for testing
        self._storage_available: bool = True
        # Max retry attempts
        self._max_retries: int = 3

    @property
    def retention_years(self) -> int:
        """Return the configured retention period in years."""
        return self._retention_years

    @property
    def entry_count(self) -> int:
        """Return the number of entries in the audit log."""
        return len(self._store)

    @property
    def retry_queue_size(self) -> int:
        """Return the number of entries waiting for retry."""
        return len(self._retry_queue)

    def log(self, entry: AuditEntry) -> None:
        """Append a new entry to the audit log.

        Records the prediction decision with full traceability metadata.
        If storage is unavailable, queues the entry for retry with
        exponential backoff (up to 3 attempts).

        Args:
            entry: The audit entry to record.

        Raises:
            ExplainabilityError: If all retry attempts are exhausted.
        """
        self._validate_entry(entry)

        if self._storage_available:
            self._store[entry.transaction_id] = entry
            logger.info(
                "Audit entry logged for transaction %s", entry.transaction_id
            )
        else:
            # Storage unavailable — attempt retries with exponential backoff
            self._retry_with_backoff(entry)

    def query(self, transaction_id: str) -> Optional[AuditEntry]:
        """Query the audit log by transaction ID.

        Must return within 200ms. Returns None with "not found" semantics
        for missing IDs.

        Args:
            transaction_id: The transaction ID to look up.

        Returns:
            The AuditEntry if found, None otherwise.
        """
        return self._store.get(transaction_id)

    def update(self, transaction_id: str, entry: AuditEntry) -> None:
        """Reject update operations — audit log is append-only.

        Args:
            transaction_id: The transaction ID to update.
            entry: The updated entry.

        Raises:
            ExplainabilityError: Always — updates are not permitted.
        """
        raise ExplainabilityError(
            "operation not permitted",
            details={
                "operation": "update",
                "transaction_id": transaction_id,
                "reason": "audit log is append-only",
            },
        )

    def delete(self, transaction_id: str) -> None:
        """Reject delete operations — audit log is append-only.

        Args:
            transaction_id: The transaction ID to delete.

        Raises:
            ExplainabilityError: Always — deletes are not permitted.
        """
        raise ExplainabilityError(
            "operation not permitted",
            details={
                "operation": "delete",
                "transaction_id": transaction_id,
                "reason": "audit log is append-only",
            },
        )

    def set_storage_available(self, available: bool) -> None:
        """Set storage availability (for testing/simulation).

        Args:
            available: Whether storage is available.
        """
        self._storage_available = available

    def flush_retry_queue(self) -> int:
        """Attempt to flush queued entries to storage.

        Returns:
            Number of entries successfully flushed.
        """
        if not self._storage_available:
            return 0

        flushed = 0
        remaining: list[AuditEntry] = []

        for entry in self._retry_queue:
            self._store[entry.transaction_id] = entry
            flushed += 1
            logger.info(
                "Flushed queued audit entry for transaction %s",
                entry.transaction_id,
            )

        self._retry_queue = remaining
        return flushed

    def is_expired(self, entry: AuditEntry) -> bool:
        """Check if an entry has exceeded its retention period.

        Args:
            entry: The audit entry to check.

        Returns:
            True if the entry is older than the retention period.
        """
        try:
            entry_time = datetime.fromisoformat(entry.timestamp)
            now = datetime.now(timezone.utc)
            age_days = (now - entry_time).days
            retention_days = self._retention_years * 365
            return age_days > retention_days
        except (ValueError, TypeError):
            # If timestamp can't be parsed, don't expire it
            return False

    def _validate_entry(self, entry: AuditEntry) -> None:
        """Validate an audit entry before logging.

        Args:
            entry: The entry to validate.

        Raises:
            ExplainabilityError: If the entry is invalid.
        """
        if not entry.transaction_id:
            raise ExplainabilityError(
                "transaction_id is required",
                details={"field": "transaction_id"},
            )
        if not entry.timestamp:
            raise ExplainabilityError(
                "timestamp is required",
                details={"field": "timestamp"},
            )
        if not entry.model_version:
            raise ExplainabilityError(
                "model_version is required",
                details={"field": "model_version"},
            )
        if not isinstance(entry.fraud_score, (int, float)):
            raise ExplainabilityError(
                "fraud_score must be numeric",
                details={"field": "fraud_score", "value": entry.fraud_score},
            )
        if not isinstance(entry.is_fraud, bool):
            raise ExplainabilityError(
                "is_fraud must be boolean",
                details={"field": "is_fraud", "value": entry.is_fraud},
            )
        if not isinstance(entry.top_5_shap, dict):
            raise ExplainabilityError(
                "top_5_shap must be a dictionary",
                details={"field": "top_5_shap"},
            )

    def _retry_with_backoff(self, entry: AuditEntry) -> None:
        """Attempt to store an entry with exponential backoff retries.

        Tries up to 3 times with delays of 0.01s, 0.02s, 0.04s (scaled
        down for in-memory implementation; production would use 1s, 2s, 4s).

        If all retries fail, queues the entry and raises an alert.

        Args:
            entry: The entry to store.

        Raises:
            ExplainabilityError: If all retry attempts are exhausted.
        """
        base_delay = 0.01  # 10ms base for in-memory (production: 1s)

        for attempt in range(1, self._max_retries + 1):
            delay = base_delay * (2 ** (attempt - 1))
            time.sleep(delay)

            if self._storage_available:
                self._store[entry.transaction_id] = entry
                logger.info(
                    "Audit entry logged after %d retry(ies) for transaction %s",
                    attempt,
                    entry.transaction_id,
                )
                return

            logger.warning(
                "Retry %d/%d failed for transaction %s",
                attempt,
                self._max_retries,
                entry.transaction_id,
            )

        # All retries exhausted — queue the entry
        self._retry_queue.append(entry)
        logger.error(
            "All %d retries exhausted for transaction %s. Entry queued.",
            self._max_retries,
            entry.transaction_id,
        )
        raise ExplainabilityError(
            "Audit log storage unavailable after retries",
            details={
                "transaction_id": entry.transaction_id,
                "retries_attempted": self._max_retries,
                "action": "queued_for_later",
            },
        )

    @staticmethod
    def create_timestamp() -> str:
        """Generate an ISO 8601 UTC timestamp with millisecond precision.

        Returns:
            ISO 8601 formatted timestamp string.
        """
        now = datetime.now(timezone.utc)
        # Format with millisecond precision: YYYY-MM-DDTHH:MM:SS.mmmZ
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
