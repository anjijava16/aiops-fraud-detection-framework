"""Unit tests for the Decision Audit Log implementation.

Tests cover:
- Append-only enforcement (reject update/delete)
- Entry logging with full metadata
- Query by transaction_id (including "not found")
- Query performance within 200ms
- Retry with exponential backoff on storage failure
- Retention policy configuration
- Entry validation

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6
"""

import time

import pytest

from src.explainability.decision_audit_log import DecisionAuditLog
from src.models.errors import ExplainabilityError
from src.models.explainability import AuditEntry


def _make_entry(
    transaction_id: str = "txn_001",
    model_version: str = "v1.0.0",
    fraud_score: float = 0.87,
    is_fraud: bool = True,
) -> AuditEntry:
    """Helper to create a valid AuditEntry."""
    return AuditEntry(
        transaction_id=transaction_id,
        timestamp=DecisionAuditLog.create_timestamp(),
        model_version=model_version,
        fraud_score=fraud_score,
        is_fraud=is_fraud,
        top_5_shap={
            "V14": -0.32,
            "V4": 0.28,
            "V12": -0.19,
            "V10": 0.15,
            "Amount": 0.11,
        },
    )


class TestAuditLogAppend:
    """Tests for the log() method — appending entries."""

    def test_log_stores_entry_successfully(self):
        log = DecisionAuditLog()
        entry = _make_entry()

        log.log(entry)

        assert log.entry_count == 1

    def test_log_records_all_required_fields(self):
        """Validates Requirement 16.1: records transaction_id, timestamp,
        model_version, fraud_score, is_fraud, top-5 SHAP."""
        log = DecisionAuditLog()
        entry = _make_entry(
            transaction_id="txn_100",
            model_version="v2.1.0",
            fraud_score=0.92,
            is_fraud=True,
        )

        log.log(entry)
        result = log.query("txn_100")

        assert result is not None
        assert result.transaction_id == "txn_100"
        assert result.model_version == "v2.1.0"
        assert result.fraud_score == 0.92
        assert result.is_fraud is True
        assert len(result.top_5_shap) == 5
        assert "T" in result.timestamp  # ISO 8601 format

    def test_log_multiple_entries(self):
        log = DecisionAuditLog()

        log.log(_make_entry(transaction_id="txn_001"))
        log.log(_make_entry(transaction_id="txn_002"))
        log.log(_make_entry(transaction_id="txn_003"))

        assert log.entry_count == 3

    def test_log_preserves_timestamp_with_ms_precision(self):
        """Validates Requirement 16.1: ISO 8601 UTC with ms precision."""
        log = DecisionAuditLog()
        timestamp = "2024-01-15T10:30:45.123Z"
        entry = AuditEntry(
            transaction_id="txn_ts",
            timestamp=timestamp,
            model_version="v1.0.0",
            fraud_score=0.5,
            is_fraud=False,
            top_5_shap={"V1": 0.1},
        )

        log.log(entry)
        result = log.query("txn_ts")

        assert result is not None
        assert result.timestamp == timestamp

    def test_log_validates_required_fields(self):
        log = DecisionAuditLog()

        with pytest.raises(ExplainabilityError, match="transaction_id"):
            log.log(AuditEntry(
                transaction_id="",
                timestamp="2024-01-01T00:00:00.000Z",
                model_version="v1",
                fraud_score=0.5,
                is_fraud=False,
                top_5_shap={"V1": 0.1},
            ))

    def test_log_validates_timestamp_required(self):
        log = DecisionAuditLog()

        with pytest.raises(ExplainabilityError, match="timestamp"):
            log.log(AuditEntry(
                transaction_id="txn_001",
                timestamp="",
                model_version="v1",
                fraud_score=0.5,
                is_fraud=False,
                top_5_shap={"V1": 0.1},
            ))

    def test_log_validates_model_version_required(self):
        log = DecisionAuditLog()

        with pytest.raises(ExplainabilityError, match="model_version"):
            log.log(AuditEntry(
                transaction_id="txn_001",
                timestamp="2024-01-01T00:00:00.000Z",
                model_version="",
                fraud_score=0.5,
                is_fraud=False,
                top_5_shap={"V1": 0.1},
            ))


class TestAuditLogAppendOnly:
    """Tests for append-only enforcement — Requirement 16.2."""

    def test_update_raises_operation_not_permitted(self):
        """Validates Requirement 16.2: reject update with error."""
        log = DecisionAuditLog()
        entry = _make_entry(transaction_id="txn_001")
        log.log(entry)

        updated_entry = _make_entry(transaction_id="txn_001", fraud_score=0.99)

        with pytest.raises(ExplainabilityError, match="operation not permitted"):
            log.update("txn_001", updated_entry)

    def test_delete_raises_operation_not_permitted(self):
        """Validates Requirement 16.2: reject delete with error."""
        log = DecisionAuditLog()
        entry = _make_entry(transaction_id="txn_001")
        log.log(entry)

        with pytest.raises(ExplainabilityError, match="operation not permitted"):
            log.delete("txn_001")

    def test_update_does_not_modify_existing_entry(self):
        log = DecisionAuditLog()
        entry = _make_entry(transaction_id="txn_001", fraud_score=0.87)
        log.log(entry)

        try:
            log.update("txn_001", _make_entry(fraud_score=0.99))
        except ExplainabilityError:
            pass

        result = log.query("txn_001")
        assert result is not None
        assert result.fraud_score == 0.87  # Original value unchanged

    def test_delete_does_not_remove_existing_entry(self):
        log = DecisionAuditLog()
        entry = _make_entry(transaction_id="txn_001")
        log.log(entry)

        try:
            log.delete("txn_001")
        except ExplainabilityError:
            pass

        result = log.query("txn_001")
        assert result is not None  # Entry still present

    def test_update_rejected_for_nonexistent_entry(self):
        """Update is rejected regardless of whether entry exists."""
        log = DecisionAuditLog()

        with pytest.raises(ExplainabilityError, match="operation not permitted"):
            log.update("txn_nonexistent", _make_entry())

    def test_delete_rejected_for_nonexistent_entry(self):
        """Delete is rejected regardless of whether entry exists."""
        log = DecisionAuditLog()

        with pytest.raises(ExplainabilityError, match="operation not permitted"):
            log.delete("txn_nonexistent")


class TestAuditLogQuery:
    """Tests for the query() method — Requirement 16.4, 16.5."""

    def test_query_returns_entry_by_transaction_id(self):
        """Validates Requirement 16.4: query by transaction_id."""
        log = DecisionAuditLog()
        entry = _make_entry(transaction_id="txn_lookup")
        log.log(entry)

        result = log.query("txn_lookup")

        assert result is not None
        assert result.transaction_id == "txn_lookup"

    def test_query_returns_none_for_missing_id(self):
        """Validates Requirement 16.5: not found for missing IDs."""
        log = DecisionAuditLog()

        result = log.query("txn_nonexistent")

        assert result is None

    def test_query_returns_exact_stored_data(self):
        """Round-trip integrity: query returns exact logged data."""
        log = DecisionAuditLog()
        shap_values = {"V14": -0.321, "V4": 0.289, "V12": -0.190, "V10": 0.152, "Amount": 0.111}
        entry = AuditEntry(
            transaction_id="txn_precise",
            timestamp="2024-06-15T14:30:45.789Z",
            model_version="v3.2.1",
            fraud_score=0.8765,
            is_fraud=True,
            top_5_shap=shap_values,
        )

        log.log(entry)
        result = log.query("txn_precise")

        assert result is not None
        assert result.transaction_id == "txn_precise"
        assert result.timestamp == "2024-06-15T14:30:45.789Z"
        assert result.model_version == "v3.2.1"
        assert result.fraud_score == 0.8765
        assert result.is_fraud is True
        assert result.top_5_shap == shap_values

    def test_query_performance_within_200ms(self):
        """Validates Requirement 16.4: query within 200ms."""
        log = DecisionAuditLog()
        # Populate with many entries
        for i in range(10000):
            log.log(_make_entry(transaction_id=f"txn_{i:05d}"))

        # Query in the middle
        start = time.perf_counter()
        result = log.query("txn_05000")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed_ms < 200.0

    def test_query_not_found_performance_within_200ms(self):
        """Validates Requirement 16.5: not found within 200ms."""
        log = DecisionAuditLog()
        for i in range(10000):
            log.log(_make_entry(transaction_id=f"txn_{i:05d}"))

        start = time.perf_counter()
        result = log.query("txn_nonexistent")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is None
        assert elapsed_ms < 200.0


class TestAuditLogRetention:
    """Tests for retention policy — Requirement 16.3."""

    def test_default_retention_is_7_years(self):
        log = DecisionAuditLog()
        assert log.retention_years == 7

    def test_configurable_retention_period(self):
        log = DecisionAuditLog(retention_years=10)
        assert log.retention_years == 10

    def test_retention_rejects_zero_years(self):
        with pytest.raises(ExplainabilityError, match="at least 1 year"):
            DecisionAuditLog(retention_years=0)

    def test_retention_rejects_negative_years(self):
        with pytest.raises(ExplainabilityError, match="at least 1 year"):
            DecisionAuditLog(retention_years=-1)

    def test_is_expired_returns_false_for_recent_entry(self):
        log = DecisionAuditLog(retention_years=7)
        entry = _make_entry()  # Created with current timestamp

        assert log.is_expired(entry) is False

    def test_is_expired_returns_true_for_old_entry(self):
        log = DecisionAuditLog(retention_years=7)
        # Create an entry with a timestamp older than 7 years
        old_entry = AuditEntry(
            transaction_id="txn_old",
            timestamp="2015-01-01T00:00:00.000Z",
            model_version="v0.1.0",
            fraud_score=0.3,
            is_fraud=False,
            top_5_shap={"V1": 0.1},
        )

        assert log.is_expired(old_entry) is True


class TestAuditLogRetry:
    """Tests for retry on storage failure — Requirement 16.6."""

    def test_retry_queues_entry_on_storage_failure(self):
        """Validates Requirement 16.6: queue entry on storage failure."""
        log = DecisionAuditLog()
        log.set_storage_available(False)
        entry = _make_entry(transaction_id="txn_retry")

        with pytest.raises(ExplainabilityError, match="unavailable after retries"):
            log.log(entry)

        assert log.retry_queue_size == 1
        assert log.entry_count == 0  # Not in primary store

    def test_retry_attempts_3_times(self):
        """Validates Requirement 16.6: 3 retries with exponential backoff."""
        log = DecisionAuditLog()
        log.set_storage_available(False)

        with pytest.raises(ExplainabilityError) as exc_info:
            log.log(_make_entry())

        assert exc_info.value.details["retries_attempted"] == 3

    def test_retry_succeeds_when_storage_recovers(self):
        """Entry stored if storage becomes available during retries."""
        log = DecisionAuditLog()
        # Start available, log some entries
        log.log(_make_entry(transaction_id="txn_before"))
        assert log.entry_count == 1

        # Storage goes down, then we queue
        log.set_storage_available(False)
        with pytest.raises(ExplainabilityError):
            log.log(_make_entry(transaction_id="txn_queued"))

        assert log.retry_queue_size == 1

        # Storage recovers — flush the queue
        log.set_storage_available(True)
        flushed = log.flush_retry_queue()

        assert flushed == 1
        assert log.retry_queue_size == 0
        assert log.query("txn_queued") is not None

    def test_flush_does_nothing_when_storage_unavailable(self):
        log = DecisionAuditLog()
        log.set_storage_available(False)

        with pytest.raises(ExplainabilityError):
            log.log(_make_entry(transaction_id="txn_q"))

        flushed = log.flush_retry_queue()
        assert flushed == 0
        assert log.retry_queue_size == 1


class TestAuditLogTimestamp:
    """Tests for timestamp generation."""

    def test_create_timestamp_is_iso_8601_utc(self):
        ts = DecisionAuditLog.create_timestamp()

        # Must contain 'T' separator and end with 'Z' (UTC)
        assert "T" in ts
        assert ts.endswith("Z")

    def test_create_timestamp_has_millisecond_precision(self):
        ts = DecisionAuditLog.create_timestamp()

        # Format: YYYY-MM-DDTHH:MM:SS.mmmZ
        # The milliseconds part is between '.' and 'Z'
        parts = ts.split(".")
        assert len(parts) == 2
        ms_part = parts[1].rstrip("Z")
        assert len(ms_part) == 3  # Exactly 3 digits for milliseconds
        assert ms_part.isdigit()

    def test_create_timestamp_is_monotonically_increasing(self):
        ts1 = DecisionAuditLog.create_timestamp()
        time.sleep(0.002)  # Small sleep to ensure different timestamps
        ts2 = DecisionAuditLog.create_timestamp()

        assert ts2 >= ts1
