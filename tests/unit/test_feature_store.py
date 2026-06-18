"""Unit tests for the Feature Store implementation.

Tests cover:
- Single record put/get round-trip integrity
- Batch ingestion with metadata tracking
- Version-based retrieval for reproducibility
- Validation of feature structure
- Performance characteristics (in-memory lookup)
- Metadata correctness (data source, timestamp, record count)
"""

import time

import pytest

from src.ingestion.feature_store import FeatureBatch, FeatureRecord, FeatureStore


def _make_features(time_val: float = 0.0, amount: float = 149.62) -> dict:
    """Helper to create a valid features dictionary."""
    return {
        "time": time_val,
        "v1_to_v28": [float(i) for i in range(1, 29)],
        "amount": amount,
    }


def _make_batch_record(
    transaction_id: str, time_val: float = 0.0, amount: float = 100.0
) -> dict:
    """Helper to create a valid batch record dictionary."""
    return {
        "transaction_id": transaction_id,
        "time": time_val,
        "v1_to_v28": [float(i) * 0.1 for i in range(1, 29)],
        "amount": amount,
    }


class TestFeatureStorePut:
    """Tests for the put() method — storing single records."""

    def test_put_stores_record_successfully(self):
        store = FeatureStore()
        features = _make_features(time_val=1.5, amount=200.0)

        record = store.put("txn_001", features)

        assert record.transaction_id == "txn_001"
        assert record.time == 1.5
        assert record.amount == 200.0
        assert record.v1_to_v28 == [float(i) for i in range(1, 29)]
        assert record.ingestion_timestamp is not None

    def test_put_increments_record_count(self):
        store = FeatureStore()
        assert store.record_count == 0

        store.put("txn_001", _make_features())
        assert store.record_count == 1

        store.put("txn_002", _make_features())
        assert store.record_count == 2

    def test_put_overwrites_existing_transaction_id(self):
        store = FeatureStore()
        store.put("txn_001", _make_features(amount=100.0))
        store.put("txn_001", _make_features(amount=999.0))

        record = store.get("txn_001")
        assert record is not None
        assert record.amount == 999.0
        assert store.record_count == 1

    def test_put_creates_independent_copy_of_features(self):
        store = FeatureStore()
        features = _make_features()
        original_v_values = list(features["v1_to_v28"])

        store.put("txn_001", features)
        # Mutate the original list
        features["v1_to_v28"][0] = -999.0

        record = store.get("txn_001")
        assert record is not None
        assert record.v1_to_v28 == original_v_values


class TestFeatureStoreGet:
    """Tests for the get() method — retrieving single records."""

    def test_get_returns_stored_record(self):
        store = FeatureStore()
        store.put("txn_001", _make_features(time_val=5.0, amount=50.0))

        record = store.get("txn_001")

        assert record is not None
        assert record.transaction_id == "txn_001"
        assert record.time == 5.0
        assert record.amount == 50.0

    def test_get_returns_none_for_missing_id(self):
        store = FeatureStore()
        assert store.get("nonexistent") is None

    def test_get_performance_within_100ms(self):
        """Verify single-record lookup completes well within 100ms."""
        store = FeatureStore()
        # Populate with a reasonable number of records
        for i in range(10000):
            store.put(f"txn_{i:05d}", _make_features())

        start = time.perf_counter()
        result = store.get("txn_05000")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed_ms < 100.0

    def test_get_preserves_feature_values_exactly(self):
        """Round-trip integrity: stored values are returned exactly."""
        store = FeatureStore()
        v_values = [i * 0.123456789 for i in range(1, 29)]
        features = {"time": 3.14159, "v1_to_v28": v_values, "amount": 0.01}

        store.put("txn_precise", features)
        record = store.get("txn_precise")

        assert record is not None
        assert record.time == 3.14159
        assert record.amount == 0.01
        assert record.v1_to_v28 == v_values


class TestFeatureStorePutBatch:
    """Tests for the put_batch() method — batch ingestion with metadata."""

    def test_put_batch_stores_all_records(self):
        store = FeatureStore()
        records = [
            _make_batch_record("txn_001"),
            _make_batch_record("txn_002"),
            _make_batch_record("txn_003"),
        ]

        batch = store.put_batch(records, data_source="csv", version="v1.0")

        assert batch.record_count == 3
        assert store.record_count == 3
        assert store.get("txn_001") is not None
        assert store.get("txn_002") is not None
        assert store.get("txn_003") is not None

    def test_put_batch_records_metadata(self):
        store = FeatureStore()
        records = [_make_batch_record("txn_001")]

        batch = store.put_batch(records, data_source="kafka", version="v2.0")

        assert batch.data_source == "kafka"
        assert batch.version == "v2.0"
        assert batch.record_count == 1
        assert batch.ingestion_timestamp is not None
        assert batch.batch_id is not None

    def test_put_batch_tracks_data_source(self):
        store = FeatureStore()

        batch_csv = store.put_batch(
            [_make_batch_record("txn_01")], data_source="csv", version="v1"
        )
        batch_kafka = store.put_batch(
            [_make_batch_record("txn_02")], data_source="kafka", version="v2"
        )
        batch_db = store.put_batch(
            [_make_batch_record("txn_03")], data_source="database", version="v3"
        )

        assert batch_csv.data_source == "csv"
        assert batch_kafka.data_source == "kafka"
        assert batch_db.data_source == "database"

    def test_put_batch_assigns_unique_batch_ids(self):
        store = FeatureStore()
        batch1 = store.put_batch(
            [_make_batch_record("txn_01")], data_source="csv", version="v1"
        )
        batch2 = store.put_batch(
            [_make_batch_record("txn_02")], data_source="csv", version="v2"
        )

        assert batch1.batch_id != batch2.batch_id

    def test_put_batch_empty_list(self):
        store = FeatureStore()
        batch = store.put_batch(records=[], data_source="csv", version="v_empty")

        assert batch.record_count == 0
        assert batch.records == []
        assert store.record_count == 0


class TestFeatureStoreGetVersion:
    """Tests for the get_version() method — version-based retrieval."""

    def test_get_version_returns_batch(self):
        store = FeatureStore()
        records = [_make_batch_record("txn_001"), _make_batch_record("txn_002")]
        store.put_batch(records, data_source="csv", version="training_v1")

        batch = store.get_version("training_v1")

        assert batch is not None
        assert batch.version == "training_v1"
        assert batch.record_count == 2
        assert len(batch.records) == 2

    def test_get_version_returns_none_for_missing(self):
        store = FeatureStore()
        assert store.get_version("nonexistent") is None

    def test_get_version_supports_reproducibility(self):
        """Features from a version remain stable for reproducible training."""
        store = FeatureStore()
        records = [
            _make_batch_record("txn_a", time_val=1.0, amount=50.0),
            _make_batch_record("txn_b", time_val=2.0, amount=100.0),
        ]
        store.put_batch(records, data_source="csv", version="run_42")

        # Later, retrieve the same version
        batch = store.get_version("run_42")
        assert batch is not None
        assert batch.records[0].transaction_id == "txn_a"
        assert batch.records[0].time == 1.0
        assert batch.records[0].amount == 50.0
        assert batch.records[1].transaction_id == "txn_b"
        assert batch.records[1].time == 2.0
        assert batch.records[1].amount == 100.0

    def test_later_version_overwrites_previous(self):
        """If the same version is used again, the latest batch wins."""
        store = FeatureStore()
        store.put_batch(
            [_make_batch_record("txn_01", amount=10.0)],
            data_source="csv",
            version="v1",
        )
        store.put_batch(
            [_make_batch_record("txn_02", amount=20.0)],
            data_source="database",
            version="v1",
        )

        batch = store.get_version("v1")
        assert batch is not None
        assert batch.data_source == "database"
        assert batch.records[0].transaction_id == "txn_02"


class TestFeatureStoreMetadata:
    """Tests for metadata retrieval."""

    def test_get_metadata_returns_correct_info(self):
        store = FeatureStore()
        records = [_make_batch_record(f"txn_{i}") for i in range(5)]
        store.put_batch(records, data_source="csv", version="v1.0")

        metadata = store.get_metadata("v1.0")

        assert metadata is not None
        assert metadata["version"] == "v1.0"
        assert metadata["data_source"] == "csv"
        assert metadata["record_count"] == 5
        assert "ingestion_timestamp" in metadata
        assert "batch_id" in metadata

    def test_get_metadata_returns_none_for_missing(self):
        store = FeatureStore()
        assert store.get_metadata("nonexistent") is None

    def test_metadata_timestamp_is_iso_format(self):
        store = FeatureStore()
        store.put_batch(
            [_make_batch_record("txn_01")], data_source="kafka", version="v1"
        )

        metadata = store.get_metadata("v1")
        assert metadata is not None
        # ISO 8601 format includes 'T' separator and '+' for timezone
        assert "T" in metadata["ingestion_timestamp"]


class TestFeatureStoreValidation:
    """Tests for input validation."""

    def test_put_rejects_missing_time(self):
        store = FeatureStore()
        features = {"v1_to_v28": [0.0] * 28, "amount": 10.0}

        with pytest.raises(ValueError, match="time"):
            store.put("txn_001", features)

    def test_put_rejects_missing_v1_to_v28(self):
        store = FeatureStore()
        features = {"time": 0.0, "amount": 10.0}

        with pytest.raises(ValueError, match="v1_to_v28"):
            store.put("txn_001", features)

    def test_put_rejects_missing_amount(self):
        store = FeatureStore()
        features = {"time": 0.0, "v1_to_v28": [0.0] * 28}

        with pytest.raises(ValueError, match="amount"):
            store.put("txn_001", features)

    def test_put_rejects_wrong_feature_count(self):
        store = FeatureStore()
        features = {"time": 0.0, "v1_to_v28": [0.0] * 27, "amount": 10.0}

        with pytest.raises(ValueError, match="28"):
            store.put("txn_001", features)

    def test_put_rejects_v1_to_v28_not_list(self):
        store = FeatureStore()
        features = {"time": 0.0, "v1_to_v28": "invalid", "amount": 10.0}

        with pytest.raises(ValueError, match="list or tuple"):
            store.put("txn_001", features)

    def test_put_batch_rejects_record_without_transaction_id(self):
        store = FeatureStore()
        records = [{"time": 0.0, "v1_to_v28": [0.0] * 28, "amount": 10.0}]

        with pytest.raises(ValueError, match="transaction_id"):
            store.put_batch(records, data_source="csv", version="v1")

    def test_put_batch_rejects_record_with_invalid_features(self):
        store = FeatureStore()
        records = [
            {"transaction_id": "txn_001", "time": 0.0, "v1_to_v28": [0.0] * 10, "amount": 5.0}
        ]

        with pytest.raises(ValueError, match="28"):
            store.put_batch(records, data_source="csv", version="v1")


class TestFeatureStoreGetBatch:
    """Tests for the get_batch() method."""

    def test_get_batch_by_id(self):
        store = FeatureStore()
        batch = store.put_batch(
            [_make_batch_record("txn_01")], data_source="csv", version="v1"
        )

        retrieved = store.get_batch(batch.batch_id)
        assert retrieved is not None
        assert retrieved.batch_id == batch.batch_id
        assert retrieved.version == "v1"

    def test_get_batch_returns_none_for_missing(self):
        store = FeatureStore()
        assert store.get_batch("nonexistent-id") is None
