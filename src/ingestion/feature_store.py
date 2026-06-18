"""Feature Store for storing and retrieving validated transaction features.

Provides an in-memory implementation backed by dictionaries for single-node
deployment. Supports versioning for reproducible model training runs and
metadata tracking per batch (data source, ingestion timestamp, record count).

Performance target: single-record lookups within 100ms (trivially met by
in-memory dict-based storage).
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class FeatureRecord:
    """A single feature record stored in the feature store.

    Attributes:
        transaction_id: Unique identifier for the transaction.
        time: Elapsed seconds since first transaction in the dataset.
        v1_to_v28: Exactly 28 PCA feature values (V1 through V28).
        amount: Transaction amount (>= 0).
        ingestion_timestamp: ISO 8601 UTC timestamp when the record was ingested.
    """

    transaction_id: str
    time: float
    v1_to_v28: list[float]  # Exactly 28 PCA features
    amount: float
    ingestion_timestamp: str


@dataclass
class FeatureBatch:
    """Metadata and records for a batch of features ingested together.

    Attributes:
        batch_id: Unique identifier for the batch.
        version: Version tag for reproducibility (e.g., model training run ID).
        data_source: Origin of the data — "kafka", "csv", or "database".
        ingestion_timestamp: ISO 8601 UTC timestamp when the batch was ingested.
        record_count: Number of records in the batch.
        records: List of FeatureRecord instances in this batch.
    """

    batch_id: str
    version: str
    data_source: str  # "kafka" | "csv" | "database"
    ingestion_timestamp: str  # ISO 8601
    record_count: int
    records: list[FeatureRecord] = field(default_factory=list)


class FeatureStore:
    """In-memory feature store with versioning and metadata tracking.

    Stores V1–V28, Time, and Amount indexed by transaction ID. Supports
    versioned batch ingestion for reproducible model training, and tracks
    metadata (data source, ingestion timestamp, record count) per batch.

    Usage:
        store = FeatureStore()

        # Store a single record
        store.put("txn_001", {"time": 0.0, "v1_to_v28": [...], "amount": 149.62})

        # Store a batch with metadata
        store.put_batch(records=[...], data_source="csv", version="v1.0")

        # Retrieve a single record
        record = store.get("txn_001")

        # Retrieve all records from a version
        batch = store.get_version("v1.0")
    """

    def __init__(self) -> None:
        """Initialize the feature store with empty storage."""
        # Primary index: transaction_id -> FeatureRecord
        self._records: dict[str, FeatureRecord] = {}
        # Version index: version -> FeatureBatch
        self._versions: dict[str, FeatureBatch] = {}
        # Batch index: batch_id -> FeatureBatch
        self._batches: dict[str, FeatureBatch] = {}

    def put(self, transaction_id: str, features: dict) -> FeatureRecord:
        """Store features for a single transaction.

        Args:
            transaction_id: Unique identifier for the transaction.
            features: Dictionary with keys:
                - "time" (float): Elapsed seconds since first transaction.
                - "v1_to_v28" (list[float]): Exactly 28 PCA feature values.
                - "amount" (float): Transaction amount.

        Returns:
            The stored FeatureRecord.

        Raises:
            ValueError: If required keys are missing or v1_to_v28 has wrong length.
        """
        self._validate_features(features)

        timestamp = datetime.now(timezone.utc).isoformat()
        record = FeatureRecord(
            transaction_id=transaction_id,
            time=features["time"],
            v1_to_v28=list(features["v1_to_v28"]),
            amount=features["amount"],
            ingestion_timestamp=timestamp,
        )
        self._records[transaction_id] = record
        return record

    def put_batch(
        self,
        records: list[dict],
        data_source: str,
        version: str,
    ) -> FeatureBatch:
        """Store a batch of feature records with metadata.

        Each record dict must contain:
            - "transaction_id" (str)
            - "time" (float)
            - "v1_to_v28" (list[float]) — exactly 28 values
            - "amount" (float)

        Args:
            records: List of feature dictionaries to store.
            data_source: Origin of the data ("kafka", "csv", or "database").
            version: Version tag for reproducibility.

        Returns:
            The created FeatureBatch with metadata.

        Raises:
            ValueError: If any record is missing required keys or has invalid data.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        batch_id = str(uuid.uuid4())

        feature_records: list[FeatureRecord] = []
        for record_data in records:
            transaction_id = record_data.get("transaction_id")
            if not transaction_id:
                raise ValueError("Each record must have a 'transaction_id' key.")

            features = {
                "time": record_data.get("time"),
                "v1_to_v28": record_data.get("v1_to_v28"),
                "amount": record_data.get("amount"),
            }
            self._validate_features(features)

            feature_record = FeatureRecord(
                transaction_id=transaction_id,
                time=features["time"],
                v1_to_v28=list(features["v1_to_v28"]),
                amount=features["amount"],
                ingestion_timestamp=timestamp,
            )
            feature_records.append(feature_record)
            self._records[transaction_id] = feature_record

        batch = FeatureBatch(
            batch_id=batch_id,
            version=version,
            data_source=data_source,
            ingestion_timestamp=timestamp,
            record_count=len(feature_records),
            records=feature_records,
        )
        self._versions[version] = batch
        self._batches[batch_id] = batch
        return batch

    def get(self, transaction_id: str) -> Optional[FeatureRecord]:
        """Retrieve features for a single transaction by ID.

        Performance: O(1) dict lookup, well within 100ms target.

        Args:
            transaction_id: The transaction ID to look up.

        Returns:
            The FeatureRecord if found, None otherwise.
        """
        return self._records.get(transaction_id)

    def get_version(self, version: str) -> Optional[FeatureBatch]:
        """Retrieve all features from a specific version/batch.

        Args:
            version: The version tag to look up.

        Returns:
            The FeatureBatch if found, None otherwise.
        """
        return self._versions.get(version)

    def get_batch(self, batch_id: str) -> Optional[FeatureBatch]:
        """Retrieve a batch by its batch ID.

        Args:
            batch_id: The unique batch identifier.

        Returns:
            The FeatureBatch if found, None otherwise.
        """
        return self._batches.get(batch_id)

    def get_metadata(self, version: str) -> Optional[dict]:
        """Retrieve metadata for a specific version.

        Args:
            version: The version tag to look up.

        Returns:
            Dictionary with data_source, ingestion_timestamp, and record_count
            if found, None otherwise.
        """
        batch = self._versions.get(version)
        if batch is None:
            return None
        return {
            "batch_id": batch.batch_id,
            "version": batch.version,
            "data_source": batch.data_source,
            "ingestion_timestamp": batch.ingestion_timestamp,
            "record_count": batch.record_count,
        }

    @property
    def record_count(self) -> int:
        """Total number of feature records in the store."""
        return len(self._records)

    @property
    def version_count(self) -> int:
        """Total number of versioned batches in the store."""
        return len(self._versions)

    def _validate_features(self, features: dict) -> None:
        """Validate that a features dictionary has the required structure.

        Args:
            features: Dictionary with time, v1_to_v28, and amount.

        Raises:
            ValueError: If required keys are missing or data is invalid.
        """
        if "time" not in features or features["time"] is None:
            raise ValueError("Feature 'time' is required.")
        if "v1_to_v28" not in features or features["v1_to_v28"] is None:
            raise ValueError("Feature 'v1_to_v28' is required.")
        if "amount" not in features or features["amount"] is None:
            raise ValueError("Feature 'amount' is required.")

        v_features = features["v1_to_v28"]
        if not isinstance(v_features, (list, tuple)):
            raise ValueError("Feature 'v1_to_v28' must be a list or tuple.")
        if len(v_features) != 28:
            raise ValueError(
                f"Feature 'v1_to_v28' must contain exactly 28 values, got {len(v_features)}."
            )
