"""Transaction-related data models for the ingestion layer."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationError:
    """Describes a single field validation failure.

    Attributes:
        field_name: Name of the field that failed validation.
        violation_type: Category of violation — one of "missing", "non_numeric", "invalid_amount".
        message: Human-readable description of the validation failure.
    """

    field_name: str
    violation_type: str  # "missing", "non_numeric", "invalid_amount"
    message: str


@dataclass
class ValidationResult:
    """Result of schema validation on a transaction record.

    Attributes:
        is_valid: True if the record passes all schema checks.
        errors: List of ValidationError instances for each failing field.
    """

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)


@dataclass
class TransactionRecord:
    """A single transaction record from the CCFD dataset.

    Attributes:
        transaction_id: Unique identifier for the transaction.
        time: Elapsed seconds since first transaction in the dataset.
        features: Dictionary mapping feature names (V1–V28) to their float values.
        amount: Transaction amount (must be >= 0).
        class_label: Fraud label (0 or 1). None during inference.
    """

    transaction_id: str
    time: float
    features: dict[str, float]  # V1-V28
    amount: float
    class_label: Optional[int] = None  # None for inference
