"""Schema validation for transaction records in the CCFD dataset.

Validates that all 30 required fields (Time, V1–V28, Amount) are present,
numeric, and that Amount >= 0. The Class field is optional (not required
during inference).
"""

from src.models.transaction import ValidationError, ValidationResult


# The 30 required fields for a valid transaction record
REQUIRED_FIELDS: list[str] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


class SchemaValidator:
    """Validates transaction records against the CCFD schema.

    The schema requires exactly 30 numeric fields: Time, V1–V28, and Amount.
    Amount must be >= 0. The Class field is optional.

    Usage:
        validator = SchemaValidator()
        result = validator.validate({"Time": 0.0, "V1": -1.35, ..., "Amount": 149.62})
    """

    def __init__(self) -> None:
        """Initialize the SchemaValidator with the required field definitions."""
        self._required_fields = REQUIRED_FIELDS

    def validate(self, record: dict) -> ValidationResult:
        """Validate a transaction record against the CCFD schema.

        Checks:
        1. All 30 required fields are present.
        2. All required fields contain numeric values (int or float).
        3. Amount is >= 0.

        Args:
            record: Dictionary representing a single transaction record.

        Returns:
            ValidationResult with is_valid=True if all checks pass,
            otherwise is_valid=False with a list of ValidationError entries.
        """
        errors: list[ValidationError] = []

        for field_name in self._required_fields:
            if field_name not in record:
                errors.append(
                    ValidationError(
                        field_name=field_name,
                        violation_type="missing",
                        message=f"Required field '{field_name}' is missing.",
                    )
                )
            else:
                value = record[field_name]
                if not isinstance(value, (int, float)):
                    errors.append(
                        ValidationError(
                            field_name=field_name,
                            violation_type="non_numeric",
                            message=f"Field '{field_name}' must be numeric, got {type(value).__name__}.",
                        )
                    )
                elif field_name == "Amount" and value < 0:
                    errors.append(
                        ValidationError(
                            field_name=field_name,
                            violation_type="invalid_amount",
                            message=f"Field 'Amount' must be >= 0, got {value}.",
                        )
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )
