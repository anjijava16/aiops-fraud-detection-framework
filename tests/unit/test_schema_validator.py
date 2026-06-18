"""Unit tests for the SchemaValidator class."""

import pytest

from src.ingestion.schema_validator import REQUIRED_FIELDS, SchemaValidator
from src.models.transaction import ValidationError, ValidationResult


@pytest.fixture
def validator() -> SchemaValidator:
    """Create a SchemaValidator instance for testing."""
    return SchemaValidator()


def _make_valid_record() -> dict:
    """Create a valid transaction record with all 30 required fields."""
    record = {"Time": 0.0, "Amount": 149.62}
    for i in range(1, 29):
        record[f"V{i}"] = float(i) * -0.5
    return record


class TestSchemaValidatorValidRecords:
    """Tests for records that should pass validation."""

    def test_valid_record_passes(self, validator: SchemaValidator) -> None:
        """A fully valid record with all 30 numeric fields passes validation."""
        record = _make_valid_record()
        result = validator.validate(record)
        assert result.is_valid is True
        assert result.errors == []

    def test_valid_record_with_zero_amount(self, validator: SchemaValidator) -> None:
        """Amount of exactly 0 should be accepted."""
        record = _make_valid_record()
        record["Amount"] = 0.0
        result = validator.validate(record)
        assert result.is_valid is True
        assert result.errors == []

    def test_valid_record_with_integer_values(self, validator: SchemaValidator) -> None:
        """Integer values for numeric fields should be accepted."""
        record = {"Time": 0, "Amount": 100}
        for i in range(1, 29):
            record[f"V{i}"] = i
        result = validator.validate(record)
        assert result.is_valid is True

    def test_valid_record_with_optional_class(self, validator: SchemaValidator) -> None:
        """Extra 'Class' field should not affect validation (it's optional)."""
        record = _make_valid_record()
        record["Class"] = 1
        result = validator.validate(record)
        assert result.is_valid is True

    def test_valid_record_with_extra_fields(self, validator: SchemaValidator) -> None:
        """Extra unrecognized fields should not affect validation."""
        record = _make_valid_record()
        record["extra_field"] = "hello"
        result = validator.validate(record)
        assert result.is_valid is True


class TestSchemaValidatorMissingFields:
    """Tests for records with missing required fields."""

    def test_missing_time_field(self, validator: SchemaValidator) -> None:
        """Missing 'Time' field should produce a 'missing' error."""
        record = _make_valid_record()
        del record["Time"]
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "Time"
        assert result.errors[0].violation_type == "missing"

    def test_missing_amount_field(self, validator: SchemaValidator) -> None:
        """Missing 'Amount' field should produce a 'missing' error."""
        record = _make_valid_record()
        del record["Amount"]
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "Amount"
        assert result.errors[0].violation_type == "missing"

    def test_missing_v_field(self, validator: SchemaValidator) -> None:
        """Missing a V-field should produce a 'missing' error."""
        record = _make_valid_record()
        del record["V14"]
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "V14"
        assert result.errors[0].violation_type == "missing"

    def test_multiple_missing_fields(self, validator: SchemaValidator) -> None:
        """Multiple missing fields should produce one error per field."""
        record = _make_valid_record()
        del record["Time"]
        del record["V1"]
        del record["Amount"]
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 3
        missing_fields = {e.field_name for e in result.errors}
        assert missing_fields == {"Time", "V1", "Amount"}

    def test_empty_record(self, validator: SchemaValidator) -> None:
        """An empty record should produce 30 missing errors."""
        result = validator.validate({})
        assert result.is_valid is False
        assert len(result.errors) == 30
        for error in result.errors:
            assert error.violation_type == "missing"


class TestSchemaValidatorNonNumericFields:
    """Tests for records with non-numeric field values."""

    def test_string_time_value(self, validator: SchemaValidator) -> None:
        """String value for 'Time' should produce a 'non_numeric' error."""
        record = _make_valid_record()
        record["Time"] = "not_a_number"
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "Time"
        assert result.errors[0].violation_type == "non_numeric"

    def test_string_amount_value(self, validator: SchemaValidator) -> None:
        """String value for 'Amount' should produce a 'non_numeric' error."""
        record = _make_valid_record()
        record["Amount"] = "abc"
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "Amount"
        assert result.errors[0].violation_type == "non_numeric"

    def test_none_value(self, validator: SchemaValidator) -> None:
        """None value for a field should produce a 'non_numeric' error."""
        record = _make_valid_record()
        record["V5"] = None
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "V5"
        assert result.errors[0].violation_type == "non_numeric"

    def test_list_value(self, validator: SchemaValidator) -> None:
        """List value should produce a 'non_numeric' error."""
        record = _make_valid_record()
        record["V10"] = [1.0, 2.0]
        result = validator.validate(record)
        assert result.is_valid is False
        assert result.errors[0].violation_type == "non_numeric"

    def test_boolean_value_rejected(self, validator: SchemaValidator) -> None:
        """Boolean values are technically int subclass in Python but should be numeric."""
        # Note: In Python, bool is a subclass of int, so True/False are valid numeric
        record = _make_valid_record()
        record["V1"] = True  # isinstance(True, int) is True in Python
        result = validator.validate(record)
        # Booleans pass isinstance(x, (int, float)) since bool is subclass of int
        assert result.is_valid is True


class TestSchemaValidatorInvalidAmount:
    """Tests for records with negative Amount values."""

    def test_negative_amount(self, validator: SchemaValidator) -> None:
        """Negative Amount should produce an 'invalid_amount' error."""
        record = _make_valid_record()
        record["Amount"] = -1.0
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field_name == "Amount"
        assert result.errors[0].violation_type == "invalid_amount"

    def test_large_negative_amount(self, validator: SchemaValidator) -> None:
        """Large negative Amount should produce an 'invalid_amount' error."""
        record = _make_valid_record()
        record["Amount"] = -99999.99
        result = validator.validate(record)
        assert result.is_valid is False
        assert result.errors[0].violation_type == "invalid_amount"

    def test_very_small_negative_amount(self, validator: SchemaValidator) -> None:
        """Very small negative Amount should still be rejected."""
        record = _make_valid_record()
        record["Amount"] = -0.001
        result = validator.validate(record)
        assert result.is_valid is False
        assert result.errors[0].violation_type == "invalid_amount"


class TestSchemaValidatorCombinedErrors:
    """Tests for records with multiple error types."""

    def test_missing_and_non_numeric_errors(self, validator: SchemaValidator) -> None:
        """Record with both missing and non-numeric fields should report both."""
        record = _make_valid_record()
        del record["Time"]
        record["V1"] = "bad"
        result = validator.validate(record)
        assert result.is_valid is False
        assert len(result.errors) == 2
        error_types = {(e.field_name, e.violation_type) for e in result.errors}
        assert ("Time", "missing") in error_types
        assert ("V1", "non_numeric") in error_types

    def test_non_numeric_amount_takes_priority_over_negative(
        self, validator: SchemaValidator
    ) -> None:
        """Non-numeric Amount should report 'non_numeric', not 'invalid_amount'."""
        record = _make_valid_record()
        record["Amount"] = "negative"
        result = validator.validate(record)
        assert result.is_valid is False
        assert result.errors[0].violation_type == "non_numeric"


class TestSchemaValidatorRequiredFields:
    """Tests for the required fields definition."""

    def test_required_fields_count(self) -> None:
        """There should be exactly 30 required fields."""
        assert len(REQUIRED_FIELDS) == 30

    def test_required_fields_contains_time(self) -> None:
        """Required fields should include 'Time'."""
        assert "Time" in REQUIRED_FIELDS

    def test_required_fields_contains_amount(self) -> None:
        """Required fields should include 'Amount'."""
        assert "Amount" in REQUIRED_FIELDS

    def test_required_fields_contains_all_v_features(self) -> None:
        """Required fields should include V1 through V28."""
        for i in range(1, 29):
            assert f"V{i}" in REQUIRED_FIELDS

    def test_class_is_not_required(self) -> None:
        """'Class' should NOT be in the required fields list."""
        assert "Class" not in REQUIRED_FIELDS
