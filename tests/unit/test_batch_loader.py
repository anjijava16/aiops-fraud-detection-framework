"""Unit tests for the BatchLoader class."""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.ingestion.batch_loader import BatchLoader, EXPECTED_COLUMNS, EXPECTED_COLUMN_COUNT
from src.models.errors import IngestionError


# --- Fixtures ---


@pytest.fixture
def loader():
    """Create a BatchLoader with default batch size."""
    return BatchLoader(batch_size=1000)


@pytest.fixture
def valid_df():
    """Create a valid 31-column DataFrame matching CCFD schema."""
    data = {
        "Time": [0.0, 1.0, 2.0],
        **{f"V{i}": [float(i)] * 3 for i in range(1, 29)},
        "Amount": [100.0, 200.0, 50.0],
        "Class": [0, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_csv(valid_df, tmp_path):
    """Write a valid CSV and return its path."""
    path = tmp_path / "valid.csv"
    valid_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_with_nulls(tmp_path):
    """Write a CSV with null values in required columns."""
    data = {
        "Time": [0.0, None, 2.0],
        **{f"V{i}": [float(i)] * 3 for i in range(1, 29)},
        "Amount": [100.0, 200.0, None],
        "Class": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "nulls.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_wrong_columns(tmp_path):
    """Write a CSV with wrong number of columns."""
    data = {
        "Time": [0.0, 1.0],
        "V1": [1.0, 2.0],
        "Amount": [100.0, 200.0],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "wrong_cols.csv"
    df.to_csv(path, index=False)
    return str(path)


# --- Constructor Tests ---


class TestBatchLoaderInit:
    """Tests for BatchLoader initialization."""

    def test_valid_batch_size(self):
        """BatchLoader accepts a positive integer batch_size."""
        loader = BatchLoader(batch_size=5000)
        assert loader.batch_size == 5000

    def test_default_batch_size(self):
        """BatchLoader uses default batch_size of 10000."""
        loader = BatchLoader()
        assert loader.batch_size == 10000

    def test_invalid_batch_size_zero(self):
        """BatchLoader rejects zero batch_size."""
        with pytest.raises(IngestionError, match="batch_size must be a positive integer"):
            BatchLoader(batch_size=0)

    def test_invalid_batch_size_negative(self):
        """BatchLoader rejects negative batch_size."""
        with pytest.raises(IngestionError, match="batch_size must be a positive integer"):
            BatchLoader(batch_size=-1)

    def test_invalid_batch_size_non_integer(self):
        """BatchLoader rejects non-integer batch_size."""
        with pytest.raises(IngestionError, match="batch_size must be a positive integer"):
            BatchLoader(batch_size=10.5)


# --- CSV Loading Tests ---


class TestLoadCSV:
    """Tests for BatchLoader.load_csv method."""

    def test_load_valid_csv(self, loader, valid_csv, valid_df):
        """Successfully loads a valid 31-column CSV file."""
        result = loader.load_csv(valid_csv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert len(result.columns) == EXPECTED_COLUMN_COUNT

    def test_load_csv_file_not_found(self, loader):
        """Raises IngestionError for non-existent file."""
        with pytest.raises(IngestionError, match="CSV file not found"):
            loader.load_csv("/nonexistent/path/to/file.csv")

    def test_load_csv_wrong_column_count(self, loader, csv_wrong_columns):
        """Raises IngestionError when CSV doesn't have exactly 31 columns."""
        with pytest.raises(IngestionError, match="Expected exactly 31 columns"):
            loader.load_csv(csv_wrong_columns)

    def test_load_csv_null_values_rejected(self, loader, csv_with_nulls):
        """Raises IngestionError when required columns contain nulls."""
        with pytest.raises(IngestionError, match="Null values found in required columns"):
            loader.load_csv(csv_with_nulls)

    def test_null_error_reports_affected_columns(self, loader, csv_with_nulls):
        """IngestionError details include exactly which columns have nulls."""
        with pytest.raises(IngestionError) as exc_info:
            loader.load_csv(csv_with_nulls)

        null_columns = exc_info.value.details["null_columns"]
        assert "Time" in null_columns
        assert "Amount" in null_columns

    def test_null_in_class_column_allowed(self, loader, tmp_path):
        """Null values in the Class column do not cause rejection."""
        data = {
            "Time": [0.0, 1.0, 2.0],
            **{f"V{i}": [float(i)] * 3 for i in range(1, 29)},
            "Amount": [100.0, 200.0, 50.0],
            "Class": [0, None, 1],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "null_class.csv"
        df.to_csv(path, index=False)

        result = loader.load_csv(str(path))
        assert len(result) == 3

    def test_load_csv_logs_record_count(self, loader, valid_csv, caplog):
        """Logs record count after successful load."""
        import logging

        with caplog.at_level(logging.INFO, logger="src.ingestion.batch_loader"):
            loader.load_csv(valid_csv)

        assert "3 records" in caplog.text

    def test_load_csv_logs_class_distribution(self, loader, valid_csv, caplog):
        """Logs class distribution after successful load."""
        import logging

        with caplog.at_level(logging.INFO, logger="src.ingestion.batch_loader"):
            loader.load_csv(valid_csv)

        assert "non-fraud" in caplog.text
        assert "fraud" in caplog.text

    def test_load_csv_too_many_columns(self, loader, tmp_path):
        """Raises IngestionError when CSV has more than 31 columns."""
        data = {
            "Time": [0.0],
            **{f"V{i}": [float(i)] for i in range(1, 29)},
            "Amount": [100.0],
            "Class": [0],
            "Extra": [999.0],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "extra_cols.csv"
        df.to_csv(path, index=False)

        with pytest.raises(IngestionError, match="Expected exactly 31 columns, got 32"):
            loader.load_csv(str(path))


# --- Database Loading Tests ---


class TestLoadDatabase:
    """Tests for BatchLoader.load_database method."""

    def test_load_database_success(self, loader, valid_df, tmp_path):
        """Successfully loads records from a SQLite database."""
        db_path = tmp_path / "test.db"
        connection_string = f"sqlite:///{db_path}"

        # Write valid data to SQLite
        from sqlalchemy import create_engine

        engine = create_engine(connection_string)
        valid_df.to_sql("transactions", engine, index=False)

        result = loader.load_database(connection_string, table="transactions")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert len(result.columns) == EXPECTED_COLUMN_COUNT

    def test_load_database_wrong_columns(self, loader, tmp_path):
        """Raises IngestionError when DB table doesn't have 31 columns."""
        db_path = tmp_path / "bad.db"
        connection_string = f"sqlite:///{db_path}"

        from sqlalchemy import create_engine

        engine = create_engine(connection_string)
        pd.DataFrame({"A": [1], "B": [2]}).to_sql("bad_table", engine, index=False)

        with pytest.raises(IngestionError, match="Expected exactly 31 columns"):
            loader.load_database(connection_string, table="bad_table")

    def test_load_database_null_values_rejected(self, loader, tmp_path):
        """Raises IngestionError when DB records contain nulls in required columns."""
        db_path = tmp_path / "nulls.db"
        connection_string = f"sqlite:///{db_path}"

        data = {
            "Time": [0.0, None],
            **{f"V{i}": [float(i)] * 2 for i in range(1, 29)},
            "Amount": [100.0, 200.0],
            "Class": [0, 1],
        }
        df = pd.DataFrame(data)

        from sqlalchemy import create_engine

        engine = create_engine(connection_string)
        df.to_sql("transactions", engine, index=False)

        with pytest.raises(IngestionError, match="Null values found in required columns"):
            loader.load_database(connection_string, table="transactions")

    def test_load_database_custom_batch_size(self, loader, valid_df, tmp_path):
        """Respects custom batch_size parameter."""
        db_path = tmp_path / "batch.db"
        connection_string = f"sqlite:///{db_path}"

        from sqlalchemy import create_engine

        engine = create_engine(connection_string)
        valid_df.to_sql("transactions", engine, index=False)

        result = loader.load_database(
            connection_string, table="transactions", batch_size=2
        )
        assert len(result) == 3

    def test_load_database_connection_failure(self, loader):
        """Raises IngestionError on database connection failure."""
        with pytest.raises(IngestionError, match="Failed to load from database"):
            loader.load_database(
                "sqlite:///nonexistent/path/db.sqlite",
                table="transactions",
            )

    def test_load_database_logs_summary(self, loader, valid_df, tmp_path, caplog):
        """Logs record count and class distribution on success."""
        import logging

        db_path = tmp_path / "log_test.db"
        connection_string = f"sqlite:///{db_path}"

        from sqlalchemy import create_engine

        engine = create_engine(connection_string)
        valid_df.to_sql("transactions", engine, index=False)

        with caplog.at_level(logging.INFO, logger="src.ingestion.batch_loader"):
            loader.load_database(connection_string, table="transactions")

        assert "3 records" in caplog.text
        assert "non-fraud" in caplog.text
