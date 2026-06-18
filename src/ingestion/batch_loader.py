"""Batch data loader for CSV files and database sources.

Loads the CCFD dataset from CSV files or database connections, validates column
structure (exactly 31 columns), checks for null values in required columns, and
logs record count and class distribution after successful loading.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.ingestion.schema_validator import REQUIRED_FIELDS
from src.models.errors import IngestionError


logger = logging.getLogger(__name__)

# Expected columns for the CCFD dataset (31 total)
EXPECTED_COLUMNS: list[str] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]

EXPECTED_COLUMN_COUNT = 31


class BatchLoader:
    """Loads transaction data from CSV files or database connections.

    Validates that the loaded data has exactly 31 columns (Time, V1–V28, Amount, Class),
    rejects batches with null values in required columns, and logs record count and
    class distribution on successful loads.

    Usage:
        loader = BatchLoader(batch_size=10000)
        df = loader.load_csv("path/to/creditcard.csv")
        df = loader.load_database("sqlite:///transactions.db", table="transactions")
    """

    def __init__(self, batch_size: int = 10000) -> None:
        """Initialize the BatchLoader.

        Args:
            batch_size: Number of records to load per batch from database sources.
                        Must be a positive integer.

        Raises:
            IngestionError: If batch_size is not a positive integer.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise IngestionError(
                "batch_size must be a positive integer",
                details={"batch_size": batch_size},
            )
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """Return the configured batch size."""
        return self._batch_size

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load a CSV file and validate its structure.

        Validates that the CSV contains exactly 31 columns (Time, V1–V28, Amount, Class).
        Rejects the batch if any required columns contain null values.
        Logs record count and class distribution on success.

        Args:
            file_path: Path to the CSV file to load.

        Returns:
            A pandas DataFrame containing the validated dataset.

        Raises:
            IngestionError: If the file cannot be read, has wrong column count,
                           or contains null values in required columns.
        """
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise IngestionError(
                f"CSV file not found: {file_path}",
                details={"file_path": file_path},
            )
        except Exception as e:
            raise IngestionError(
                f"Failed to read CSV file: {file_path}",
                details={"file_path": file_path, "error": str(e)},
            )

        self._validate_columns(df, source=file_path)
        self._check_nulls(df, source=file_path)
        self._log_summary(df, source=file_path)

        return df

    def load_database(
        self,
        connection_string: str,
        table: str,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load transaction records from a database table.

        Queries the specified table and loads records in batches of the configured
        (or overridden) batch size. Validates column structure and null values.

        Args:
            connection_string: SQLAlchemy-compatible database connection string.
            table: Name of the database table to query.
            batch_size: Optional override for the default batch size.

        Returns:
            A pandas DataFrame containing all loaded and validated records.

        Raises:
            IngestionError: If the connection fails, column count is wrong,
                           or null values are found in required columns.
        """
        effective_batch_size = batch_size if batch_size is not None else self._batch_size

        try:
            chunks = pd.read_sql_table(
                table_name=table,
                con=connection_string,
                chunksize=effective_batch_size,
            )
            df = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            raise IngestionError(
                f"Failed to load from database: {connection_string}",
                details={
                    "connection_string": connection_string,
                    "table": table,
                    "error": str(e),
                },
            )

        self._validate_columns(df, source=f"database:{table}")
        self._check_nulls(df, source=f"database:{table}")
        self._log_summary(df, source=f"database:{table}")

        return df

    def _validate_columns(self, df: pd.DataFrame, source: str) -> None:
        """Validate that the DataFrame has exactly 31 columns.

        Args:
            df: The DataFrame to validate.
            source: Description of the data source for error messages.

        Raises:
            IngestionError: If the column count is not exactly 31.
        """
        actual_count = len(df.columns)
        if actual_count != EXPECTED_COLUMN_COUNT:
            raise IngestionError(
                f"Expected exactly {EXPECTED_COLUMN_COUNT} columns, got {actual_count}",
                details={
                    "source": source,
                    "expected_columns": EXPECTED_COLUMN_COUNT,
                    "actual_columns": actual_count,
                    "columns_found": list(df.columns),
                },
            )

    def _check_nulls(self, df: pd.DataFrame, source: str) -> None:
        """Check for null values in required columns.

        Required columns are the 30 fields from REQUIRED_FIELDS (Time, V1–V28, Amount).
        The Class column is not required to be non-null.

        Args:
            df: The DataFrame to check.
            source: Description of the data source for error messages.

        Raises:
            IngestionError: If any required columns contain null values,
                           reporting exactly which columns are affected.
        """
        null_columns: list[str] = []

        for col in REQUIRED_FIELDS:
            if col in df.columns and df[col].isnull().any():
                null_columns.append(col)

        if null_columns:
            raise IngestionError(
                f"Null values found in required columns: {null_columns}",
                details={
                    "source": source,
                    "null_columns": null_columns,
                },
            )

    def _log_summary(self, df: pd.DataFrame, source: str) -> None:
        """Log record count and class distribution after successful load.

        Args:
            df: The successfully loaded DataFrame.
            source: Description of the data source.
        """
        record_count = len(df)
        logger.info(
            "Batch loaded successfully from %s: %d records",
            source,
            record_count,
        )

        if "Class" in df.columns:
            class_counts = df["Class"].value_counts().to_dict()
            fraud_count = class_counts.get(1, 0)
            non_fraud_count = class_counts.get(0, 0)
            logger.info(
                "Class distribution — non-fraud: %d, fraud: %d (%.4f%% fraud)",
                non_fraud_count,
                fraud_count,
                (fraud_count / record_count * 100) if record_count > 0 else 0.0,
            )
