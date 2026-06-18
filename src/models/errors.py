"""Custom exception hierarchy for the AIOps Fraud Detection Framework.

Each layer has a dedicated exception class to enable targeted error handling
and clear identification of failure origin in pipeline orchestration.
"""


class FrameworkError(Exception):
    """Base exception for all framework errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(FrameworkError):
    """Raised when configuration loading, parsing, or validation fails.

    Examples: missing YAML file, invalid parameter range, missing required field.
    """

    pass


class IngestionError(FrameworkError):
    """Raised when data ingestion encounters an unrecoverable error.

    Examples: Kafka connection failure, CSV column mismatch, null values in required columns.
    """

    pass


class PreprocessingError(FrameworkError):
    """Raised when preprocessing operations fail.

    Examples: insufficient minority samples for SMOTE, unsupported scaler type,
    correlation threshold out of range, dataset too small for k-fold.
    """

    pass


class EnsembleError(FrameworkError):
    """Raised when ensemble model training or prediction fails.

    Examples: hyperparameter out of valid range, data format error,
    zero samples, mismatched dimensions, base model failure.
    """

    pass


class ExplainabilityError(FrameworkError):
    """Raised when explainability computation fails.

    Examples: SHAP timeout, transaction not found, LIME surrogate failure.
    """

    pass


class MLflowError(FrameworkError):
    """Raised when MLflow operations fail.

    Examples: tracking server unreachable, model registration failure,
    promotion threshold not met, run logging failure.
    """

    pass


class APIError(FrameworkError):
    """Raised when FastAPI serving layer encounters an error.

    Examples: model not loaded, authentication failure, rate limit exceeded.
    """

    pass


class MonitoringError(FrameworkError):
    """Raised when monitoring operations fail.

    Examples: drift detection failure, alert delivery exhausted retries,
    Prometheus export error, retraining loop failure.
    """

    pass


class PipelineError(FrameworkError):
    """Raised when pipeline orchestration encounters an error.

    Examples: layer execution failure, timeout exceeded, concurrent execution conflict.
    """

    pass
