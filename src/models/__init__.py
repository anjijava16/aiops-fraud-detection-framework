"""Shared data models and type definitions for the AIOps Fraud Detection Framework."""

from src.models.transaction import TransactionRecord, ValidationResult, ValidationError
from src.models.config_schema import PreprocessingConfig, DataSplits
from src.models.ensemble import ModelResult, EnsemblePrediction, EnsembleConfig
from src.models.explainability import SHAPExplanation, LIMEExplanation, AuditEntry
from src.models.monitoring import DriftResult, AlertEvent, RetrainResult
from src.models.errors import (
    ConfigError,
    IngestionError,
    PreprocessingError,
    EnsembleError,
    ExplainabilityError,
    MLflowError,
    APIError,
    MonitoringError,
    PipelineError,
)

__all__ = [
    # Transaction models
    "TransactionRecord",
    "ValidationResult",
    "ValidationError",
    # Config models
    "PreprocessingConfig",
    "DataSplits",
    # Ensemble models
    "ModelResult",
    "EnsemblePrediction",
    "EnsembleConfig",
    # Explainability models
    "SHAPExplanation",
    "LIMEExplanation",
    "AuditEntry",
    # Monitoring models
    "DriftResult",
    "AlertEvent",
    "RetrainResult",
    # Errors
    "ConfigError",
    "IngestionError",
    "PreprocessingError",
    "EnsembleError",
    "ExplainabilityError",
    "MLflowError",
    "APIError",
    "MonitoringError",
    "PipelineError",
]
