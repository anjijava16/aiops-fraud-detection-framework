"""Unit tests for shared data models and type definitions."""

import numpy as np
import pytest

from src.models.transaction import TransactionRecord, ValidationResult, ValidationError
from src.models.config_schema import PreprocessingConfig, DataSplits
from src.models.ensemble import ModelResult, EnsemblePrediction, EnsembleConfig
from src.models.explainability import SHAPExplanation, LIMEExplanation, AuditEntry
from src.models.monitoring import DriftResult, AlertEvent, RetrainResult
from src.models.errors import (
    FrameworkError,
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


class TestTransactionModels:
    def test_transaction_record_creation(self):
        record = TransactionRecord(
            transaction_id="tx-001",
            time=1.5,
            features={"V1": 0.5, "V2": -0.3},
            amount=100.0,
            class_label=1,
        )
        assert record.transaction_id == "tx-001"
        assert record.time == 1.5
        assert record.features == {"V1": 0.5, "V2": -0.3}
        assert record.amount == 100.0
        assert record.class_label == 1

    def test_transaction_record_inference_mode(self):
        record = TransactionRecord(
            transaction_id="tx-002",
            time=0.0,
            features={"V1": 1.0},
            amount=50.0,
        )
        assert record.class_label is None

    def test_validation_error(self):
        error = ValidationError(
            field_name="Amount",
            violation_type="invalid_amount",
            message="Amount must be >= 0",
        )
        assert error.field_name == "Amount"
        assert error.violation_type == "invalid_amount"

    def test_validation_result_valid(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert result.errors == []

    def test_validation_result_invalid(self):
        errors = [
            ValidationError("V1", "missing", "Field V1 is required"),
            ValidationError("Amount", "non_numeric", "Amount must be numeric"),
        ]
        result = ValidationResult(is_valid=False, errors=errors)
        assert not result.is_valid
        assert len(result.errors) == 2


class TestConfigModels:
    def test_preprocessing_config_defaults(self):
        config = PreprocessingConfig()
        assert config.smote_target_ratio == 1.0
        assert config.scaler_type == "StandardScaler"
        assert config.correlation_threshold == 0.95
        assert config.test_size == 0.2
        assert config.k_folds == 5
        assert config.random_seed == 42

    def test_preprocessing_config_custom(self):
        config = PreprocessingConfig(
            smote_target_ratio=0.5,
            scaler_type="MinMaxScaler",
            correlation_threshold=0.9,
            test_size=0.3,
            k_folds=10,
            random_seed=123,
        )
        assert config.scaler_type == "MinMaxScaler"
        assert config.k_folds == 10

    def test_data_splits(self):
        splits = DataSplits(
            X_train=np.array([[1, 2], [3, 4]]),
            X_test=np.array([[5, 6]]),
            y_train=np.array([0, 1]),
            y_test=np.array([0]),
        )
        assert splits.X_train.shape == (2, 2)
        assert splits.removed_features == []
        assert splits.scaler_params == {}
        assert splits.cv_folds == []


class TestEnsembleModels:
    def test_model_result(self):
        result = ModelResult(
            model="mock_model",
            probabilities=np.array([0.1, 0.9, 0.5]),
            metrics={"f1": 0.85, "auc_roc": 0.92},
            training_time_seconds=45.3,
        )
        assert result.training_time_seconds == 45.3
        assert len(result.probabilities) == 3

    def test_ensemble_prediction(self):
        pred = EnsemblePrediction(
            fraud_score=0.75,
            is_fraud=True,
            model_scores={"xgboost": 0.8, "lightgbm": 0.7, "catboost": 0.75},
            model_version="v1.2.0",
        )
        assert pred.fraud_score == 0.75
        assert pred.is_fraud is True
        assert pred.model_version == "v1.2.0"

    def test_ensemble_config_defaults(self):
        config = EnsembleConfig()
        assert abs(sum(config.weights.values()) - 1.0) < 1e-9
        assert config.decision_threshold == 0.5
        assert "xgboost" in config.weights
        assert "lightgbm" in config.weights
        assert "catboost" in config.weights


class TestExplainabilityModels:
    def test_shap_explanation(self):
        explanation = SHAPExplanation(
            feature_names=["V1", "V14", "V12"],
            shap_values={"V1": 0.3, "V14": 0.25, "V12": -0.2},
            base_value=0.5,
            predicted_score=0.8,
        )
        assert explanation.base_value == 0.5
        assert explanation.shap_values["V1"] == 0.3

    def test_lime_explanation(self):
        explanation = LIMEExplanation(
            feature_contributions=[("V1", 0.3), ("V14", -0.2)],
            fidelity_score=0.45,
            is_low_confidence=True,
            num_perturbation_samples=5000,
        )
        assert explanation.is_low_confidence is True
        assert explanation.fidelity_score == 0.45

    def test_audit_entry(self):
        entry = AuditEntry(
            transaction_id="tx-100",
            timestamp="2024-06-15T10:30:00.123Z",
            model_version="v2.1.0",
            fraud_score=0.92,
            is_fraud=True,
            top_5_shap={"V14": 0.4, "V12": 0.3, "V1": -0.2, "V17": 0.15, "V3": -0.1},
        )
        assert entry.transaction_id == "tx-100"
        assert len(entry.top_5_shap) == 5


class TestMonitoringModels:
    def test_drift_result_no_drift(self):
        result = DriftResult(is_drifted=False)
        assert not result.is_drifted
        assert result.drifted_features == []
        assert result.psi_values == {}

    def test_drift_result_with_drift(self):
        result = DriftResult(
            is_drifted=True,
            drifted_features=["V1", "V14"],
            psi_values={"V1": 0.25, "V14": 0.31},
            ks_results={"V1": (0.15, 0.01), "V14": (0.18, 0.003)},
            detection_timestamp="2024-06-15T12:00:00Z",
        )
        assert len(result.drifted_features) == 2
        assert result.ks_results["V1"] == (0.15, 0.01)

    def test_alert_event(self):
        alert = AlertEvent(
            alert_type="drift",
            drifted_features=["V1"],
            psi_values={"V1": 0.25},
            current_f1=0.88,
            timestamp="2024-06-15T12:00:00Z",
            dashboard_link="http://grafana:3000/d/fraud",
        )
        assert alert.alert_type == "drift"
        assert alert.current_f1 == 0.88

    def test_retrain_result_success(self):
        result = RetrainResult(
            success=True,
            new_model_version="v3.0",
            new_f1=0.93,
            old_f1=0.89,
            promoted=True,
        )
        assert result.promoted is True
        assert result.reason is None

    def test_retrain_result_failure(self):
        result = RetrainResult(
            success=True,
            new_model_version="v3.0",
            new_f1=0.87,
            old_f1=0.89,
            promoted=False,
            reason="New model F1 does not exceed current production F1",
        )
        assert result.promoted is False
        assert "F1" in result.reason


class TestErrorHierarchy:
    def test_all_errors_inherit_from_framework_error(self):
        error_classes = [
            ConfigError,
            IngestionError,
            PreprocessingError,
            EnsembleError,
            ExplainabilityError,
            MLflowError,
            APIError,
            MonitoringError,
            PipelineError,
        ]
        for cls in error_classes:
            assert issubclass(cls, FrameworkError)
            assert issubclass(cls, Exception)

    def test_error_message_and_details(self):
        error = ConfigError(
            "Invalid config",
            details={"section": "ensemble", "key": "learning_rate", "reason": "out of range"},
        )
        assert error.message == "Invalid config"
        assert error.details["section"] == "ensemble"
        assert str(error) == "Invalid config"

    def test_error_default_details(self):
        error = IngestionError("Kafka disconnected")
        assert error.details == {}

    def test_errors_are_catchable_as_exception(self):
        with pytest.raises(Exception):
            raise PipelineError("Pipeline timeout exceeded")

    def test_errors_are_catchable_as_framework_error(self):
        with pytest.raises(FrameworkError):
            raise MonitoringError("Alert delivery failed")

    def test_each_error_is_distinct(self):
        """Ensure different error types are not confused."""
        with pytest.raises(ConfigError):
            raise ConfigError("config issue")

        # A ConfigError should not be caught as an IngestionError
        with pytest.raises(ConfigError):
            try:
                raise ConfigError("config issue")
            except IngestionError:
                pytest.fail("ConfigError should not be caught as IngestionError")
