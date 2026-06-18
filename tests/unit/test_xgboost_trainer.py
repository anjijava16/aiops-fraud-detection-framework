"""Unit tests for XGBoostTrainer."""

import numpy as np
import pytest

from src.ensemble.xgboost_trainer import XGBoostTrainer
from src.models.ensemble import ModelResult
from src.models.errors import EnsembleError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """Create a small binary classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    y = np.concatenate([np.zeros(80), np.ones(20)])
    return X, y


@pytest.fixture
def trainer():
    """Return a default XGBoostTrainer."""
    return XGBoostTrainer(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=20,
        scale_pos_weight=1.0,
        logging_interval=5,
    )


# ---------------------------------------------------------------------------
# Hyperparameter Validation Tests
# ---------------------------------------------------------------------------


class TestHyperparameterValidation:
    """Tests for hyperparameter range validation."""

    def test_valid_hyperparameters(self):
        """Default valid hyperparameters should not raise."""
        trainer = XGBoostTrainer(
            learning_rate=0.1, max_depth=6, n_estimators=300, scale_pos_weight=1.0
        )
        assert trainer.learning_rate == 0.1
        assert trainer.max_depth == 6
        assert trainer.n_estimators == 300
        assert trainer.scale_pos_weight == 1.0

    def test_learning_rate_boundary_upper(self):
        """learning_rate=1.0 is the upper bound and should be valid."""
        trainer = XGBoostTrainer(learning_rate=1.0, max_depth=6, n_estimators=100, scale_pos_weight=1.0)
        assert trainer.learning_rate == 1.0

    def test_learning_rate_zero_raises(self):
        """learning_rate=0.0 is invalid (exclusive lower bound)."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.0, max_depth=6, n_estimators=100, scale_pos_weight=1.0)
        assert "learning_rate" in str(exc_info.value)
        assert exc_info.value.details["param"] == "learning_rate"
        assert exc_info.value.details["valid_range"] == "(0.0, 1.0]"

    def test_learning_rate_negative_raises(self):
        """Negative learning_rate is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=-0.5, max_depth=6, n_estimators=100, scale_pos_weight=1.0)
        assert exc_info.value.details["param"] == "learning_rate"

    def test_learning_rate_above_one_raises(self):
        """learning_rate > 1.0 is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=1.1, max_depth=6, n_estimators=100, scale_pos_weight=1.0)
        assert exc_info.value.details["param"] == "learning_rate"

    def test_max_depth_boundary_lower(self):
        """max_depth=1 should be valid."""
        trainer = XGBoostTrainer(learning_rate=0.1, max_depth=1, n_estimators=100, scale_pos_weight=1.0)
        assert trainer.max_depth == 1

    def test_max_depth_boundary_upper(self):
        """max_depth=50 should be valid."""
        trainer = XGBoostTrainer(learning_rate=0.1, max_depth=50, n_estimators=100, scale_pos_weight=1.0)
        assert trainer.max_depth == 50

    def test_max_depth_zero_raises(self):
        """max_depth=0 is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.1, max_depth=0, n_estimators=100, scale_pos_weight=1.0)
        assert exc_info.value.details["param"] == "max_depth"
        assert exc_info.value.details["valid_range"] == "[1, 50]"

    def test_max_depth_above_50_raises(self):
        """max_depth=51 is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.1, max_depth=51, n_estimators=100, scale_pos_weight=1.0)
        assert exc_info.value.details["param"] == "max_depth"

    def test_n_estimators_boundary_lower(self):
        """n_estimators=1 should be valid."""
        trainer = XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=1, scale_pos_weight=1.0)
        assert trainer.n_estimators == 1

    def test_n_estimators_boundary_upper(self):
        """n_estimators=10000 should be valid."""
        trainer = XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=10000, scale_pos_weight=1.0)
        assert trainer.n_estimators == 10000

    def test_n_estimators_zero_raises(self):
        """n_estimators=0 is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=0, scale_pos_weight=1.0)
        assert exc_info.value.details["param"] == "n_estimators"
        assert exc_info.value.details["valid_range"] == "[1, 10000]"

    def test_n_estimators_above_10000_raises(self):
        """n_estimators=10001 is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=10001, scale_pos_weight=1.0)
        assert exc_info.value.details["param"] == "n_estimators"

    def test_scale_pos_weight_zero_raises(self):
        """scale_pos_weight=0.0 is invalid (exclusive lower bound)."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=100, scale_pos_weight=0.0)
        assert exc_info.value.details["param"] == "scale_pos_weight"
        assert exc_info.value.details["valid_range"] == "(0.0, 10000.0]"

    def test_scale_pos_weight_negative_raises(self):
        """Negative scale_pos_weight is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=100, scale_pos_weight=-1.0)
        assert exc_info.value.details["param"] == "scale_pos_weight"

    def test_scale_pos_weight_boundary_upper(self):
        """scale_pos_weight=10000.0 should be valid."""
        trainer = XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=100, scale_pos_weight=10000.0)
        assert trainer.scale_pos_weight == 10000.0

    def test_scale_pos_weight_above_10000_raises(self):
        """scale_pos_weight > 10000.0 is invalid."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=0.1, max_depth=6, n_estimators=100, scale_pos_weight=10001.0)
        assert exc_info.value.details["param"] == "scale_pos_weight"

    def test_error_details_contain_value(self):
        """EnsembleError details should include the invalid value."""
        with pytest.raises(EnsembleError) as exc_info:
            XGBoostTrainer(learning_rate=2.5, max_depth=6, n_estimators=100, scale_pos_weight=1.0)
        assert exc_info.value.details["value"] == 2.5


# ---------------------------------------------------------------------------
# Data Validation Tests
# ---------------------------------------------------------------------------


class TestDataValidation:
    """Tests for training data validation."""

    def test_zero_samples_raises(self, trainer):
        """Training with zero samples should raise EnsembleError."""
        X = np.empty((0, 10))
        y = np.empty((0,))
        with pytest.raises(EnsembleError) as exc_info:
            trainer.train(X, y)
        assert "zero samples" in str(exc_info.value).lower()

    def test_mismatched_dimensions_raises(self, trainer):
        """Mismatched X and y lengths should raise EnsembleError."""
        X = np.random.default_rng(0).standard_normal((100, 10))
        y = np.zeros(50)
        with pytest.raises(EnsembleError) as exc_info:
            trainer.train(X, y)
        assert "mismatched" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------


class TestTraining:
    """Tests for the training workflow."""

    def test_train_returns_model_result(self, trainer, sample_data):
        """train() should return a ModelResult instance."""
        X, y = sample_data
        result = trainer.train(X, y)
        assert isinstance(result, ModelResult)

    def test_train_model_is_xgboost(self, trainer, sample_data):
        """The model in ModelResult should be an XGBClassifier."""
        X, y = sample_data
        result = trainer.train(X, y)
        import xgboost as xgb_module
        assert isinstance(result.model, xgb_module.XGBClassifier)

    def test_probabilities_shape(self, trainer, sample_data):
        """Probabilities should have shape (n_samples,)."""
        X, y = sample_data
        result = trainer.train(X, y)
        assert result.probabilities.shape == (X.shape[0],)

    def test_probabilities_range(self, trainer, sample_data):
        """All probabilities should be in [0.0, 1.0]."""
        X, y = sample_data
        result = trainer.train(X, y)
        assert np.all(result.probabilities >= 0.0)
        assert np.all(result.probabilities <= 1.0)

    def test_metrics_keys_present(self, trainer, sample_data):
        """ModelResult metrics should contain required keys."""
        X, y = sample_data
        result = trainer.train(X, y)
        required_keys = {"log_loss", "auc_roc", "f1", "precision", "recall"}
        assert required_keys <= set(result.metrics.keys())

    def test_training_time_positive(self, trainer, sample_data):
        """Training time should be a positive value."""
        X, y = sample_data
        result = trainer.train(X, y)
        assert result.training_time_seconds > 0.0

    def test_metrics_values_reasonable(self, trainer, sample_data):
        """Metric values should be in valid ranges."""
        X, y = sample_data
        result = trainer.train(X, y)
        assert 0.0 <= result.metrics["auc_roc"] <= 1.0
        assert result.metrics["log_loss"] >= 0.0
        assert 0.0 <= result.metrics["f1"] <= 1.0
        assert 0.0 <= result.metrics["precision"] <= 1.0
        assert 0.0 <= result.metrics["recall"] <= 1.0


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------


class TestPredictProba:
    """Tests for the predict_proba method."""

    def test_predict_proba_before_training_raises(self, trainer):
        """predict_proba should raise if model not trained."""
        X = np.random.default_rng(0).standard_normal((10, 10))
        with pytest.raises(EnsembleError) as exc_info:
            trainer.predict_proba(X)
        assert "not been trained" in str(exc_info.value).lower()

    def test_predict_proba_after_training(self, trainer, sample_data):
        """predict_proba should return valid probabilities after training."""
        X, y = sample_data
        trainer.train(X, y)
        probs = trainer.predict_proba(X)
        assert probs.shape == (X.shape[0],)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_proba_new_data(self, trainer, sample_data):
        """predict_proba should work on new unseen data."""
        X, y = sample_data
        trainer.train(X, y)
        X_new = np.random.default_rng(99).standard_normal((20, 10))
        probs = trainer.predict_proba(X_new)
        assert probs.shape == (20,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# Logging Tests
# ---------------------------------------------------------------------------


class TestLogging:
    """Tests for metrics logging at configurable intervals."""

    def test_logging_interval_stored(self):
        """logging_interval should be stored as attribute."""
        trainer = XGBoostTrainer(
            learning_rate=0.1, max_depth=3, n_estimators=50,
            scale_pos_weight=1.0, logging_interval=5,
        )
        assert trainer.logging_interval == 5

    def test_default_logging_interval(self):
        """Default logging_interval should be 10."""
        trainer = XGBoostTrainer(
            learning_rate=0.1, max_depth=3, n_estimators=50, scale_pos_weight=1.0,
        )
        assert trainer.logging_interval == 10

    def test_training_logs_metrics(self, sample_data, caplog):
        """Training should log metrics at the configured interval."""
        import logging

        trainer = XGBoostTrainer(
            learning_rate=0.3, max_depth=3, n_estimators=20,
            scale_pos_weight=1.0, logging_interval=10,
        )
        X, y = sample_data
        with caplog.at_level(logging.INFO, logger="src.ensemble.xgboost_trainer"):
            trainer.train(X, y)
        # At least one log message with "round" should be present
        log_messages = [r.message for r in caplog.records]
        round_logs = [m for m in log_messages if "round" in m.lower()]
        assert len(round_logs) >= 1
