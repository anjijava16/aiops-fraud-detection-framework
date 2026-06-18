"""Unit tests for CatBoostTrainer."""

import logging

import numpy as np
import pytest

from src.ensemble.catboost_trainer import CatBoostTrainer
from src.models.ensemble import ModelResult
from src.models.errors import EnsembleError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_dataset():
    """Create a small binary classification dataset for testing."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 10

    X = rng.standard_normal((n_samples, n_features))
    # Create linearly separable classes with some noise
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def default_trainer():
    """Create a CatBoostTrainer with default parameters."""
    return CatBoostTrainer(
        learning_rate=0.1,
        depth=4,
        iterations=50,
        auto_class_weights="Balanced",
        logging_interval=10,
    )


# ---------------------------------------------------------------------------
# Initialization and Validation Tests
# ---------------------------------------------------------------------------


class TestCatBoostTrainerInit:
    """Tests for CatBoostTrainer initialization and parameter validation."""

    def test_valid_default_params(self):
        """Trainer initializes with valid default-like params."""
        trainer = CatBoostTrainer(
            learning_rate=0.1, depth=6, iterations=300, auto_class_weights="Balanced"
        )
        assert trainer.learning_rate == 0.1
        assert trainer.depth == 6
        assert trainer.iterations == 300
        assert trainer.auto_class_weights == "Balanced"
        assert trainer.logging_interval == 10

    def test_valid_boundary_low(self):
        """Trainer accepts minimum valid boundary values."""
        trainer = CatBoostTrainer(
            learning_rate=0.001, depth=1, iterations=1, auto_class_weights=None
        )
        assert trainer.learning_rate == 0.001
        assert trainer.depth == 1
        assert trainer.iterations == 1
        assert trainer.auto_class_weights is None

    def test_valid_boundary_high(self):
        """Trainer accepts maximum valid boundary values."""
        trainer = CatBoostTrainer(
            learning_rate=1.0, depth=16, iterations=10000, auto_class_weights="SqrtBalanced"
        )
        assert trainer.learning_rate == 1.0
        assert trainer.depth == 16
        assert trainer.iterations == 10000
        assert trainer.auto_class_weights == "SqrtBalanced"

    def test_invalid_learning_rate_too_low(self):
        """Raises EnsembleError when learning_rate < 0.001."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.0001, depth=6, iterations=100)
        assert "learning_rate" in str(exc_info.value)

    def test_invalid_learning_rate_too_high(self):
        """Raises EnsembleError when learning_rate > 1.0."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=1.5, depth=6, iterations=100)
        assert "learning_rate" in str(exc_info.value)

    def test_invalid_learning_rate_zero(self):
        """Raises EnsembleError when learning_rate is 0."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.0, depth=6, iterations=100)
        assert "learning_rate" in str(exc_info.value)

    def test_invalid_depth_too_low(self):
        """Raises EnsembleError when depth < 1."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.1, depth=0, iterations=100)
        assert "depth" in str(exc_info.value)

    def test_invalid_depth_too_high(self):
        """Raises EnsembleError when depth > 16."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.1, depth=17, iterations=100)
        assert "depth" in str(exc_info.value)

    def test_invalid_iterations_too_low(self):
        """Raises EnsembleError when iterations < 1."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.1, depth=6, iterations=0)
        assert "iterations" in str(exc_info.value)

    def test_invalid_iterations_too_high(self):
        """Raises EnsembleError when iterations > 10000."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.1, depth=6, iterations=10001)
        assert "iterations" in str(exc_info.value)

    def test_invalid_auto_class_weights(self):
        """Raises EnsembleError for unsupported auto_class_weights value."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(
                learning_rate=0.1, depth=6, iterations=100, auto_class_weights="InvalidWeight"
            )
        assert "auto_class_weights" in str(exc_info.value)

    def test_invalid_logging_interval_zero(self):
        """Raises EnsembleError when logging_interval < 1."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.1, depth=6, iterations=100, logging_interval=0)
        assert "logging_interval" in str(exc_info.value)

    def test_invalid_depth_not_integer(self):
        """Raises EnsembleError when depth is not an integer."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.1, depth=6.5, iterations=100)
        assert "depth" in str(exc_info.value)

    def test_invalid_iterations_not_integer(self):
        """Raises EnsembleError when iterations is not an integer."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=0.1, depth=6, iterations=100.5)
        assert "iterations" in str(exc_info.value)

    def test_error_details_contain_parameter_name(self):
        """EnsembleError details include parameter name and value."""
        with pytest.raises(EnsembleError) as exc_info:
            CatBoostTrainer(learning_rate=5.0, depth=6, iterations=100)
        assert exc_info.value.details["parameter"] == "learning_rate"
        assert exc_info.value.details["value"] == 5.0


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------


class TestCatBoostTrainerTrain:
    """Tests for CatBoostTrainer.train() method."""

    def test_train_returns_model_result(self, default_trainer, binary_dataset):
        """train() returns a ModelResult with expected fields."""
        X, y = binary_dataset
        result = default_trainer.train(X, y)

        assert isinstance(result, ModelResult)
        assert result.model is not None
        assert result.training_time_seconds > 0

    def test_train_probabilities_shape(self, default_trainer, binary_dataset):
        """Predicted probabilities have correct shape (n_samples,)."""
        X, y = binary_dataset
        result = default_trainer.train(X, y)

        assert result.probabilities.shape == (X.shape[0],)

    def test_train_probabilities_bounded(self, default_trainer, binary_dataset):
        """All predicted probabilities are in [0.0, 1.0]."""
        X, y = binary_dataset
        result = default_trainer.train(X, y)

        assert np.all(result.probabilities >= 0.0)
        assert np.all(result.probabilities <= 1.0)

    def test_train_metrics_present(self, default_trainer, binary_dataset):
        """Metrics dict contains all expected keys."""
        X, y = binary_dataset
        result = default_trainer.train(X, y)

        expected_keys = {"f1", "auc_roc", "precision", "recall", "log_loss"}
        assert set(result.metrics.keys()) == expected_keys

    def test_train_metrics_valid_ranges(self, default_trainer, binary_dataset):
        """Metric values are within valid ranges."""
        X, y = binary_dataset
        result = default_trainer.train(X, y)

        assert 0.0 <= result.metrics["auc_roc"] <= 1.0
        assert 0.0 <= result.metrics["f1"] <= 1.0
        assert 0.0 <= result.metrics["precision"] <= 1.0
        assert 0.0 <= result.metrics["recall"] <= 1.0
        assert result.metrics["log_loss"] >= 0.0

    def test_train_empty_X_raises(self, default_trainer):
        """Raises EnsembleError when X_train is empty."""
        X = np.empty((0, 10))
        y = np.empty(0)

        with pytest.raises(EnsembleError) as exc_info:
            default_trainer.train(X, y)
        assert "zero samples" in str(exc_info.value).lower()

    def test_train_dimension_mismatch_raises(self, default_trainer):
        """Raises EnsembleError when X and y have different sample counts."""
        X = np.random.randn(100, 10)
        y = np.array([0, 1, 0])  # Mismatched length

        with pytest.raises(EnsembleError) as exc_info:
            default_trainer.train(X, y)
        assert "mismatch" in str(exc_info.value).lower()

    def test_train_1d_X_raises(self, default_trainer):
        """Raises EnsembleError when X_train is 1-dimensional."""
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([0, 1, 0])

        with pytest.raises(EnsembleError):
            default_trainer.train(X, y)

    def test_train_with_none_auto_class_weights(self, binary_dataset):
        """Training succeeds with auto_class_weights=None."""
        X, y = binary_dataset
        trainer = CatBoostTrainer(
            learning_rate=0.1, depth=4, iterations=20, auto_class_weights=None
        )
        result = trainer.train(X, y)
        assert isinstance(result, ModelResult)

    def test_train_with_sqrt_balanced(self, binary_dataset):
        """Training succeeds with auto_class_weights='SqrtBalanced'."""
        X, y = binary_dataset
        trainer = CatBoostTrainer(
            learning_rate=0.1, depth=4, iterations=20, auto_class_weights="SqrtBalanced"
        )
        result = trainer.train(X, y)
        assert isinstance(result, ModelResult)

    def test_train_logs_metrics(self, default_trainer, binary_dataset, caplog):
        """Training logs metrics at configured interval."""
        X, y = binary_dataset
        with caplog.at_level(logging.INFO, logger="src.ensemble.catboost_trainer"):
            default_trainer.train(X, y)

        # Should log iteration-level metrics
        log_messages = [r.message for r in caplog.records]
        iteration_logs = [m for m in log_messages if "iteration" in m.lower()]
        assert len(iteration_logs) > 0


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------


class TestCatBoostTrainerPredictProba:
    """Tests for CatBoostTrainer.predict_proba() method."""

    def test_predict_proba_before_train_raises(self, default_trainer):
        """Raises EnsembleError if predict_proba called before train."""
        X = np.random.randn(10, 5)

        with pytest.raises(EnsembleError) as exc_info:
            default_trainer.predict_proba(X)
        assert "not been trained" in str(exc_info.value).lower()

    def test_predict_proba_shape(self, default_trainer, binary_dataset):
        """predict_proba returns array with shape (n_samples,)."""
        X, y = binary_dataset
        default_trainer.train(X, y)

        proba = default_trainer.predict_proba(X)
        assert proba.shape == (X.shape[0],)

    def test_predict_proba_bounded(self, default_trainer, binary_dataset):
        """All predicted probabilities are in [0.0, 1.0]."""
        X, y = binary_dataset
        default_trainer.train(X, y)

        proba = default_trainer.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_single_sample(self, default_trainer, binary_dataset):
        """predict_proba works for a single sample (1D input)."""
        X, y = binary_dataset
        default_trainer.train(X, y)

        single = X[0]  # 1D array
        proba = default_trainer.predict_proba(single)
        assert proba.shape == (1,)
        assert 0.0 <= proba[0] <= 1.0

    def test_predict_proba_new_data(self, default_trainer, binary_dataset):
        """predict_proba works on unseen data."""
        X, y = binary_dataset
        default_trainer.train(X, y)

        X_new = np.random.default_rng(99).standard_normal((50, X.shape[1]))
        proba = default_trainer.predict_proba(X_new)
        assert proba.shape == (50,)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
