"""Unit tests for the LightGBM trainer module."""

import numpy as np
import pytest

from src.ensemble.lightgbm_trainer import LightGBMTrainer
from src.models.ensemble import ModelResult
from src.models.errors import EnsembleError


class TestLightGBMTrainerInit:
    """Tests for LightGBMTrainer initialization and hyperparameter validation."""

    def test_default_initialization(self):
        """Trainer initializes with default hyperparameters."""
        trainer = LightGBMTrainer()
        assert trainer.learning_rate == 0.1
        assert trainer.num_leaves == 31
        assert trainer.n_estimators == 300
        assert trainer.is_unbalance is True
        assert trainer.logging_interval == 10

    def test_custom_initialization(self):
        """Trainer initializes with custom hyperparameters."""
        trainer = LightGBMTrainer(
            learning_rate=0.05,
            num_leaves=64,
            n_estimators=100,
            is_unbalance=False,
            logging_interval=5,
        )
        assert trainer.learning_rate == 0.05
        assert trainer.num_leaves == 64
        assert trainer.n_estimators == 100
        assert trainer.is_unbalance is False
        assert trainer.logging_interval == 5

    def test_learning_rate_boundary_max(self):
        """learning_rate=1.0 is valid (upper boundary inclusive)."""
        trainer = LightGBMTrainer(learning_rate=1.0)
        assert trainer.learning_rate == 1.0

    def test_learning_rate_boundary_near_zero(self):
        """learning_rate just above 0.0 is valid."""
        trainer = LightGBMTrainer(learning_rate=0.001)
        assert trainer.learning_rate == 0.001

    def test_num_leaves_boundary_min(self):
        """num_leaves=2 is valid (lower boundary)."""
        trainer = LightGBMTrainer(num_leaves=2)
        assert trainer.num_leaves == 2

    def test_num_leaves_boundary_max(self):
        """num_leaves=131072 is valid (upper boundary)."""
        trainer = LightGBMTrainer(num_leaves=131072)
        assert trainer.num_leaves == 131072

    def test_n_estimators_boundary_min(self):
        """n_estimators=1 is valid (lower boundary)."""
        trainer = LightGBMTrainer(n_estimators=1)
        assert trainer.n_estimators == 1

    def test_n_estimators_boundary_max(self):
        """n_estimators=10000 is valid (upper boundary)."""
        trainer = LightGBMTrainer(n_estimators=10000)
        assert trainer.n_estimators == 10000


class TestLightGBMTrainerHyperparameterValidation:
    """Tests for hyperparameter validation errors."""

    def test_learning_rate_zero_raises_error(self):
        """learning_rate=0.0 raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(learning_rate=0.0)
        assert exc_info.value.details["param_name"] == "learning_rate"
        assert exc_info.value.details["provided_value"] == 0.0
        assert "(0.0, 1.0]" in exc_info.value.details["valid_range"]

    def test_learning_rate_negative_raises_error(self):
        """Negative learning_rate raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(learning_rate=-0.1)
        assert exc_info.value.details["param_name"] == "learning_rate"

    def test_learning_rate_above_one_raises_error(self):
        """learning_rate > 1.0 raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(learning_rate=1.5)
        assert exc_info.value.details["param_name"] == "learning_rate"
        assert exc_info.value.details["provided_value"] == 1.5

    def test_num_leaves_below_min_raises_error(self):
        """num_leaves < 2 raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(num_leaves=1)
        assert exc_info.value.details["param_name"] == "num_leaves"
        assert exc_info.value.details["provided_value"] == 1
        assert "[2, 131072]" in exc_info.value.details["valid_range"]

    def test_num_leaves_above_max_raises_error(self):
        """num_leaves > 131072 raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(num_leaves=200000)
        assert exc_info.value.details["param_name"] == "num_leaves"

    def test_n_estimators_zero_raises_error(self):
        """n_estimators=0 raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(n_estimators=0)
        assert exc_info.value.details["param_name"] == "n_estimators"
        assert exc_info.value.details["provided_value"] == 0
        assert "[1, 10000]" in exc_info.value.details["valid_range"]

    def test_n_estimators_above_max_raises_error(self):
        """n_estimators > 10000 raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(n_estimators=10001)
        assert exc_info.value.details["param_name"] == "n_estimators"

    def test_is_unbalance_non_bool_raises_error(self):
        """Non-boolean is_unbalance raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(is_unbalance=1)  # type: ignore
        assert exc_info.value.details["param_name"] == "is_unbalance"
        assert exc_info.value.details["provided_value"] == 1

    def test_is_unbalance_string_raises_error(self):
        """String is_unbalance raises EnsembleError."""
        with pytest.raises(EnsembleError) as exc_info:
            LightGBMTrainer(is_unbalance="true")  # type: ignore
        assert exc_info.value.details["param_name"] == "is_unbalance"


class TestLightGBMTrainerTrain:
    """Tests for the train() method."""

    @pytest.fixture
    def binary_dataset(self):
        """Create a simple binary classification dataset."""
        rng = np.random.default_rng(42)
        n_samples = 200
        n_features = 10

        # Create separable classes
        X_class0 = rng.normal(loc=0.0, scale=1.0, size=(n_samples // 2, n_features))
        X_class1 = rng.normal(loc=2.0, scale=1.0, size=(n_samples // 2, n_features))
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.float64)
        return X, y

    @pytest.fixture
    def trainer(self):
        """Create a trainer with fast settings for testing."""
        return LightGBMTrainer(
            learning_rate=0.1,
            num_leaves=31,
            n_estimators=10,
            is_unbalance=True,
            logging_interval=5,
        )

    def test_train_returns_model_result(self, trainer, binary_dataset):
        """train() returns a ModelResult instance."""
        X, y = binary_dataset
        result = trainer.train(X, y)
        assert isinstance(result, ModelResult)

    def test_train_model_is_not_none(self, trainer, binary_dataset):
        """ModelResult.model is a trained LightGBM model."""
        X, y = binary_dataset
        result = trainer.train(X, y)
        assert result.model is not None

    def test_train_probabilities_shape(self, trainer, binary_dataset):
        """Probabilities shape matches number of training samples."""
        X, y = binary_dataset
        result = trainer.train(X, y)
        assert result.probabilities.shape == (X.shape[0],)

    def test_train_probabilities_in_valid_range(self, trainer, binary_dataset):
        """All probabilities are in [0.0, 1.0]."""
        X, y = binary_dataset
        result = trainer.train(X, y)
        assert np.all(result.probabilities >= 0.0)
        assert np.all(result.probabilities <= 1.0)

    def test_train_metrics_contain_required_keys(self, trainer, binary_dataset):
        """Metrics dict contains log_loss, auc_roc, f1, precision, recall."""
        X, y = binary_dataset
        result = trainer.train(X, y)
        required_keys = {"log_loss", "auc_roc", "f1", "precision", "recall"}
        assert required_keys.issubset(result.metrics.keys())

    def test_train_metrics_values_are_valid(self, trainer, binary_dataset):
        """Metric values are non-negative and bounded."""
        X, y = binary_dataset
        result = trainer.train(X, y)
        assert 0.0 <= result.metrics["auc_roc"] <= 1.0
        assert 0.0 <= result.metrics["f1"] <= 1.0
        assert 0.0 <= result.metrics["precision"] <= 1.0
        assert 0.0 <= result.metrics["recall"] <= 1.0
        assert result.metrics["log_loss"] >= 0.0

    def test_train_training_time_positive(self, trainer, binary_dataset):
        """Training time is a positive float."""
        X, y = binary_dataset
        result = trainer.train(X, y)
        assert result.training_time_seconds > 0.0

    def test_train_with_is_unbalance_false(self, binary_dataset):
        """Training works with is_unbalance=False."""
        X, y = binary_dataset
        trainer = LightGBMTrainer(n_estimators=5, is_unbalance=False)
        result = trainer.train(X, y)
        assert isinstance(result, ModelResult)


class TestLightGBMTrainerDataValidation:
    """Tests for data format error handling."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer with minimal settings."""
        return LightGBMTrainer(n_estimators=5)

    def test_zero_samples_raises_error(self, trainer):
        """Empty arrays raise EnsembleError."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        with pytest.raises(EnsembleError) as exc_info:
            trainer.train(X, y)
        assert exc_info.value.details["error_type"] == "zero_samples"

    def test_mismatched_dimensions_raises_error(self, trainer):
        """Mismatched X and y dimensions raise EnsembleError."""
        X = np.random.randn(10, 5)
        y = np.array([0, 1, 0, 1, 0])  # Only 5 labels for 10 samples
        with pytest.raises(EnsembleError) as exc_info:
            trainer.train(X, y)
        assert exc_info.value.details["error_type"] == "dimension_mismatch"

    def test_nan_in_features_raises_error(self, trainer):
        """NaN values in X raise EnsembleError."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        with pytest.raises(EnsembleError) as exc_info:
            trainer.train(X, y)
        assert exc_info.value.details["error_type"] == "nan_values"
        assert 1 in exc_info.value.details["nan_columns"]

    def test_nan_in_labels_raises_error(self, trainer):
        """NaN values in y raise EnsembleError."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0.0, np.nan, 1.0])
        with pytest.raises(EnsembleError) as exc_info:
            trainer.train(X, y)
        assert exc_info.value.details["error_type"] == "nan_values"

    def test_non_numeric_features_raises_error(self, trainer):
        """Non-numeric X raises EnsembleError."""
        X = np.array([["a", "b"], ["c", "d"]])
        y = np.array([0, 1])
        with pytest.raises(EnsembleError) as exc_info:
            trainer.train(X, y)
        assert exc_info.value.details["error_type"] == "non_numeric"


class TestLightGBMTrainerPredictProba:
    """Tests for the predict_proba() method."""

    @pytest.fixture
    def trained_trainer(self):
        """Create a trained trainer."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 5
        X = rng.normal(size=(n_samples, n_features))
        y = (X[:, 0] > 0).astype(int)

        trainer = LightGBMTrainer(n_estimators=10)
        trainer.train(X, y)
        return trainer, n_features

    def test_predict_proba_returns_array(self, trained_trainer):
        """predict_proba returns numpy array."""
        trainer, n_features = trained_trainer
        X_test = np.random.randn(20, n_features)
        result = trainer.predict_proba(X_test)
        assert isinstance(result, np.ndarray)

    def test_predict_proba_shape(self, trained_trainer):
        """predict_proba output shape matches input samples."""
        trainer, n_features = trained_trainer
        X_test = np.random.randn(20, n_features)
        result = trainer.predict_proba(X_test)
        assert result.shape == (20,)

    def test_predict_proba_values_in_range(self, trained_trainer):
        """All predicted probabilities are in [0.0, 1.0]."""
        trainer, n_features = trained_trainer
        X_test = np.random.randn(50, n_features)
        result = trainer.predict_proba(X_test)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_predict_proba_untrained_raises_error(self):
        """predict_proba on untrained model raises EnsembleError."""
        trainer = LightGBMTrainer(n_estimators=5)
        X = np.random.randn(10, 5)
        with pytest.raises(EnsembleError) as exc_info:
            trainer.predict_proba(X)
        assert exc_info.value.details["error_type"] == "model_not_trained"

    def test_predict_proba_empty_input_raises_error(self, trained_trainer):
        """predict_proba with zero samples raises EnsembleError."""
        trainer, n_features = trained_trainer
        X = np.array([]).reshape(0, n_features)
        with pytest.raises(EnsembleError) as exc_info:
            trainer.predict_proba(X)
        assert exc_info.value.details["error_type"] == "zero_samples"

    def test_predict_proba_nan_raises_error(self, trained_trainer):
        """predict_proba with NaN input raises EnsembleError."""
        trainer, n_features = trained_trainer
        X = np.array([[1.0] * n_features, [np.nan] + [1.0] * (n_features - 1)])
        with pytest.raises(EnsembleError) as exc_info:
            trainer.predict_proba(X)
        assert exc_info.value.details["error_type"] == "nan_values"

    def test_predict_proba_non_numeric_raises_error(self, trained_trainer):
        """predict_proba with non-numeric input raises EnsembleError."""
        trainer, _ = trained_trainer
        X = np.array([["a", "b", "c", "d", "e"]])
        with pytest.raises(EnsembleError) as exc_info:
            trainer.predict_proba(X)
        assert exc_info.value.details["error_type"] == "non_numeric"
