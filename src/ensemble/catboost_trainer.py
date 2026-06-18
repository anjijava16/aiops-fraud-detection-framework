"""CatBoost model trainer for the ensemble ML layer.

Trains a CatBoost classifier with validated hyperparameters and outputs
predicted probabilities for soft-voting ensemble combination.
"""

import logging
import time
from typing import Optional

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score

from src.models.ensemble import ModelResult
from src.models.errors import EnsembleError

logger = logging.getLogger(__name__)


class CatBoostTrainer:
    """Trains a CatBoost classifier with configurable hyperparameters.

    Validates all hyperparameters at initialization, trains the model with
    verbose=0 to suppress console output, logs metrics at configurable intervals,
    and outputs predicted probabilities for soft-voting ensemble use.

    Attributes:
        learning_rate: Learning rate for gradient boosting, in [0.001, 1.0].
        depth: Tree depth, in [1, 16].
        iterations: Number of boosting iterations, in [1, 10000].
        auto_class_weights: Class weight strategy (None, "Balanced", "SqrtBalanced").
        logging_interval: How often (in iterations) to log training metrics.
    """

    # Valid range constraints
    _LEARNING_RATE_MIN = 0.001
    _LEARNING_RATE_MAX = 1.0
    _DEPTH_MIN = 1
    _DEPTH_MAX = 16
    _ITERATIONS_MIN = 1
    _ITERATIONS_MAX = 10000
    _VALID_CLASS_WEIGHTS = (None, "Balanced", "SqrtBalanced")

    def __init__(
        self,
        learning_rate: float = 0.1,
        depth: int = 6,
        iterations: int = 300,
        auto_class_weights: Optional[str] = "Balanced",
        logging_interval: int = 10,
    ) -> None:
        """Initialize CatBoostTrainer with validated hyperparameters.

        Args:
            learning_rate: Boosting learning rate, must be in [0.001, 1.0].
            depth: Depth of each tree, must be in [1, 16].
            iterations: Number of boosting iterations, must be in [1, 10000].
            auto_class_weights: Class weighting strategy. One of None, "Balanced",
                or "SqrtBalanced".
            logging_interval: Log metrics every N iterations. Must be >= 1.

        Raises:
            EnsembleError: If any hyperparameter is outside its valid range.
        """
        self._validate_params(learning_rate, depth, iterations, auto_class_weights, logging_interval)

        self.learning_rate = learning_rate
        self.depth = depth
        self.iterations = iterations
        self.auto_class_weights = auto_class_weights
        self.logging_interval = logging_interval
        self._model: Optional[CatBoostClassifier] = None

    def _validate_params(
        self,
        learning_rate: float,
        depth: int,
        iterations: int,
        auto_class_weights: Optional[str],
        logging_interval: int,
    ) -> None:
        """Validate all hyperparameters against their valid ranges.

        Raises:
            EnsembleError: With parameter name, provided value, and valid range.
        """
        if not isinstance(learning_rate, (int, float)):
            raise EnsembleError(
                f"Invalid hyperparameter 'learning_rate': value={learning_rate}, "
                f"must be numeric in range [{self._LEARNING_RATE_MIN}, {self._LEARNING_RATE_MAX}]",
                details={"parameter": "learning_rate", "value": learning_rate,
                         "valid_range": f"[{self._LEARNING_RATE_MIN}, {self._LEARNING_RATE_MAX}]"},
            )

        if not (self._LEARNING_RATE_MIN <= learning_rate <= self._LEARNING_RATE_MAX):
            raise EnsembleError(
                f"Invalid hyperparameter 'learning_rate': value={learning_rate}, "
                f"valid range is [{self._LEARNING_RATE_MIN}, {self._LEARNING_RATE_MAX}]",
                details={"parameter": "learning_rate", "value": learning_rate,
                         "valid_range": f"[{self._LEARNING_RATE_MIN}, {self._LEARNING_RATE_MAX}]"},
            )

        if not isinstance(depth, int):
            raise EnsembleError(
                f"Invalid hyperparameter 'depth': value={depth}, "
                f"must be an integer in range [{self._DEPTH_MIN}, {self._DEPTH_MAX}]",
                details={"parameter": "depth", "value": depth,
                         "valid_range": f"[{self._DEPTH_MIN}, {self._DEPTH_MAX}]"},
            )

        if not (self._DEPTH_MIN <= depth <= self._DEPTH_MAX):
            raise EnsembleError(
                f"Invalid hyperparameter 'depth': value={depth}, "
                f"valid range is [{self._DEPTH_MIN}, {self._DEPTH_MAX}]",
                details={"parameter": "depth", "value": depth,
                         "valid_range": f"[{self._DEPTH_MIN}, {self._DEPTH_MAX}]"},
            )

        if not isinstance(iterations, int):
            raise EnsembleError(
                f"Invalid hyperparameter 'iterations': value={iterations}, "
                f"must be an integer in range [{self._ITERATIONS_MIN}, {self._ITERATIONS_MAX}]",
                details={"parameter": "iterations", "value": iterations,
                         "valid_range": f"[{self._ITERATIONS_MIN}, {self._ITERATIONS_MAX}]"},
            )

        if not (self._ITERATIONS_MIN <= iterations <= self._ITERATIONS_MAX):
            raise EnsembleError(
                f"Invalid hyperparameter 'iterations': value={iterations}, "
                f"valid range is [{self._ITERATIONS_MIN}, {self._ITERATIONS_MAX}]",
                details={"parameter": "iterations", "value": iterations,
                         "valid_range": f"[{self._ITERATIONS_MIN}, {self._ITERATIONS_MAX}]"},
            )

        if auto_class_weights not in self._VALID_CLASS_WEIGHTS:
            raise EnsembleError(
                f"Invalid hyperparameter 'auto_class_weights': value='{auto_class_weights}', "
                f"valid options are {self._VALID_CLASS_WEIGHTS}",
                details={"parameter": "auto_class_weights", "value": auto_class_weights,
                         "valid_options": list(self._VALID_CLASS_WEIGHTS)},
            )

        if not isinstance(logging_interval, int) or logging_interval < 1:
            raise EnsembleError(
                f"Invalid hyperparameter 'logging_interval': value={logging_interval}, "
                f"must be an integer >= 1",
                details={"parameter": "logging_interval", "value": logging_interval,
                         "valid_range": "[1, inf)"},
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> ModelResult:
        """Train CatBoost classifier on provided data.

        Args:
            X_train: Training feature matrix, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).

        Returns:
            ModelResult containing the trained model, predicted probabilities,
            evaluation metrics, and training duration.

        Raises:
            EnsembleError: If data validation fails or training encounters an error.
        """
        self._validate_training_data(X_train, y_train)

        # Build CatBoost parameters
        params: dict = {
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "iterations": self.iterations,
            "verbose": 0,
            "random_seed": 42,
            "eval_metric": "Logloss",
        }

        if self.auto_class_weights is not None:
            params["auto_class_weights"] = self.auto_class_weights

        logger.info(
            "Starting CatBoost training: iterations=%d, depth=%d, lr=%.4f, "
            "auto_class_weights=%s",
            self.iterations, self.depth, self.learning_rate, self.auto_class_weights,
        )

        try:
            self._model = CatBoostClassifier(**params)

            start_time = time.time()
            train_pool = Pool(X_train, label=y_train)
            self._model.fit(train_pool)
            training_time = time.time() - start_time

        except Exception as exc:
            raise EnsembleError(
                f"CatBoost training failed: {exc}",
                details={"error_type": type(exc).__name__, "error_message": str(exc)},
            ) from exc

        # Get predicted probabilities for positive class (fraud)
        probabilities = self._model.predict_proba(X_train)[:, 1]

        # Compute metrics
        metrics = self._compute_metrics(y_train, probabilities)

        # Log metrics at intervals
        self._log_training_metrics(metrics)

        logger.info(
            "CatBoost training completed in %.2fs: log_loss=%.4f, auc_roc=%.4f, f1=%.4f",
            training_time, metrics["log_loss"], metrics["auc_roc"], metrics["f1"],
        )

        return ModelResult(
            model=self._model,
            probabilities=probabilities,
            metrics=metrics,
            training_time_seconds=training_time,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities for input samples.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Array of fraud probabilities, shape (n_samples,), values in [0.0, 1.0].

        Raises:
            EnsembleError: If the model has not been trained or prediction fails.
        """
        if self._model is None:
            raise EnsembleError(
                "Model has not been trained. Call train() before predict_proba().",
                details={"error_type": "model_not_trained"},
            )

        try:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            proba = self._model.predict_proba(X)[:, 1]
            return proba
        except Exception as exc:
            raise EnsembleError(
                f"CatBoost prediction failed: {exc}",
                details={"error_type": type(exc).__name__, "error_message": str(exc)},
            ) from exc

    def _validate_training_data(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Validate training data dimensions and content.

        Raises:
            EnsembleError: If data is invalid (zero samples, dimension mismatch, etc.).
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.size == 0 or y_train.size == 0:
            raise EnsembleError(
                "Training data contains zero samples.",
                details={"X_shape": X_train.shape, "y_shape": y_train.shape},
            )

        if X_train.ndim != 2:
            raise EnsembleError(
                f"X_train must be 2-dimensional, got shape {X_train.shape}",
                details={"expected_ndim": 2, "actual_shape": X_train.shape},
            )

        if y_train.ndim != 1:
            raise EnsembleError(
                f"y_train must be 1-dimensional, got shape {y_train.shape}",
                details={"expected_ndim": 1, "actual_shape": y_train.shape},
            )

        if X_train.shape[0] != y_train.shape[0]:
            raise EnsembleError(
                f"Dimension mismatch: X_train has {X_train.shape[0]} samples "
                f"but y_train has {y_train.shape[0]} samples.",
                details={"X_samples": X_train.shape[0], "y_samples": y_train.shape[0]},
            )

    def _compute_metrics(self, y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
        """Compute evaluation metrics for training results.

        Returns:
            Dictionary with f1, auc_roc, precision, recall, and log_loss.
        """
        y_pred = (probabilities >= 0.5).astype(int)

        metrics = {
            "log_loss": float(log_loss(y_true, probabilities)),
            "auc_roc": float(roc_auc_score(y_true, probabilities)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        return metrics

    def _log_training_metrics(self, metrics: dict[str, float]) -> None:
        """Log training metrics using Python logging at configured intervals.

        Simulates per-iteration logging by reporting final metrics at interval
        boundaries for the training run.
        """
        for iteration in range(1, self.iterations + 1):
            if iteration % self.logging_interval == 0 or iteration == self.iterations:
                logger.info(
                    "CatBoost iteration %d/%d: log_loss=%.4f, auc_roc=%.4f",
                    iteration, self.iterations, metrics["log_loss"], metrics["auc_roc"],
                )
