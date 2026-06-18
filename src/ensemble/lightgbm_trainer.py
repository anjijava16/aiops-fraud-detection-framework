"""LightGBM trainer with configurable hyperparameters for ensemble fraud detection.

Trains a LightGBM classifier with validated hyperparameters and outputs
predicted probabilities for use in the soft-voting ensemble.
"""

import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score

from src.models.ensemble import ModelResult
from src.models.errors import EnsembleError

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """Trains a LightGBM classifier with validated hyperparameters.

    Attributes:
        learning_rate: Boosting learning rate, must be in (0.0, 1.0].
        num_leaves: Maximum number of leaves per tree, must be in [2, 131072].
        n_estimators: Number of boosting rounds, must be in [1, 10000].
        is_unbalance: Whether to use LightGBM's is_unbalance for class imbalance.
        logging_interval: Log training metrics every N rounds (default 10).
    """

    # Valid hyperparameter ranges
    _PARAM_RANGES = {
        "learning_rate": {"min": 0.0, "max": 1.0, "min_exclusive": True, "max_exclusive": False},
        "num_leaves": {"min": 2, "max": 131072, "min_exclusive": False, "max_exclusive": False},
        "n_estimators": {"min": 1, "max": 10000, "min_exclusive": False, "max_exclusive": False},
    }

    def __init__(
        self,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        n_estimators: int = 300,
        is_unbalance: bool = True,
        logging_interval: int = 10,
    ) -> None:
        """Initialize LightGBMTrainer with validated hyperparameters.

        Args:
            learning_rate: Boosting learning rate in (0.0, 1.0].
            num_leaves: Max leaves per tree in [2, 131072].
            n_estimators: Number of boosting rounds in [1, 10000].
            is_unbalance: Use LightGBM is_unbalance for class weight handling.
            logging_interval: Log metrics every N rounds. Default 10.

        Raises:
            EnsembleError: If any hyperparameter is outside its valid range.
        """
        self._validate_hyperparameters(learning_rate, num_leaves, n_estimators, is_unbalance)

        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.is_unbalance = is_unbalance
        self.logging_interval = logging_interval
        self._model: Any = None

    def _validate_hyperparameters(
        self,
        learning_rate: float,
        num_leaves: int,
        n_estimators: int,
        is_unbalance: Any,
    ) -> None:
        """Validate all hyperparameters against defined ranges.

        Raises:
            EnsembleError: With parameter name, provided value, and valid range.
        """
        # Validate learning_rate: (0.0, 1.0]
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0.0 or learning_rate > 1.0:
            raise EnsembleError(
                f"Hyperparameter 'learning_rate' has invalid value {learning_rate}. "
                f"Valid range: (0.0, 1.0].",
                details={
                    "param_name": "learning_rate",
                    "provided_value": learning_rate,
                    "valid_range": "(0.0, 1.0]",
                },
            )

        # Validate num_leaves: [2, 131072]
        if not isinstance(num_leaves, int) or num_leaves < 2 or num_leaves > 131072:
            raise EnsembleError(
                f"Hyperparameter 'num_leaves' has invalid value {num_leaves}. "
                f"Valid range: [2, 131072].",
                details={
                    "param_name": "num_leaves",
                    "provided_value": num_leaves,
                    "valid_range": "[2, 131072]",
                },
            )

        # Validate n_estimators: [1, 10000]
        if not isinstance(n_estimators, int) or n_estimators < 1 or n_estimators > 10000:
            raise EnsembleError(
                f"Hyperparameter 'n_estimators' has invalid value {n_estimators}. "
                f"Valid range: [1, 10000].",
                details={
                    "param_name": "n_estimators",
                    "provided_value": n_estimators,
                    "valid_range": "[1, 10000]",
                },
            )

        # Validate is_unbalance: must be boolean
        if not isinstance(is_unbalance, bool):
            raise EnsembleError(
                f"Hyperparameter 'is_unbalance' has invalid value {is_unbalance}. "
                f"Must be a boolean (True or False).",
                details={
                    "param_name": "is_unbalance",
                    "provided_value": is_unbalance,
                    "valid_range": "boolean (True or False)",
                },
            )

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data for format errors.

        Checks for:
        - Zero samples
        - Mismatched dimensions between X and y
        - Non-numeric values
        - NaN values in features

        Raises:
            EnsembleError: With specific error type for data format issues.
        """
        # Check for zero samples
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise EnsembleError(
                "Training data contains zero samples.",
                details={"error_type": "zero_samples", "X_shape": X.shape, "y_shape": y.shape},
            )

        # Check mismatched dimensions
        if X.shape[0] != y.shape[0]:
            raise EnsembleError(
                f"Mismatched dimensions: X has {X.shape[0]} samples but y has {y.shape[0]} samples.",
                details={
                    "error_type": "dimension_mismatch",
                    "X_samples": X.shape[0],
                    "y_samples": y.shape[0],
                },
            )

        # Check for non-numeric values in X
        if not np.issubdtype(X.dtype, np.number):
            raise EnsembleError(
                "Training features X contain non-numeric values.",
                details={"error_type": "non_numeric", "dtype": str(X.dtype)},
            )

        # Check for NaN values in X
        if np.isnan(X).any():
            nan_cols = np.where(np.isnan(X).any(axis=0))[0].tolist()
            raise EnsembleError(
                f"Training features X contain NaN values in columns: {nan_cols}.",
                details={"error_type": "nan_values", "nan_columns": nan_cols},
            )

        # Check for NaN values in y
        if np.isnan(y).any():
            raise EnsembleError(
                "Training labels y contain NaN values.",
                details={"error_type": "nan_values", "location": "y"},
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> ModelResult:
        """Train LightGBM classifier on the provided data.

        Args:
            X_train: Training features array, shape (n_samples, n_features).
            y_train: Training labels array, shape (n_samples,).

        Returns:
            ModelResult with trained model, probabilities, metrics, and timing.

        Raises:
            EnsembleError: If data validation fails (NaN, non-numeric, mismatched dims).
        """
        self._validate_data(X_train, y_train)

        start_time = time.time()

        # Build LightGBM parameters
        params = {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "is_unbalance": self.is_unbalance,
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "verbose": -1,
            "random_state": 42,
        }

        logger.info(
            "Starting LightGBM training with params: learning_rate=%.4f, "
            "num_leaves=%d, n_estimators=%d, is_unbalance=%s",
            self.learning_rate,
            self.num_leaves,
            self.n_estimators,
            self.is_unbalance,
        )

        # Create and train the model using sklearn API
        model = lgb.LGBMClassifier(
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            is_unbalance=self.is_unbalance,
            objective="binary",
            verbose=-1,
            random_state=42,
        )

        # Define callback for logging metrics at intervals
        callbacks = [self._create_logging_callback()]

        model.fit(
            X_train,
            y_train,
            callbacks=callbacks,
        )

        self._model = model
        training_time = time.time() - start_time

        # Generate predicted probabilities
        probabilities = model.predict_proba(X_train)[:, 1]

        # Compute metrics
        y_pred = (probabilities >= 0.5).astype(int)
        metrics = {
            "log_loss": log_loss(y_train, probabilities),
            "auc_roc": roc_auc_score(y_train, probabilities),
            "f1": f1_score(y_train, y_pred),
            "precision": precision_score(y_train, y_pred, zero_division=0),
            "recall": recall_score(y_train, y_pred, zero_division=0),
        }

        logger.info(
            "LightGBM training completed in %.2fs. Metrics: log_loss=%.4f, auc_roc=%.4f, f1=%.4f",
            training_time,
            metrics["log_loss"],
            metrics["auc_roc"],
            metrics["f1"],
        )

        return ModelResult(
            model=model,
            probabilities=probabilities,
            metrics=metrics,
            training_time_seconds=training_time,
        )

    def _create_logging_callback(self) -> Any:
        """Create a LightGBM callback that logs metrics at the configured interval."""
        interval = self.logging_interval

        def _log_evaluation(env: Any) -> None:
            """Log evaluation metrics at the configured interval."""
            if (env.iteration + 1) % interval == 0 or env.iteration == 0:
                # Extract metrics from evaluation results if available
                if env.evaluation_result_list:
                    metrics_str = ", ".join(
                        f"{name}={value:.4f}"
                        for _, name, value, _ in env.evaluation_result_list
                    )
                    logger.info(
                        "LightGBM round %d/%d: %s",
                        env.iteration + 1,
                        env.end_iteration,
                        metrics_str,
                    )
                else:
                    logger.info(
                        "LightGBM round %d/%d completed.",
                        env.iteration + 1,
                        env.end_iteration,
                    )

        _log_evaluation.order = 10  # type: ignore[attr-defined]
        return _log_evaluation

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities for input features.

        Args:
            X: Feature array, shape (n_samples, n_features).

        Returns:
            Array of predicted probabilities, shape (n_samples,).

        Raises:
            EnsembleError: If model has not been trained or data is invalid.
        """
        if self._model is None:
            raise EnsembleError(
                "Model has not been trained. Call train() before predict_proba().",
                details={"error_type": "model_not_trained"},
            )

        # Validate input data
        if not isinstance(X, np.ndarray):
            raise EnsembleError(
                "Input X must be a numpy ndarray.",
                details={"error_type": "invalid_type", "provided_type": type(X).__name__},
            )

        if X.shape[0] == 0:
            raise EnsembleError(
                "Input X contains zero samples.",
                details={"error_type": "zero_samples", "X_shape": X.shape},
            )

        if not np.issubdtype(X.dtype, np.number):
            raise EnsembleError(
                "Input features X contain non-numeric values.",
                details={"error_type": "non_numeric", "dtype": str(X.dtype)},
            )

        if np.isnan(X).any():
            nan_cols = np.where(np.isnan(X).any(axis=0))[0].tolist()
            raise EnsembleError(
                f"Input features X contain NaN values in columns: {nan_cols}.",
                details={"error_type": "nan_values", "nan_columns": nan_cols},
            )

        probabilities = self._model.predict_proba(X)[:, 1]
        return probabilities
