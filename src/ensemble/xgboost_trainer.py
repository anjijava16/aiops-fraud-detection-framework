"""XGBoost trainer with configurable hyperparameters for the ensemble ML layer."""

import logging
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score

from src.models.ensemble import ModelResult
from src.models.errors import EnsembleError

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Train an XGBoost classifier with validated hyperparameters.

    Validates all hyperparameters on initialization and raises EnsembleError
    for any value outside the allowed range. Logs training metrics (log_loss,
    AUC-ROC) at a configurable interval during boosting.

    Attributes:
        learning_rate: Step size shrinkage, range (0.0, 1.0].
        max_depth: Maximum tree depth, range [1, 50].
        n_estimators: Number of boosting rounds, range [1, 10000].
        scale_pos_weight: Ratio for class imbalance, range (0.0, 10000.0].
        logging_interval: Log metrics every N rounds (default 10).
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 300,
        scale_pos_weight: float = 1.0,
        logging_interval: int = 10,
    ) -> None:
        self._validate_hyperparameters(learning_rate, max_depth, n_estimators, scale_pos_weight)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.logging_interval = logging_interval
        self._model: xgb.XGBClassifier | None = None

    @staticmethod
    def _validate_hyperparameters(
        learning_rate: float,
        max_depth: int,
        n_estimators: int,
        scale_pos_weight: float,
    ) -> None:
        """Validate hyperparameter ranges; raise EnsembleError on invalid values."""
        if not (0.0 < learning_rate <= 1.0):
            raise EnsembleError(
                f"Invalid hyperparameter 'learning_rate': {learning_rate}. "
                f"Valid range: (0.0, 1.0].",
                details={"param": "learning_rate", "value": learning_rate, "valid_range": "(0.0, 1.0]"},
            )
        if not (1 <= max_depth <= 50):
            raise EnsembleError(
                f"Invalid hyperparameter 'max_depth': {max_depth}. "
                f"Valid range: [1, 50].",
                details={"param": "max_depth", "value": max_depth, "valid_range": "[1, 50]"},
            )
        if not (1 <= n_estimators <= 10000):
            raise EnsembleError(
                f"Invalid hyperparameter 'n_estimators': {n_estimators}. "
                f"Valid range: [1, 10000].",
                details={"param": "n_estimators", "value": n_estimators, "valid_range": "[1, 10000]"},
            )
        if not (0.0 < scale_pos_weight <= 10000.0):
            raise EnsembleError(
                f"Invalid hyperparameter 'scale_pos_weight': {scale_pos_weight}. "
                f"Valid range: (0.0, 10000.0].",
                details={"param": "scale_pos_weight", "value": scale_pos_weight, "valid_range": "(0.0, 10000.0]"},
            )

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data dimensions and sample count."""
        if X.shape[0] == 0:
            raise EnsembleError(
                "Training data contains zero samples.",
                details={"X_shape": X.shape, "y_shape": y.shape},
            )
        if X.shape[0] != y.shape[0]:
            raise EnsembleError(
                f"Mismatched dimensions: X has {X.shape[0]} samples but y has {y.shape[0]} samples.",
                details={"X_shape": X.shape, "y_shape": y.shape},
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> ModelResult:
        """Train an XGBoost classifier on the provided data.

        Args:
            X_train: Feature matrix of shape (n_samples, n_features).
            y_train: Binary label array of shape (n_samples,).

        Returns:
            ModelResult containing the trained model, predicted probabilities,
            evaluation metrics, and training time.

        Raises:
            EnsembleError: If data validation fails (zero samples or dimension mismatch).
        """
        self._validate_data(X_train, y_train)

        start_time = time.time()

        # Custom callback for logging at configurable intervals
        callbacks = [_MetricsLoggingCallback(self.logging_interval)]

        self._model = xgb.XGBClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
            verbosity=0,
            callbacks=callbacks,
        )

        # Set up evaluation set for metric logging during training
        eval_set = [(X_train, y_train)]

        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        training_time = time.time() - start_time

        # Get predicted probabilities
        probabilities = self._model.predict_proba(X_train)[:, 1]

        # Compute metrics
        y_pred = (probabilities >= 0.5).astype(int)
        metrics = {
            "log_loss": log_loss(y_train, probabilities),
            "auc_roc": roc_auc_score(y_train, probabilities),
            "f1": f1_score(y_train, y_pred, zero_division=0.0),
            "precision": precision_score(y_train, y_pred, zero_division=0.0),
            "recall": recall_score(y_train, y_pred, zero_division=0.0),
        }

        logger.info(
            "XGBoost training complete: log_loss=%.4f, auc_roc=%.4f, f1=%.4f (%.2fs)",
            metrics["log_loss"],
            metrics["auc_roc"],
            metrics["f1"],
            training_time,
        )

        return ModelResult(
            model=self._model,
            probabilities=probabilities,
            metrics=metrics,
            training_time_seconds=training_time,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities for the positive class.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of probabilities in [0.0, 1.0] for the positive class.

        Raises:
            EnsembleError: If the model has not been trained yet.
        """
        if self._model is None:
            raise EnsembleError(
                "Model has not been trained. Call train() first.",
                details={},
            )
        return self._model.predict_proba(X)[:, 1]


class _MetricsLoggingCallback(xgb.callback.TrainingCallback):
    """XGBoost callback that logs metrics at a configurable interval."""

    def __init__(self, logging_interval: int = 10) -> None:
        self.logging_interval = logging_interval

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict) -> bool:
        """Log metrics every `logging_interval` rounds."""
        if (epoch + 1) % self.logging_interval == 0:
            # Extract metrics from evaluation log
            metrics_str_parts = []
            for dataset, metrics in evals_log.items():
                for metric_name, values in metrics.items():
                    value = values[-1] if values else 0.0
                    metrics_str_parts.append(f"{dataset}_{metric_name}={value:.4f}")

            if metrics_str_parts:
                logger.info(
                    "XGBoost round %d: %s",
                    epoch + 1,
                    ", ".join(metrics_str_parts),
                )
        # Return False to continue training
        return False
