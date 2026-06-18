"""Soft-voting ensemble combiner for fraud detection.

Combines predicted probabilities from XGBoost, LightGBM, and CatBoost
using configurable weighted averaging and a decision threshold for
binary fraud/non-fraud classification.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.ensemble import EnsembleConfig, EnsemblePrediction
from src.models.errors import EnsembleError

logger = logging.getLogger(__name__)

_EXPECTED_MODELS = {"xgboost", "lightgbm", "catboost"}
_WEIGHT_SUM_TOLERANCE = 1e-6
_MODEL_VERSION = "1.0.0"


class SoftVotingEnsemble:
    """Soft-voting ensemble that combines base model probabilities.

    Computes the weighted average of predicted probabilities from XGBoost,
    LightGBM, and CatBoost, then applies a decision threshold for binary
    classification.

    Args:
        weights: Model weights keyed by model name. Must be positive and sum to 1.0.
        decision_threshold: Probability threshold for fraud classification (0, 1).
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        decision_threshold: float = 0.5,
    ) -> None:
        if weights is None:
            weights = {"xgboost": 1 / 3, "lightgbm": 1 / 3, "catboost": 1 / 3}

        self._validate_weights(weights)
        self._validate_threshold(decision_threshold)

        self.weights = weights
        self.decision_threshold = decision_threshold

    def _validate_weights(self, weights: dict[str, float]) -> None:
        """Validate that weights are positive and sum to 1.0."""
        if set(weights.keys()) != _EXPECTED_MODELS:
            raise EnsembleError(
                f"Weights must be provided for exactly {_EXPECTED_MODELS}, "
                f"got {set(weights.keys())}",
                details={"provided_keys": list(weights.keys())},
            )

        for name, weight in weights.items():
            if weight <= 0.0 or weight >= 1.0:
                raise EnsembleError(
                    f"Weight for '{name}' must be in (0.0, 1.0) exclusive, got {weight}",
                    details={"model": name, "weight": weight},
                )

        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise EnsembleError(
                f"Weights must sum to 1.0 (within tolerance {_WEIGHT_SUM_TOLERANCE}), "
                f"got {weight_sum}",
                details={"weight_sum": weight_sum, "tolerance": _WEIGHT_SUM_TOLERANCE},
            )

    def _validate_threshold(self, threshold: float) -> None:
        """Validate that threshold is in (0, 1) exclusive."""
        if threshold <= 0.0 or threshold >= 1.0:
            raise EnsembleError(
                f"Decision threshold must be in (0.0, 1.0) exclusive, got {threshold}",
                details={"threshold": threshold},
            )

    def predict(
        self, model_probabilities: dict[str, np.ndarray]
    ) -> EnsemblePrediction:
        """Compute ensemble prediction for a single sample.

        Args:
            model_probabilities: Dict mapping model name to probability array
                (single value or 1-element array).

        Returns:
            EnsemblePrediction with fraud_score, is_fraud, model_scores, model_version.

        Raises:
            EnsembleError: If any base model is missing or fails to produce predictions.
        """
        self._validate_model_probabilities(model_probabilities)

        model_scores: dict[str, float] = {}
        fraud_score = 0.0

        for model_name in _EXPECTED_MODELS:
            prob = model_probabilities[model_name]
            score = float(np.atleast_1d(prob).flat[0])
            model_scores[model_name] = score
            fraud_score += self.weights[model_name] * score

        is_fraud = fraud_score >= self.decision_threshold

        return EnsemblePrediction(
            fraud_score=fraud_score,
            is_fraud=is_fraud,
            model_scores=model_scores,
            model_version=_MODEL_VERSION,
        )

    def predict_batch(
        self,
        model_probabilities: dict[str, np.ndarray],
        y_true: Optional[np.ndarray] = None,
    ) -> list[EnsemblePrediction]:
        """Compute ensemble predictions for a batch of samples.

        Args:
            model_probabilities: Dict mapping model name to probability arrays
                of shape (n_samples,).
            y_true: Optional true labels. If provided, metrics are logged.

        Returns:
            List of EnsemblePrediction objects, one per sample.

        Raises:
            EnsembleError: If any base model is missing or arrays have mismatched lengths.
        """
        self._validate_model_probabilities(model_probabilities)
        self._validate_batch_shapes(model_probabilities)

        n_samples = len(next(iter(model_probabilities.values())))
        predictions: list[EnsemblePrediction] = []

        # Compute weighted average across all samples at once
        fraud_scores = np.zeros(n_samples)
        for model_name in _EXPECTED_MODELS:
            probs = np.asarray(model_probabilities[model_name])
            fraud_scores += self.weights[model_name] * probs

        is_fraud_array = fraud_scores >= self.decision_threshold

        for i in range(n_samples):
            model_scores = {
                name: float(model_probabilities[name][i])
                for name in _EXPECTED_MODELS
            }
            predictions.append(
                EnsemblePrediction(
                    fraud_score=float(fraud_scores[i]),
                    is_fraud=bool(is_fraud_array[i]),
                    model_scores=model_scores,
                    model_version=_MODEL_VERSION,
                )
            )

        # Log metrics if ground truth is provided
        if y_true is not None:
            self._log_metrics(is_fraud_array, fraud_scores, y_true)

        return predictions

    def _validate_model_probabilities(
        self, model_probabilities: dict[str, np.ndarray]
    ) -> None:
        """Validate that all expected models have provided probabilities."""
        missing_models = _EXPECTED_MODELS - set(model_probabilities.keys())
        if missing_models:
            raise EnsembleError(
                f"Base model(s) failed to produce predictions: {sorted(missing_models)}",
                details={"missing_models": sorted(list(missing_models))},
            )

        for model_name in _EXPECTED_MODELS:
            prob = model_probabilities[model_name]
            if prob is None:
                raise EnsembleError(
                    f"Base model '{model_name}' produced None predictions",
                    details={"failed_model": model_name},
                )

    def _validate_batch_shapes(
        self, model_probabilities: dict[str, np.ndarray]
    ) -> None:
        """Validate that all probability arrays have the same length."""
        lengths = {
            name: len(np.atleast_1d(probs))
            for name, probs in model_probabilities.items()
            if name in _EXPECTED_MODELS
        }
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise EnsembleError(
                f"Probability arrays have mismatched lengths: {lengths}",
                details={"lengths": lengths},
            )

    def _log_metrics(
        self,
        is_fraud_array: np.ndarray,
        fraud_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> None:
        """Log ensemble evaluation metrics using Python logging."""
        y_pred = is_fraud_array.astype(int)
        y_true_int = np.asarray(y_true).astype(int)

        f1 = f1_score(y_true_int, y_pred, zero_division=0)
        precision = precision_score(y_true_int, y_pred, zero_division=0)
        recall = recall_score(y_true_int, y_pred, zero_division=0)

        # AUC-ROC requires at least two classes in y_true
        unique_classes = np.unique(y_true_int)
        if len(unique_classes) >= 2:
            auc_roc = roc_auc_score(y_true_int, fraud_scores)
        else:
            auc_roc = float("nan")
            logger.warning(
                "AUC-ROC cannot be computed: only one class present in y_true"
            )

        logger.info(
            "Ensemble metrics — F1: %.4f, Precision: %.4f, Recall: %.4f, AUC-ROC: %.4f",
            f1,
            precision,
            recall,
            auc_roc,
        )
