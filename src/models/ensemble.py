"""Ensemble ML layer data models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModelResult:
    """Output from training a single base model (XGBoost, LightGBM, or CatBoost).

    Attributes:
        model: The trained model object.
        probabilities: Predicted fraud probabilities, shape (n_samples,).
        metrics: Evaluation metrics — keys include f1, auc_roc, precision, recall, log_loss.
        training_time_seconds: Wall-clock training duration in seconds.
    """

    model: Any  # Trained model object
    probabilities: np.ndarray  # Shape: (n_samples,)
    metrics: dict[str, float]  # {f1, auc_roc, precision, recall, log_loss}
    training_time_seconds: float


@dataclass
class EnsemblePrediction:
    """Result of ensemble soft-voting prediction.

    Attributes:
        fraud_score: Weighted average probability in [0.0, 1.0].
        is_fraud: True if fraud_score >= decision threshold.
        model_scores: Per-model probabilities keyed by model name.
        model_version: Version identifier of the ensemble model.
    """

    fraud_score: float  # Weighted average probability [0.0, 1.0]
    is_fraud: bool  # fraud_score >= threshold
    model_scores: dict[str, float]  # Per-model probabilities
    model_version: str


@dataclass
class EnsembleConfig:
    """Configuration for the soft-voting ensemble.

    Attributes:
        weights: Model weights keyed by model name. Must sum to 1.0.
        decision_threshold: Probability threshold for binary classification.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "xgboost": 1 / 3,
            "lightgbm": 1 / 3,
            "catboost": 1 / 3,
        }
    )
    decision_threshold: float = 0.5
