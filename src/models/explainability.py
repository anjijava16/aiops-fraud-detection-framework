"""Explainability layer data models."""

from dataclasses import dataclass


@dataclass
class SHAPExplanation:
    """SHAP explanation for a single prediction.

    Attributes:
        feature_names: Ordered list of feature names.
        shap_values: Mapping of feature name to its SHAP contribution value.
        base_value: Expected model output (average prediction).
        predicted_score: Model's predicted fraud probability for this instance.
    """

    feature_names: list[str]
    shap_values: dict[str, float]  # feature_name -> SHAP value
    base_value: float
    predicted_score: float


@dataclass
class LIMEExplanation:
    """LIME local surrogate explanation for a single prediction.

    Attributes:
        feature_contributions: Top-N (feature_name, contribution) pairs.
        fidelity_score: R-squared of the local surrogate model.
        is_low_confidence: True if fidelity_score < 0.6.
        num_perturbation_samples: Number of perturbation samples used.
    """

    feature_contributions: list[tuple[str, float]]  # Top-N
    fidelity_score: float  # R-squared of surrogate
    is_low_confidence: bool  # True if fidelity < 0.6
    num_perturbation_samples: int


@dataclass
class AuditEntry:
    """A single entry in the decision audit log.

    Attributes:
        transaction_id: Unique transaction identifier.
        timestamp: ISO 8601 UTC with millisecond precision.
        model_version: Version of the model that produced the prediction.
        fraud_score: Predicted fraud probability.
        is_fraud: Binary classification result.
        top_5_shap: Top-5 SHAP feature contributions by absolute magnitude.
    """

    transaction_id: str
    timestamp: str  # ISO 8601 UTC with ms precision
    model_version: str
    fraud_score: float
    is_fraud: bool
    top_5_shap: dict[str, float]  # Top-5 SHAP contributions
