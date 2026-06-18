"""AIOps monitoring layer data models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DriftResult:
    """Result of drift detection analysis.

    Attributes:
        is_drifted: True if any feature exceeds drift thresholds.
        drifted_features: Names of features flagged as drifted.
        psi_values: PSI value per feature.
        ks_results: KS-test results per feature as (statistic, p-value) tuples.
        detection_timestamp: ISO 8601 UTC timestamp of detection.
    """

    is_drifted: bool
    drifted_features: list[str] = field(default_factory=list)
    psi_values: dict[str, float] = field(default_factory=dict)  # feature -> PSI
    ks_results: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )  # feature -> (stat, p-value)
    detection_timestamp: str = ""


@dataclass
class AlertEvent:
    """An alert event emitted by the monitoring layer.

    Attributes:
        alert_type: Category — "drift", "latency", or "f1_drop".
        drifted_features: Names of features that triggered the alert.
        psi_values: PSI values for drifted features.
        current_f1: Current live F1 score at time of alert.
        timestamp: ISO 8601 UTC timestamp of alert emission.
        dashboard_link: URL to the Grafana dashboard for context.
    """

    alert_type: str  # "drift", "latency", "f1_drop"
    drifted_features: list[str] = field(default_factory=list)
    psi_values: dict[str, float] = field(default_factory=dict)
    current_f1: float = 0.0
    timestamp: str = ""
    dashboard_link: str = ""


@dataclass
class RetrainResult:
    """Result of an automated retraining cycle.

    Attributes:
        success: True if retraining completed without errors.
        new_model_version: Version string of the retrained model, if successful.
        new_f1: F1 score of the retrained model.
        old_f1: F1 score of the previous production model.
        promoted: True if the retrained model was promoted.
        reason: Explanation if promotion was aborted.
    """

    success: bool
    new_model_version: Optional[str] = None
    new_f1: Optional[float] = None
    old_f1: Optional[float] = None
    promoted: bool = False
    reason: Optional[str] = None  # If not promoted
