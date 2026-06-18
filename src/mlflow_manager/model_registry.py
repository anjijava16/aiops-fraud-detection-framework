"""MLflow model registry - version management and staged promotion.

Provides ModelRegistry class that handles model registration, stage transitions,
and promotion logic based on metric comparison against current production model.

Validates: Requirements 18.1, 18.2, 18.3, 18.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.models.errors import MLflowError

logger = logging.getLogger(__name__)


@dataclass
class PromotionThresholds:
    """Minimum metric thresholds required for automatic promotion.

    Attributes:
        f1_min: Minimum F1 score (default 0.85).
        auc_roc_min: Minimum AUC-ROC score (default 0.90).
        precision_min: Minimum precision (default 0.80).
        recall_min: Minimum recall (default 0.75).
    """

    f1_min: float = 0.85
    auc_roc_min: float = 0.90
    precision_min: float = 0.80
    recall_min: float = 0.75


class ModelStage:
    """Constants for model lifecycle stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"


@dataclass
class RegistrationResult:
    """Result of a model registration attempt.

    Attributes:
        registered: Whether the model was registered.
        model_version: The version number assigned.
        model_name: The registered model name.
        stage: The stage assigned to the model.
        reason: Explanation if registration was skipped.
    """

    registered: bool
    model_version: str | None = None
    model_name: str | None = None
    stage: str = ModelStage.NONE
    reason: str | None = None


class ModelRegistry:
    """Manages model versions and stage transitions in MLflow Model Registry.

    Supports registering models, promoting through stages (None -> Staging -> Production),
    and enforcing promotion thresholds. On new promotion to Production, the previous
    Production model is transitioned to None.

    Usage:
        registry = ModelRegistry(tracking_uri="http://localhost:5000", model_name="fraud-model")
        result = registry.register_model(run_id="abc123", metrics={"f1": 0.92, ...})
        registry.promote_to_production(version="1", metrics={"f1": 0.92, ...})
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        model_name: str = "fraud-detection-ensemble",
        thresholds: PromotionThresholds | None = None,
    ) -> None:
        """Initialize the ModelRegistry.

        Args:
            tracking_uri: MLflow tracking server URI.
            model_name: Name of the registered model.
            thresholds: Promotion thresholds. Uses defaults if None.
        """
        self._tracking_uri = tracking_uri
        self._model_name = model_name
        self._thresholds = thresholds or PromotionThresholds()

        try:
            import mlflow

            mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow = mlflow
            self._client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
        except Exception as e:
            raise MLflowError(
                f"Failed to initialize MLflow model registry: {e}",
                details={"tracking_uri": tracking_uri},
            )

    @property
    def model_name(self) -> str:
        """Return the registered model name."""
        return self._model_name

    @property
    def thresholds(self) -> PromotionThresholds:
        """Return the current promotion thresholds."""
        return self._thresholds

    def register_model(
        self,
        run_id: str,
        metrics: dict[str, float],
        artifact_path: str = "model",
    ) -> RegistrationResult:
        """Register a model when F1 exceeds the current Production model (or first model).

        Args:
            run_id: The MLflow run ID containing the model artifact.
            metrics: Model evaluation metrics (must include 'f1').
            artifact_path: Path within the run's artifact store.

        Returns:
            RegistrationResult with registration details.

        Raises:
            MLflowError: If registration fails unexpectedly.
        """
        candidate_f1 = metrics.get("f1", 0.0)

        # Check if there's a current production model
        production_f1 = self._get_production_f1()

        # Register if candidate exceeds production, or if no production model exists
        if production_f1 is not None and candidate_f1 <= production_f1:
            return RegistrationResult(
                registered=False,
                reason=f"Candidate F1 ({candidate_f1:.4f}) does not exceed "
                f"production F1 ({production_f1:.4f})",
            )

        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = self._mlflow.register_model(model_uri, self._model_name)
            version = result.version

            logger.info(
                f"Registered model '{self._model_name}' version {version} "
                f"(F1={candidate_f1:.4f})"
            )

            return RegistrationResult(
                registered=True,
                model_version=str(version),
                model_name=self._model_name,
                stage=ModelStage.NONE,
            )
        except Exception as e:
            raise MLflowError(
                f"Failed to register model: {e}",
                details={"run_id": run_id, "model_name": self._model_name},
            )

    def promote_to_staging(self, version: str) -> None:
        """Transition a model version to Staging stage.

        Args:
            version: The model version number to promote.

        Raises:
            MLflowError: If transition fails.
        """
        self._transition_stage(version, ModelStage.STAGING)

    def promote_to_production(
        self,
        version: str,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Promote a model version to Production after threshold validation.

        Validates metrics against promotion thresholds before promotion.
        On success, transitions the previous Production model to None.

        Args:
            version: The model version number to promote.
            metrics: Model metrics to validate against thresholds.

        Returns:
            Dict with promotion details including any failed thresholds.

        Raises:
            MLflowError: If thresholds are not met or transition fails.
        """
        # Validate against promotion thresholds
        failed_thresholds = self._validate_thresholds(metrics)

        if failed_thresholds:
            raise MLflowError(
                "Promotion thresholds not met",
                details={"failed_thresholds": failed_thresholds},
            )

        # Transition previous production model to None
        self._demote_current_production()

        # Promote the new model to Production
        self._transition_stage(version, ModelStage.PRODUCTION)

        logger.info(f"Promoted model '{self._model_name}' version {version} to Production")

        return {
            "version": version,
            "stage": ModelStage.PRODUCTION,
            "metrics": metrics,
        }

    def get_production_model_version(self) -> str | None:
        """Get the version number of the current Production model.

        Returns:
            Version string or None if no production model exists.
        """
        try:
            versions = self._client.get_latest_versions(
                self._model_name, stages=["Production"]
            )
            if versions:
                return versions[0].version
            return None
        except Exception:
            return None

    def _get_production_f1(self) -> float | None:
        """Get the F1 score of the current Production model.

        Returns:
            F1 score or None if no production model exists.
        """
        try:
            versions = self._client.get_latest_versions(
                self._model_name, stages=["Production"]
            )
            if not versions:
                return None

            run_id = versions[0].run_id
            run = self._client.get_run(run_id)
            f1 = run.data.metrics.get("f1")
            return f1
        except Exception:
            return None

    def _validate_thresholds(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Validate metrics against promotion thresholds.

        Args:
            metrics: Model evaluation metrics.

        Returns:
            List of failed threshold dicts, empty if all pass.
        """
        failed = []

        checks = [
            ("f1", metrics.get("f1", 0.0), self._thresholds.f1_min),
            ("auc_roc", metrics.get("auc_roc", 0.0), self._thresholds.auc_roc_min),
            ("precision", metrics.get("precision", 0.0), self._thresholds.precision_min),
            ("recall", metrics.get("recall", 0.0), self._thresholds.recall_min),
        ]

        for metric_name, actual, required in checks:
            if actual < required:
                failed.append({
                    "metric": metric_name,
                    "actual": actual,
                    "required": required,
                })

        return failed

    def _demote_current_production(self) -> None:
        """Transition the current Production model to None stage."""
        try:
            versions = self._client.get_latest_versions(
                self._model_name, stages=["Production"]
            )
            for version in versions:
                self._client.transition_model_version_stage(
                    name=self._model_name,
                    version=version.version,
                    stage=ModelStage.NONE,
                )
                logger.info(
                    f"Demoted model version {version.version} from Production to None"
                )
        except Exception:
            # No existing production model, or model not found
            pass

    def _transition_stage(self, version: str, stage: str) -> None:
        """Transition a model version to the specified stage.

        Args:
            version: The model version number.
            stage: The target stage.

        Raises:
            MLflowError: If the transition fails.
        """
        try:
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=version,
                stage=stage,
            )
        except Exception as e:
            raise MLflowError(
                f"Failed to transition model to {stage}: {e}",
                details={"version": version, "stage": stage},
            )
