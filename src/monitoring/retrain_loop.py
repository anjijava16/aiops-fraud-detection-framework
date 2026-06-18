"""Automated retraining loop triggered by drift detection events.

Manages retraining lifecycle: trigger evaluation, data assembly, pipeline
execution, model registration, and promotion decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

import numpy as np

from src.models.monitoring import DriftResult, RetrainResult

logger = logging.getLogger(__name__)


class PipelineExecutor(Protocol):
    """Protocol for pipeline execution callables.

    Implementors should execute the full ML pipeline (preprocessing, training,
    evaluation) and return a dict with 'model_version' and 'f1_score' keys.
    """

    def __call__(self, data: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Execute the pipeline.

        Args:
            data: Combined training data (features).
            labels: Combined training labels.

        Returns:
            Dict with keys: 'model_version' (str), 'f1_score' (float).
        """
        ...


class ModelRegistrar(Protocol):
    """Protocol for model registration and promotion.

    Implementors should handle MLflow registration and stage promotion.
    """

    def register_and_promote(self, model_version: str, stage: str) -> bool:
        """Register model and promote to given stage.

        Args:
            model_version: The version string to register.
            stage: Target stage ("Staging", "Production").

        Returns:
            True if promotion succeeded.
        """
        ...

    def get_production_f1(self) -> Optional[float]:
        """Get the current production model's F1 score.

        Returns:
            F1 score or None if no production model exists.
        """
        ...


class RetrainLoop:
    """Manages automated retraining triggered by drift detection.

    Features:
        - Triggers full pipeline re-execution when >= N features drifted
        - Combines recent production data with original training data
        - Registers retrained model in MLflow, promotes to Staging
        - Aborts promotion if retrained F1 <= current Production F1
        - Queues retraining requests while one is in progress (max 1 queued)
        - Skips retraining if < min_records available
    """

    def __init__(
        self,
        drift_feature_threshold: int = 3,
        min_records: int = 1000,
        pipeline_executor: Optional[PipelineExecutor] = None,
        model_registrar: Optional[ModelRegistrar] = None,
        data_provider: Optional[Callable[[], tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        """Initialize the RetrainLoop.

        Args:
            drift_feature_threshold: Minimum drifted features to trigger retrain (default 3).
            min_records: Minimum records required for retraining (default 1000).
            pipeline_executor: Callable that executes the ML pipeline.
            model_registrar: Object handling model registration and promotion.
            data_provider: Callable returning (features, labels) for retraining.
        """
        self._drift_feature_threshold = drift_feature_threshold
        self._min_records = min_records
        self._pipeline_executor = pipeline_executor
        self._model_registrar = model_registrar
        self._data_provider = data_provider

        self._is_retraining: bool = False
        self._queued_request: Optional[DriftResult] = None

    @property
    def drift_feature_threshold(self) -> int:
        """Return the drift feature threshold for triggering retrain."""
        return self._drift_feature_threshold

    @property
    def min_records(self) -> int:
        """Return the minimum records required for retraining."""
        return self._min_records

    @property
    def is_retraining(self) -> bool:
        """Return True if a retraining is currently in progress."""
        return self._is_retraining

    @property
    def has_queued_request(self) -> bool:
        """Return True if there is a queued retraining request."""
        return self._queued_request is not None

    def should_trigger(self, drift_result: DriftResult) -> bool:
        """Determine if retraining should be triggered based on drift result.

        Args:
            drift_result: The drift detection result.

        Returns:
            True if number of drifted features >= threshold.
        """
        return len(drift_result.drifted_features) >= self._drift_feature_threshold

    def trigger(self, drift_result: Optional[DriftResult] = None) -> RetrainResult:
        """Trigger the retraining loop.

        If a retraining is already in progress, the request is queued (max 1).

        Args:
            drift_result: Optional drift result that triggered this retrain.

        Returns:
            RetrainResult indicating outcome.
        """
        # Check if drift_result meets threshold (if provided)
        if drift_result and not self.should_trigger(drift_result):
            return RetrainResult(
                success=False,
                promoted=False,
                reason=(
                    f"Only {len(drift_result.drifted_features)} features drifted, "
                    f"threshold is {self._drift_feature_threshold}"
                ),
            )

        # Queue if already retraining
        if self._is_retraining:
            self._queued_request = drift_result
            logger.info("Retraining in progress — request queued")
            return RetrainResult(
                success=False,
                promoted=False,
                reason="Retraining already in progress, request queued",
            )

        return self._execute_retrain()

    def process_queue(self) -> Optional[RetrainResult]:
        """Process the queued retraining request if one exists.

        Returns:
            RetrainResult if a queued request was processed, None otherwise.
        """
        if self._queued_request is None:
            return None

        queued = self._queued_request
        self._queued_request = None

        if queued and not self.should_trigger(queued):
            return RetrainResult(
                success=False,
                promoted=False,
                reason="Queued drift result no longer meets threshold",
            )

        return self._execute_retrain()

    def _execute_retrain(self) -> RetrainResult:
        """Execute the actual retraining pipeline.

        Returns:
            RetrainResult with outcome details.
        """
        self._is_retraining = True

        try:
            # Get data for retraining
            if self._data_provider is None:
                self._is_retraining = False
                return RetrainResult(
                    success=False,
                    promoted=False,
                    reason="No data provider configured",
                )

            data, labels = self._data_provider()

            # Check minimum records
            if len(data) < self._min_records:
                logger.warning(
                    "Retraining skipped: only %d records available, minimum is %d",
                    len(data),
                    self._min_records,
                )
                self._is_retraining = False
                return RetrainResult(
                    success=False,
                    promoted=False,
                    reason=(
                        f"Insufficient data: {len(data)} records "
                        f"(minimum: {self._min_records})"
                    ),
                )

            # Execute pipeline
            if self._pipeline_executor is None:
                self._is_retraining = False
                return RetrainResult(
                    success=False,
                    promoted=False,
                    reason="No pipeline executor configured",
                )

            result = self._pipeline_executor(data, labels)
            new_model_version = result["model_version"]
            new_f1 = result["f1_score"]

            # Get current production F1
            old_f1: Optional[float] = None
            if self._model_registrar:
                old_f1 = self._model_registrar.get_production_f1()

            # Decide on promotion
            if old_f1 is not None and new_f1 <= old_f1:
                logger.warning(
                    "Retrained model F1 (%.4f) <= production F1 (%.4f). "
                    "Promotion aborted.",
                    new_f1,
                    old_f1,
                )
                self._is_retraining = False
                return RetrainResult(
                    success=True,
                    new_model_version=new_model_version,
                    new_f1=new_f1,
                    old_f1=old_f1,
                    promoted=False,
                    reason=(
                        f"New F1 ({new_f1:.4f}) <= current production F1 ({old_f1:.4f})"
                    ),
                )

            # Promote to Staging
            promoted = False
            if self._model_registrar:
                promoted = self._model_registrar.register_and_promote(
                    new_model_version, "Staging"
                )

            logger.info(
                "Retraining complete: model=%s, F1=%.4f, promoted=%s",
                new_model_version,
                new_f1,
                promoted,
            )

            self._is_retraining = False
            return RetrainResult(
                success=True,
                new_model_version=new_model_version,
                new_f1=new_f1,
                old_f1=old_f1,
                promoted=promoted,
            )

        except Exception as e:
            logger.error("Retraining failed: %s", str(e))
            self._is_retraining = False
            return RetrainResult(
                success=False,
                promoted=False,
                reason=f"Retraining error: {str(e)}",
            )
