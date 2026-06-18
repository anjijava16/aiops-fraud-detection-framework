"""Pipeline orchestrator - execute layers in order with failure handling.

Provides PipelineOrchestrator class that executes the fraud detection pipeline
layers sequentially: Ingestion → Preprocessing → Training → Explainability →
MLflow → Deployment → Monitoring. Supports dry-run mode, concurrent execution
rejection, and execution duration logging.

Validates: Requirements 28.1, 28.2, 28.3, 28.4, 28.5
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.models.errors import PipelineError

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DRY_RUN_COMPLETED = "dry_run_completed"
    CONFLICT = "conflict"


class LayerName(str, Enum):
    """Names of pipeline layers in execution order."""

    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EXPLAINABILITY = "explainability"
    MLFLOW = "mlflow"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class LayerResult:
    """Result from executing a single pipeline layer.

    Attributes:
        layer: Name of the layer.
        success: Whether the layer executed successfully.
        duration_seconds: Execution duration in seconds.
        error: Error message if the layer failed.
        details: Additional execution details.
    """

    layer: str
    success: bool
    duration_seconds: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result from a full pipeline execution.

    Attributes:
        status: Final pipeline status.
        layer_results: Results from each executed layer.
        total_duration_seconds: Total execution time.
        failed_layer: Name of the layer that failed (if any).
        error: Error message from the failed layer.
    """

    status: PipelineStatus
    layer_results: list[LayerResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    failed_layer: str | None = None
    error: str | None = None


class PipelineOrchestrator:
    """Orchestrates end-to-end pipeline execution across all layers.

    Executes layers in order: Ingestion → Preprocessing → Training →
    Explainability → MLflow → Deployment → Monitoring.

    Features:
    - Halts on failure, logs context, emits alert
    - Supports dry-run mode (validates config without executing)
    - Logs execution durations per layer and total
    - Rejects concurrent execution with conflict status

    Usage:
        orchestrator = PipelineOrchestrator(config=config_dict)
        orchestrator.register_layer("ingestion", ingest_fn)
        orchestrator.register_layer("preprocessing", preprocess_fn)
        result = orchestrator.execute()
    """

    LAYER_ORDER = [
        LayerName.INGESTION,
        LayerName.PREPROCESSING,
        LayerName.TRAINING,
        LayerName.EXPLAINABILITY,
        LayerName.MLFLOW,
        LayerName.DEPLOYMENT,
        LayerName.MONITORING,
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
        on_failure: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialize the PipelineOrchestrator.

        Args:
            config: Pipeline configuration dictionary.
            dry_run: If True, validate without executing layer logic.
            on_failure: Optional callback invoked on layer failure(layer_name, error_msg).
        """
        self._config = config or {}
        self._dry_run = dry_run
        self._on_failure = on_failure
        self._layers: dict[str, Callable[..., Any]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._status = PipelineStatus.IDLE
        self._last_result: PipelineResult | None = None

    @property
    def status(self) -> PipelineStatus:
        """Return the current pipeline status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Return True if the pipeline is currently executing."""
        return self._running

    @property
    def last_result(self) -> PipelineResult | None:
        """Return the result of the last execution."""
        return self._last_result

    def register_layer(self, name: str, handler: Callable[..., Any]) -> None:
        """Register a layer execution handler.

        Args:
            name: Layer name (must match LayerName enum value).
            handler: Callable that executes the layer logic.
                     Should accept config dict and return any result.
        """
        self._layers[name] = handler

    def execute(self, config_override: dict[str, Any] | None = None) -> PipelineResult:
        """Execute the full pipeline.

        Runs layers in order, halting on first failure. Rejects concurrent
        execution attempts.

        Args:
            config_override: Optional config overrides for this execution.

        Returns:
            PipelineResult with execution details.

        Raises:
            PipelineError: If concurrent execution is attempted.
        """
        # Reject concurrent execution
        if not self._lock.acquire(blocking=False):
            result = PipelineResult(status=PipelineStatus.CONFLICT)
            raise PipelineError(
                "Pipeline execution rejected: another execution is in progress",
                details={"status": "conflict"},
            )

        try:
            self._running = True
            self._status = PipelineStatus.RUNNING

            effective_config = {**self._config, **(config_override or {})}
            start_time = time.time()
            layer_results: list[LayerResult] = []

            logger.info(
                f"Pipeline execution started (dry_run={self._dry_run}, "
                f"layers={len(self.LAYER_ORDER)})"
            )

            for layer_name in self.LAYER_ORDER:
                layer_start = time.time()

                if self._dry_run:
                    # In dry-run mode, just validate the layer is registered
                    layer_result = LayerResult(
                        layer=layer_name.value,
                        success=True,
                        duration_seconds=0.0,
                        details={"mode": "dry_run", "registered": layer_name.value in self._layers},
                    )
                    layer_results.append(layer_result)
                    logger.info(f"[DRY-RUN] Layer '{layer_name.value}' validated")
                    continue

                # Execute the layer
                handler = self._layers.get(layer_name.value)
                if handler is None:
                    # Skip unregistered layers with a warning
                    logger.warning(f"Layer '{layer_name.value}' has no registered handler, skipping")
                    layer_result = LayerResult(
                        layer=layer_name.value,
                        success=True,
                        duration_seconds=0.0,
                        details={"skipped": True, "reason": "no handler registered"},
                    )
                    layer_results.append(layer_result)
                    continue

                try:
                    handler(effective_config)
                    duration = time.time() - layer_start
                    layer_result = LayerResult(
                        layer=layer_name.value,
                        success=True,
                        duration_seconds=duration,
                    )
                    layer_results.append(layer_result)
                    logger.info(
                        f"Layer '{layer_name.value}' completed in {duration:.2f}s"
                    )

                except Exception as e:
                    duration = time.time() - layer_start
                    error_msg = str(e)
                    layer_result = LayerResult(
                        layer=layer_name.value,
                        success=False,
                        duration_seconds=duration,
                        error=error_msg,
                    )
                    layer_results.append(layer_result)

                    logger.error(
                        f"Layer '{layer_name.value}' FAILED after {duration:.2f}s: {error_msg}"
                    )

                    # Emit alert on failure
                    if self._on_failure:
                        try:
                            self._on_failure(layer_name.value, error_msg)
                        except Exception as alert_err:
                            logger.warning(f"Failed to emit failure alert: {alert_err}")

                    # Halt pipeline
                    total_duration = time.time() - start_time
                    result = PipelineResult(
                        status=PipelineStatus.FAILED,
                        layer_results=layer_results,
                        total_duration_seconds=total_duration,
                        failed_layer=layer_name.value,
                        error=error_msg,
                    )
                    self._status = PipelineStatus.FAILED
                    self._last_result = result
                    return result

            # All layers completed successfully
            total_duration = time.time() - start_time
            final_status = (
                PipelineStatus.DRY_RUN_COMPLETED
                if self._dry_run
                else PipelineStatus.COMPLETED
            )

            result = PipelineResult(
                status=final_status,
                layer_results=layer_results,
                total_duration_seconds=total_duration,
            )
            self._status = final_status
            self._last_result = result

            logger.info(
                f"Pipeline execution completed in {total_duration:.2f}s "
                f"(status={final_status.value})"
            )

            return result

        finally:
            self._running = False
            self._lock.release()
