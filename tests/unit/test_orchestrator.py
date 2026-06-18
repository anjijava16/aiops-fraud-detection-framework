"""Unit tests for PipelineOrchestrator."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.models.errors import PipelineError
from src.pipeline.orchestrator import (
    LayerName,
    LayerResult,
    PipelineOrchestrator,
    PipelineResult,
    PipelineStatus,
)


class TestPipelineStatus:
    """Tests for PipelineStatus enum."""

    def test_status_values(self):
        """Test PipelineStatus enum values."""
        assert PipelineStatus.IDLE == "idle"
        assert PipelineStatus.RUNNING == "running"
        assert PipelineStatus.COMPLETED == "completed"
        assert PipelineStatus.FAILED == "failed"
        assert PipelineStatus.DRY_RUN_COMPLETED == "dry_run_completed"
        assert PipelineStatus.CONFLICT == "conflict"


class TestLayerName:
    """Tests for LayerName enum."""

    def test_layer_order(self):
        """Test that LAYER_ORDER contains all expected layers."""
        assert LayerName.INGESTION in PipelineOrchestrator.LAYER_ORDER
        assert LayerName.PREPROCESSING in PipelineOrchestrator.LAYER_ORDER
        assert LayerName.TRAINING in PipelineOrchestrator.LAYER_ORDER
        assert LayerName.EXPLAINABILITY in PipelineOrchestrator.LAYER_ORDER
        assert LayerName.MLFLOW in PipelineOrchestrator.LAYER_ORDER
        assert LayerName.DEPLOYMENT in PipelineOrchestrator.LAYER_ORDER
        assert LayerName.MONITORING in PipelineOrchestrator.LAYER_ORDER


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator class."""

    def test_initial_status_idle(self):
        """Test that orchestrator starts in IDLE status."""
        orch = PipelineOrchestrator()
        assert orch.status == PipelineStatus.IDLE
        assert orch.is_running is False

    def test_execute_all_layers_success(self):
        """Test successful execution of all registered layers."""
        orch = PipelineOrchestrator()

        execution_order = []
        for layer in PipelineOrchestrator.LAYER_ORDER:
            orch.register_layer(
                layer.value,
                lambda cfg, name=layer.value: execution_order.append(name),
            )

        result = orch.execute()

        assert result.status == PipelineStatus.COMPLETED
        assert result.total_duration_seconds >= 0
        assert len(result.layer_results) == 7
        assert all(r.success for r in result.layer_results)
        assert execution_order == [l.value for l in PipelineOrchestrator.LAYER_ORDER]

    def test_execute_halts_on_failure(self):
        """Test that execution halts on first layer failure."""
        orch = PipelineOrchestrator()

        orch.register_layer("ingestion", lambda cfg: None)
        orch.register_layer("preprocessing", lambda cfg: (_ for _ in ()).throw(RuntimeError("SMOTE failed")))
        orch.register_layer("training", lambda cfg: None)

        result = orch.execute()

        assert result.status == PipelineStatus.FAILED
        assert result.failed_layer == "preprocessing"
        assert "SMOTE failed" in result.error
        # Only 2 layer results (ingestion success + preprocessing failure)
        executed = [r for r in result.layer_results if not r.details.get("skipped")]
        assert len(executed) == 2

    def test_execute_logs_durations(self):
        """Test that execution durations are logged per layer."""
        orch = PipelineOrchestrator()
        orch.register_layer("ingestion", lambda cfg: time.sleep(0.01))
        orch.register_layer("preprocessing", lambda cfg: time.sleep(0.01))

        result = orch.execute()

        for layer_result in result.layer_results:
            if not layer_result.details.get("skipped"):
                assert layer_result.duration_seconds >= 0

    def test_dry_run_mode(self):
        """Test dry-run mode validates without executing."""
        orch = PipelineOrchestrator(dry_run=True)
        execution_count = {"count": 0}

        orch.register_layer("ingestion", lambda cfg: execution_count.__setitem__("count", 1))

        result = orch.execute()

        assert result.status == PipelineStatus.DRY_RUN_COMPLETED
        assert execution_count["count"] == 0
        assert len(result.layer_results) == 7

    def test_concurrent_execution_rejected(self):
        """Test that concurrent execution raises PipelineError."""
        orch = PipelineOrchestrator()
        barrier = threading.Barrier(2, timeout=5)

        def slow_handler(cfg):
            barrier.wait()
            time.sleep(0.1)

        orch.register_layer("ingestion", slow_handler)

        # Start first execution in background
        first_result = [None]

        def run_first():
            first_result[0] = orch.execute()

        t = threading.Thread(target=run_first)
        t.start()

        # Wait for first execution to start
        barrier.wait()
        time.sleep(0.01)

        # Attempt concurrent execution
        with pytest.raises(PipelineError, match="another execution is in progress"):
            orch.execute()

        t.join(timeout=5)

    def test_on_failure_callback(self):
        """Test that on_failure callback is invoked on layer failure."""
        failures = []
        orch = PipelineOrchestrator(on_failure=lambda layer, err: failures.append((layer, err)))

        orch.register_layer("ingestion", lambda cfg: (_ for _ in ()).throw(ValueError("bad data")))

        result = orch.execute()

        assert result.status == PipelineStatus.FAILED
        assert len(failures) == 1
        assert failures[0][0] == "ingestion"
        assert "bad data" in failures[0][1]

    def test_unregistered_layers_skipped(self):
        """Test that layers without handlers are skipped."""
        orch = PipelineOrchestrator()
        # Register only ingestion
        orch.register_layer("ingestion", lambda cfg: None)

        result = orch.execute()

        assert result.status == PipelineStatus.COMPLETED
        # All layers reported but most are skipped
        skipped = [r for r in result.layer_results if r.details.get("skipped")]
        assert len(skipped) == 6  # all except ingestion

    def test_status_after_completion(self):
        """Test status is COMPLETED after successful execution."""
        orch = PipelineOrchestrator()
        orch.execute()
        assert orch.status == PipelineStatus.COMPLETED
        assert orch.is_running is False

    def test_status_after_failure(self):
        """Test status is FAILED after failed execution."""
        orch = PipelineOrchestrator()
        orch.register_layer("ingestion", lambda cfg: (_ for _ in ()).throw(RuntimeError("error")))
        orch.execute()
        assert orch.status == PipelineStatus.FAILED

    def test_last_result_stored(self):
        """Test that last_result stores the most recent execution result."""
        orch = PipelineOrchestrator()
        assert orch.last_result is None

        orch.execute()
        assert orch.last_result is not None
        assert orch.last_result.status == PipelineStatus.COMPLETED
