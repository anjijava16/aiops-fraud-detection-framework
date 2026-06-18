"""Unit tests for RetrainLoop."""

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models.monitoring import DriftResult, RetrainResult
from src.monitoring.retrain_loop import RetrainLoop


@pytest.fixture
def drift_result_3_features():
    """Drift result with 3 drifted features (meets default threshold)."""
    return DriftResult(
        is_drifted=True,
        drifted_features=["V1", "V2", "V3"],
        psi_values={"V1": 0.35, "V2": 0.28, "V3": 0.22},
        ks_results={
            "V1": (0.45, 0.001),
            "V2": (0.38, 0.003),
            "V3": (0.30, 0.01),
        },
        detection_timestamp="2024-01-15T10:00:00Z",
    )


@pytest.fixture
def drift_result_2_features():
    """Drift result with 2 drifted features (below default threshold)."""
    return DriftResult(
        is_drifted=True,
        drifted_features=["V1", "V2"],
        psi_values={"V1": 0.35, "V2": 0.28},
        ks_results={"V1": (0.45, 0.001), "V2": (0.38, 0.003)},
        detection_timestamp="2024-01-15T10:00:00Z",
    )


@pytest.fixture
def mock_pipeline_executor():
    """Mock pipeline executor that returns a successful training result."""
    executor = MagicMock()
    executor.return_value = {
        "model_version": "v2.0.0",
        "f1_score": 0.95,
    }
    return executor


@pytest.fixture
def mock_model_registrar():
    """Mock model registrar."""
    registrar = MagicMock()
    registrar.get_production_f1.return_value = 0.90
    registrar.register_and_promote.return_value = True
    return registrar


@pytest.fixture
def mock_data_provider():
    """Mock data provider returning sufficient data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((2000, 30))
    labels = rng.integers(0, 2, size=2000)
    return MagicMock(return_value=(data, labels))


@pytest.fixture
def retrain_loop(mock_pipeline_executor, mock_model_registrar, mock_data_provider):
    """Create a RetrainLoop with mocked dependencies."""
    return RetrainLoop(
        drift_feature_threshold=3,
        min_records=1000,
        pipeline_executor=mock_pipeline_executor,
        model_registrar=mock_model_registrar,
        data_provider=mock_data_provider,
    )


class TestRetrainLoopInit:
    """Test RetrainLoop initialization."""

    def test_default_parameters(self):
        loop = RetrainLoop()
        assert loop.drift_feature_threshold == 3
        assert loop.min_records == 1000
        assert loop.is_retraining is False
        assert loop.has_queued_request is False

    def test_custom_parameters(self):
        loop = RetrainLoop(drift_feature_threshold=5, min_records=500)
        assert loop.drift_feature_threshold == 5
        assert loop.min_records == 500


class TestShouldTrigger:
    """Test trigger threshold logic."""

    def test_triggers_at_threshold(self, retrain_loop, drift_result_3_features):
        assert retrain_loop.should_trigger(drift_result_3_features) is True

    def test_does_not_trigger_below_threshold(self, retrain_loop, drift_result_2_features):
        assert retrain_loop.should_trigger(drift_result_2_features) is False

    def test_triggers_above_threshold(self, retrain_loop):
        result = DriftResult(
            is_drifted=True,
            drifted_features=["V1", "V2", "V3", "V4", "V5"],
            detection_timestamp="2024-01-15T10:00:00Z",
        )
        assert retrain_loop.should_trigger(result) is True


class TestTriggerRetrain:
    """Test retraining execution."""

    def test_successful_retrain_with_promotion(
        self, retrain_loop, drift_result_3_features, mock_pipeline_executor, mock_model_registrar
    ):
        result = retrain_loop.trigger(drift_result_3_features)

        assert result.success is True
        assert result.new_model_version == "v2.0.0"
        assert result.new_f1 == 0.95
        assert result.old_f1 == 0.90
        assert result.promoted is True
        mock_pipeline_executor.assert_called_once()
        mock_model_registrar.register_and_promote.assert_called_once_with(
            "v2.0.0", "Staging"
        )

    def test_retrain_aborted_new_f1_not_better(
        self, mock_data_provider, mock_model_registrar
    ):
        """Promotion aborted if new F1 <= current production F1."""
        executor = MagicMock()
        executor.return_value = {"model_version": "v2.0.0", "f1_score": 0.88}
        mock_model_registrar.get_production_f1.return_value = 0.90

        loop = RetrainLoop(
            pipeline_executor=executor,
            model_registrar=mock_model_registrar,
            data_provider=mock_data_provider,
        )

        result = loop.trigger()
        assert result.success is True
        assert result.promoted is False
        assert "0.88" in result.reason
        assert "0.90" in result.reason

    def test_retrain_below_drift_threshold(self, retrain_loop, drift_result_2_features):
        """Retrain not triggered when drift is below threshold."""
        result = retrain_loop.trigger(drift_result_2_features)
        assert result.success is False
        assert result.promoted is False
        assert "threshold" in result.reason.lower()

    def test_retrain_insufficient_data(self, mock_pipeline_executor, mock_model_registrar):
        """Retrain skipped when not enough data available."""
        small_data = np.random.default_rng(0).standard_normal((500, 30))
        small_labels = np.random.default_rng(0).integers(0, 2, size=500)
        data_provider = MagicMock(return_value=(small_data, small_labels))

        loop = RetrainLoop(
            min_records=1000,
            pipeline_executor=mock_pipeline_executor,
            model_registrar=mock_model_registrar,
            data_provider=data_provider,
        )

        result = loop.trigger()
        assert result.success is False
        assert "500 records" in result.reason
        mock_pipeline_executor.assert_not_called()

    def test_retrain_no_data_provider(self):
        """Retrain fails gracefully without data provider."""
        loop = RetrainLoop(data_provider=None)
        result = loop.trigger()
        assert result.success is False
        assert "data provider" in result.reason.lower()

    def test_retrain_no_pipeline_executor(self, mock_data_provider):
        """Retrain fails gracefully without pipeline executor."""
        loop = RetrainLoop(
            pipeline_executor=None, data_provider=mock_data_provider
        )
        result = loop.trigger()
        assert result.success is False
        assert "pipeline executor" in result.reason.lower()

    def test_retrain_no_production_model(
        self, mock_pipeline_executor, mock_data_provider
    ):
        """Promotion succeeds when no production model exists."""
        registrar = MagicMock()
        registrar.get_production_f1.return_value = None
        registrar.register_and_promote.return_value = True

        loop = RetrainLoop(
            pipeline_executor=mock_pipeline_executor,
            model_registrar=registrar,
            data_provider=mock_data_provider,
        )

        result = loop.trigger()
        assert result.success is True
        assert result.promoted is True
        assert result.old_f1 is None


class TestQueueing:
    """Test retraining request queueing."""

    def test_queue_while_retraining(
        self, drift_result_3_features, mock_data_provider
    ):
        """Request is queued when retraining is in progress."""
        # Create a long-running pipeline executor
        import threading

        barrier = threading.Barrier(2, timeout=5)

        def slow_executor(data, labels):
            barrier.wait()  # Wait for test to proceed
            return {"model_version": "v2.0.0", "f1_score": 0.95}

        registrar = MagicMock()
        registrar.get_production_f1.return_value = 0.90
        registrar.register_and_promote.return_value = True

        loop = RetrainLoop(
            pipeline_executor=slow_executor,
            model_registrar=registrar,
            data_provider=mock_data_provider,
        )

        # Simulate in-progress state manually
        loop._is_retraining = True

        result = loop.trigger(drift_result_3_features)
        assert result.success is False
        assert "queued" in result.reason.lower()
        assert loop.has_queued_request is True

    def test_max_one_queued_request(self, retrain_loop, drift_result_3_features):
        """Only one request is queued at a time."""
        retrain_loop._is_retraining = True

        # Queue first
        retrain_loop.trigger(drift_result_3_features)
        assert retrain_loop.has_queued_request is True

        # Queue second — replaces first
        other_drift = DriftResult(
            is_drifted=True,
            drifted_features=["V4", "V5", "V6"],
            detection_timestamp="2024-01-15T11:00:00Z",
        )
        retrain_loop.trigger(other_drift)
        assert retrain_loop.has_queued_request is True

    def test_process_queue(self, retrain_loop, drift_result_3_features):
        """Queued request is processed after current retrain completes."""
        retrain_loop._is_retraining = True
        retrain_loop.trigger(drift_result_3_features)
        retrain_loop._is_retraining = False

        result = retrain_loop.process_queue()
        assert result is not None
        assert result.success is True

    def test_process_empty_queue(self, retrain_loop):
        """Processing empty queue returns None."""
        result = retrain_loop.process_queue()
        assert result is None


class TestErrorHandling:
    """Test error handling during retraining."""

    def test_pipeline_exception_handled(self, mock_data_provider, mock_model_registrar):
        """Pipeline exceptions are caught and reported."""
        executor = MagicMock(side_effect=RuntimeError("Training crashed"))

        loop = RetrainLoop(
            pipeline_executor=executor,
            model_registrar=mock_model_registrar,
            data_provider=mock_data_provider,
        )

        result = loop.trigger()
        assert result.success is False
        assert "Training crashed" in result.reason
        assert loop.is_retraining is False  # Cleaned up
