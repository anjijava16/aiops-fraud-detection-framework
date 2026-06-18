"""Unit tests for ModelRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.mlflow_manager.model_registry import (
    ModelRegistry,
    ModelStage,
    PromotionThresholds,
    RegistrationResult,
)
from src.models.errors import MLflowError


class TestPromotionThresholds:
    """Tests for PromotionThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        thresholds = PromotionThresholds()
        assert thresholds.f1_min == 0.85
        assert thresholds.auc_roc_min == 0.90
        assert thresholds.precision_min == 0.80
        assert thresholds.recall_min == 0.75

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = PromotionThresholds(f1_min=0.90, auc_roc_min=0.95)
        assert thresholds.f1_min == 0.90
        assert thresholds.auc_roc_min == 0.95


class TestModelStage:
    """Tests for ModelStage constants."""

    def test_stage_values(self):
        """Test stage constant strings."""
        assert ModelStage.NONE == "None"
        assert ModelStage.STAGING == "Staging"
        assert ModelStage.PRODUCTION == "Production"


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def _create_registry(self):
        """Helper to create a registry with mocked MLflow."""
        mock_mlflow = MagicMock()
        mock_client = MagicMock()

        with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.tracking": MagicMock()}):
            registry = ModelRegistry.__new__(ModelRegistry)
            registry._tracking_uri = "http://localhost:5000"
            registry._model_name = "fraud-model"
            registry._thresholds = PromotionThresholds()
            registry._mlflow = mock_mlflow
            registry._client = mock_client
            return registry, mock_mlflow, mock_client

    def test_register_model_first_model(self):
        """Test registering first model when no production model exists."""
        registry, mock_mlflow, mock_client = self._create_registry()
        mock_client.get_latest_versions.return_value = []

        mock_result = MagicMock()
        mock_result.version = "1"
        mock_mlflow.register_model.return_value = mock_result

        result = registry.register_model(
            run_id="run-123",
            metrics={"f1": 0.90},
        )

        assert result.registered is True
        assert result.model_version == "1"
        assert result.model_name == "fraud-model"

    def test_register_model_exceeds_production(self):
        """Test registering model when F1 exceeds production model."""
        registry, mock_mlflow, mock_client = self._create_registry()

        # Mock production model with F1=0.85
        mock_version = MagicMock()
        mock_version.run_id = "old-run"
        mock_client.get_latest_versions.return_value = [mock_version]
        mock_run = MagicMock()
        mock_run.data.metrics = {"f1": 0.85}
        mock_client.get_run.return_value = mock_run

        mock_result = MagicMock()
        mock_result.version = "2"
        mock_mlflow.register_model.return_value = mock_result

        result = registry.register_model(
            run_id="new-run",
            metrics={"f1": 0.90},
        )

        assert result.registered is True
        assert result.model_version == "2"

    def test_register_model_does_not_exceed_production(self):
        """Test that registration is skipped when F1 doesn't exceed production."""
        registry, mock_mlflow, mock_client = self._create_registry()

        mock_version = MagicMock()
        mock_version.run_id = "old-run"
        mock_client.get_latest_versions.return_value = [mock_version]
        mock_run = MagicMock()
        mock_run.data.metrics = {"f1": 0.95}
        mock_client.get_run.return_value = mock_run

        result = registry.register_model(
            run_id="new-run",
            metrics={"f1": 0.90},
        )

        assert result.registered is False
        assert "does not exceed" in result.reason

    def test_promote_to_production_success(self):
        """Test successful promotion to Production with valid metrics."""
        registry, mock_mlflow, mock_client = self._create_registry()
        mock_client.get_latest_versions.return_value = []

        metrics = {"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.85}
        result = registry.promote_to_production(version="1", metrics=metrics)

        assert result["version"] == "1"
        assert result["stage"] == ModelStage.PRODUCTION
        mock_client.transition_model_version_stage.assert_called_with(
            name="fraud-model", version="1", stage="Production"
        )

    def test_promote_to_production_fails_thresholds(self):
        """Test that promotion fails when thresholds are not met."""
        registry, mock_mlflow, mock_client = self._create_registry()

        metrics = {"f1": 0.70, "auc_roc": 0.80, "precision": 0.60, "recall": 0.50}

        with pytest.raises(MLflowError, match="Promotion thresholds not met"):
            registry.promote_to_production(version="1", metrics=metrics)

    def test_promote_demotes_previous_production(self):
        """Test that promoting a new model demotes the previous production model."""
        registry, mock_mlflow, mock_client = self._create_registry()

        # Set up existing production model
        mock_old_version = MagicMock()
        mock_old_version.version = "1"
        mock_client.get_latest_versions.return_value = [mock_old_version]

        metrics = {"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.85}
        registry.promote_to_production(version="2", metrics=metrics)

        # Verify the old model was demoted to None
        calls = mock_client.transition_model_version_stage.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["version"] == "1"
        assert calls[0][1]["stage"] == "None"
        assert calls[1][1]["version"] == "2"
        assert calls[1][1]["stage"] == "Production"

    def test_promote_to_staging(self):
        """Test promoting a model to Staging stage."""
        registry, mock_mlflow, mock_client = self._create_registry()

        registry.promote_to_staging(version="1")

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="fraud-model", version="1", stage="Staging"
        )

    def test_get_production_model_version(self):
        """Test getting the current production model version."""
        registry, mock_mlflow, mock_client = self._create_registry()

        mock_version = MagicMock()
        mock_version.version = "3"
        mock_client.get_latest_versions.return_value = [mock_version]

        version = registry.get_production_model_version()
        assert version == "3"

    def test_get_production_model_version_none(self):
        """Test that None is returned when no production model exists."""
        registry, mock_mlflow, mock_client = self._create_registry()
        mock_client.get_latest_versions.return_value = []

        version = registry.get_production_model_version()
        assert version is None
