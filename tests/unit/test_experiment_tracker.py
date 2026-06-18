"""Unit tests for ExperimentTracker."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.errors import MLflowError


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""

    @patch("src.mlflow_manager.experiment_tracker.mlflow", create=True)
    def _create_tracker(self, mock_mlflow=None):
        """Helper to create a tracker with mocked MLflow."""
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            with patch("src.mlflow_manager.experiment_tracker.ExperimentTracker.__init__", return_value=None):
                from src.mlflow_manager.experiment_tracker import ExperimentTracker

                tracker = ExperimentTracker.__new__(ExperimentTracker)
                tracker._tracking_uri = "http://localhost:5000"
                tracker._experiment_name = "test-experiment"
                tracker._run_id = None
                tracker._active = False
                tracker._mlflow = mock_mlflow
                return tracker, mock_mlflow

    def test_start_run_creates_run(self):
        """Test that start_run creates a new MLflow run and returns run_id."""
        from src.mlflow_manager.experiment_tracker import ExperimentTracker

        tracker, mock_mlflow = self._create_tracker()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value = mock_run

        run_id = tracker.start_run(run_name="test-run")

        assert run_id == "test-run-123"
        assert tracker.is_active is True
        assert tracker.run_id == "test-run-123"
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    def test_start_run_raises_if_already_active(self):
        """Test that start_run raises MLflowError if a run is already active."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True
        tracker._run_id = "existing-run"

        with pytest.raises(MLflowError, match="Cannot start a new run"):
            tracker.start_run()

    def test_end_run(self):
        """Test that end_run ends the active run."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True
        tracker._run_id = "test-run-123"

        tracker.end_run()

        assert tracker.is_active is False
        mock_mlflow.end_run.assert_called_once()

    def test_end_run_raises_if_not_active(self):
        """Test that end_run raises MLflowError if no run is active."""
        tracker, mock_mlflow = self._create_tracker()

        with pytest.raises(MLflowError, match="No active run"):
            tracker.end_run()

    def test_log_hyperparameters(self):
        """Test logging hyperparameters as run parameters."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True

        params = {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 300}
        tracker.log_hyperparameters(params)

        mock_mlflow.log_params.assert_called_once_with(
            {"learning_rate": "0.1", "max_depth": "6", "n_estimators": "300"}
        )

    def test_log_hyperparameters_raises_if_not_active(self):
        """Test that log_hyperparameters raises if no run is active."""
        tracker, mock_mlflow = self._create_tracker()

        with pytest.raises(MLflowError, match="No active run"):
            tracker.log_hyperparameters({"lr": 0.1})

    def test_log_metrics(self):
        """Test logging final metrics."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True

        metrics = {"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.90, "log_loss": 0.15}
        tracker.log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics)

    def test_log_metrics_raises_if_not_active(self):
        """Test that log_metrics raises if no run is active."""
        tracker, mock_mlflow = self._create_tracker()

        with pytest.raises(MLflowError, match="No active run"):
            tracker.log_metrics({"f1": 0.9})

    def test_log_artifact(self):
        """Test logging an artifact file."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"model data")
            artifact_path = f.name

        tracker.log_artifact(artifact_path)
        mock_mlflow.log_artifact.assert_called_once_with(artifact_path)

        Path(artifact_path).unlink()

    def test_log_artifact_raises_if_file_missing(self):
        """Test that log_artifact raises if the file doesn't exist."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True

        with pytest.raises(MLflowError, match="Artifact file not found"):
            tracker.log_artifact("/nonexistent/path/model.pkl")

    def test_log_artifact_raises_if_not_active(self):
        """Test that log_artifact raises if no run is active."""
        tracker, mock_mlflow = self._create_tracker()

        with pytest.raises(MLflowError, match="No active run"):
            tracker.log_artifact("/some/path")

    def test_tag_run(self):
        """Test tagging a run with metadata."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True

        tracker.tag_run(
            dataset_version="v2.0",
            framework_version="0.1.0",
            git_commit="abc123",
        )

        mock_mlflow.set_tags.assert_called_once_with({
            "dataset_version": "v2.0",
            "framework_version": "0.1.0",
            "git_commit": "abc123",
        })

    def test_tag_run_auto_detects_git_commit(self):
        """Test that tag_run attempts git commit detection when not provided."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = MagicMock(
                returncode=0, stdout="deadbeef123\n"
            )
            tracker.tag_run(dataset_version="v1.0", framework_version="0.1.0")

        call_args = mock_mlflow.set_tags.call_args[0][0]
        assert call_args["git_commit"] == "deadbeef123"

    def test_tag_run_git_unavailable(self):
        """Test that tag_run falls back to 'unavailable' if git fails."""
        tracker, mock_mlflow = self._create_tracker()
        tracker._active = True

        with patch("subprocess.run", side_effect=FileNotFoundError):
            tracker.tag_run(dataset_version="v1.0", framework_version="0.1.0")

        call_args = mock_mlflow.set_tags.call_args[0][0]
        assert call_args["git_commit"] == "unavailable"

    def test_tag_run_raises_if_not_active(self):
        """Test that tag_run raises if no run is active."""
        tracker, mock_mlflow = self._create_tracker()

        with pytest.raises(MLflowError, match="No active run"):
            tracker.tag_run()
