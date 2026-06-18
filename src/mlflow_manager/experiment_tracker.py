"""MLflow experiment tracking - log parameters, metrics, artifacts, and run tags.

Provides ExperimentTracker class that wraps MLflow's tracking API to log
hyperparameters, evaluation metrics, model artifacts, and run metadata tags
for each training run.

Validates: Requirements 17.1, 17.2, 17.3, 17.4
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from src.models.errors import MLflowError

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks ML experiments by logging parameters, metrics, artifacts, and tags to MLflow.

    Usage:
        tracker = ExperimentTracker(tracking_uri="http://localhost:5000", experiment_name="fraud-detection")
        run_id = tracker.start_run(run_name="xgboost-v1")
        tracker.log_hyperparameters({"learning_rate": 0.1, "max_depth": 6})
        tracker.log_metrics({"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.90, "log_loss": 0.15})
        tracker.log_artifact("/path/to/model.pkl")
        tracker.tag_run(dataset_version="v1.0", framework_version="0.1.0")
        tracker.end_run()
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "fraud-detection",
    ) -> None:
        """Initialize the ExperimentTracker.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the MLflow experiment.
        """
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id: str | None = None
        self._active = False

        try:
            import mlflow

            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._mlflow = mlflow
        except Exception as e:
            raise MLflowError(
                f"Failed to initialize MLflow tracking: {e}",
                details={"tracking_uri": tracking_uri, "experiment_name": experiment_name},
            )

    @property
    def run_id(self) -> str | None:
        """Return the current active run ID, or None if no run is active."""
        return self._run_id

    @property
    def is_active(self) -> bool:
        """Return True if a run is currently active."""
        return self._active

    def start_run(self, run_name: str | None = None) -> str:
        """Start a new MLflow run.

        Args:
            run_name: Optional name for the run.

        Returns:
            The run ID string.

        Raises:
            MLflowError: If a run is already active or MLflow fails.
        """
        if self._active:
            raise MLflowError(
                "Cannot start a new run while another is active",
                details={"current_run_id": self._run_id},
            )

        try:
            run = self._mlflow.start_run(run_name=run_name)
            self._run_id = run.info.run_id
            self._active = True
            logger.info(f"Started MLflow run: {self._run_id} (name={run_name})")
            return self._run_id
        except Exception as e:
            raise MLflowError(
                f"Failed to start MLflow run: {e}",
                details={"run_name": run_name},
            )

    def end_run(self) -> None:
        """End the current active MLflow run.

        Raises:
            MLflowError: If no run is active.
        """
        if not self._active:
            raise MLflowError("No active run to end")

        try:
            self._mlflow.end_run()
            logger.info(f"Ended MLflow run: {self._run_id}")
            self._active = False
        except Exception as e:
            raise MLflowError(
                f"Failed to end MLflow run: {e}",
                details={"run_id": self._run_id},
            )

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters as run parameters.

        Args:
            params: Dictionary of hyperparameter name-value pairs.

        Raises:
            MLflowError: If no run is active or logging fails.
        """
        if not self._active:
            raise MLflowError("No active run. Call start_run() first.")

        try:
            # MLflow params must be strings or have str representation
            str_params = {k: str(v) for k, v in params.items()}
            self._mlflow.log_params(str_params)
            logger.info(f"Logged {len(params)} hyperparameters")
        except Exception as e:
            raise MLflowError(
                f"Failed to log hyperparameters: {e}",
                details={"param_count": len(params)},
            )

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log final metrics: F1, AUC-ROC, precision, recall, log_loss.

        Args:
            metrics: Dictionary of metric name-value pairs.

        Raises:
            MLflowError: If no run is active or logging fails.
        """
        if not self._active:
            raise MLflowError("No active run. Call start_run() first.")

        try:
            self._mlflow.log_metrics(metrics)
            logger.info(f"Logged metrics: {metrics}")
        except Exception as e:
            raise MLflowError(
                f"Failed to log metrics: {e}",
                details={"metrics": metrics},
            )

    def log_artifact(self, artifact_path: str, artifact_dir: str | None = None) -> None:
        """Log a model artifact, preprocessing pipeline, or feature importance plot.

        Args:
            artifact_path: Local path to the artifact file.
            artifact_dir: Optional subdirectory in the MLflow artifact store.

        Raises:
            MLflowError: If no run is active, file doesn't exist, or logging fails.
        """
        if not self._active:
            raise MLflowError("No active run. Call start_run() first.")

        path = Path(artifact_path)
        if not path.exists():
            raise MLflowError(
                f"Artifact file not found: {artifact_path}",
                details={"artifact_path": artifact_path},
            )

        try:
            if artifact_dir:
                self._mlflow.log_artifact(str(path), artifact_dir)
            else:
                self._mlflow.log_artifact(str(path))
            logger.info(f"Logged artifact: {artifact_path}")
        except Exception as e:
            raise MLflowError(
                f"Failed to log artifact: {e}",
                details={"artifact_path": artifact_path},
            )

    def tag_run(
        self,
        dataset_version: str = "unknown",
        framework_version: str = "unknown",
        git_commit: str | None = None,
    ) -> None:
        """Tag the current run with metadata.

        Tags include dataset_version, framework_version, and git_commit.
        If git_commit is not provided, attempts to detect from git; falls back to "unavailable".

        Args:
            dataset_version: Version of the dataset used for training.
            framework_version: Version of the framework.
            git_commit: Git commit hash. Auto-detected if None.

        Raises:
            MLflowError: If no run is active or tagging fails.
        """
        if not self._active:
            raise MLflowError("No active run. Call start_run() first.")

        if git_commit is None:
            git_commit = self._get_git_commit()

        tags = {
            "dataset_version": dataset_version,
            "framework_version": framework_version,
            "git_commit": git_commit,
        }

        try:
            self._mlflow.set_tags(tags)
            logger.info(f"Tagged run with: {tags}")
        except Exception as e:
            raise MLflowError(
                f"Failed to tag run: {e}",
                details={"tags": tags},
            )

    def _get_git_commit(self) -> str:
        """Attempt to get the current git commit hash.

        Returns:
            The git commit hash string, or "unavailable" if detection fails.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return "unavailable"
