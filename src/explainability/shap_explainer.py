"""SHAP explainer for global and local model explanations.

Provides TreeExplainer-based SHAP computation for ensemble models,
generating summary plots, bar plots, waterfall plots, and force plots.
Supports timeout handling, MLflow artifact logging, and JSON-serializable output.
"""

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for plot generation
import matplotlib.pyplot as plt

import shap

from src.models.errors import ExplainabilityError
from src.models.explainability import SHAPExplanation

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainer using TreeExplainer for global and local explanations.

    Computes SHAP values for tree-based ensemble models, generates visualization
    plots (summary, bar, waterfall, force), and returns JSON-serializable results.

    Attributes:
        top_n_features: Number of top features to display in bar plots (1-30).
        timeout_seconds: Maximum computation time before aborting (default 300s).
        output_dir: Directory for persisting plot PNG files.
        dpi: Resolution for saved plot images (minimum 150).
    """

    def __init__(
        self,
        top_n_features: int = 20,
        timeout_seconds: int = 300,
        output_dir: str = "./shap_plots",
        dpi: int = 150,
    ) -> None:
        """Initialize the SHAP explainer.

        Args:
            top_n_features: Number of top features for bar plots (1-30).
            timeout_seconds: Timeout for SHAP computation in seconds.
            output_dir: Directory path for saving plot artifacts.
            dpi: DPI resolution for saved plots (must be >= 150).

        Raises:
            ExplainabilityError: If parameters are out of valid range.
        """
        if not (1 <= top_n_features <= 30):
            raise ExplainabilityError(
                f"Invalid top_n_features: {top_n_features}. Must be between 1 and 30.",
                details={"param": "top_n_features", "value": top_n_features},
            )
        if timeout_seconds <= 0:
            raise ExplainabilityError(
                f"Invalid timeout_seconds: {timeout_seconds}. Must be positive.",
                details={"param": "timeout_seconds", "value": timeout_seconds},
            )
        if dpi < 150:
            raise ExplainabilityError(
                f"Invalid dpi: {dpi}. Must be at least 150.",
                details={"param": "dpi", "value": dpi},
            )

        self.top_n_features = top_n_features
        self.timeout_seconds = timeout_seconds
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self._explainer: shap.TreeExplainer | None = None

    def _ensure_output_dir(self) -> None:
        """Create output directory if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_explainer(self, model: Any) -> shap.TreeExplainer:
        """Create a TreeExplainer for the given model.

        Args:
            model: A tree-based model (XGBoost, LightGBM, CatBoost, or sklearn).

        Returns:
            A configured shap.TreeExplainer instance.
        """
        return shap.TreeExplainer(model)

    def _compute_with_timeout(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a function with a timeout constraint.

        Uses threading to enforce timeout in a cross-platform manner.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value.

        Raises:
            ExplainabilityError: If computation exceeds the timeout.
        """
        result_container: dict[str, Any] = {}
        error_container: dict[str, Any] = {}

        def _target() -> None:
            try:
                result_container["value"] = func(*args, **kwargs)
            except Exception as e:
                error_container["exception"] = e

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            raise ExplainabilityError(
                f"SHAP computation exceeded timeout of {self.timeout_seconds} seconds.",
                details={"timeout_seconds": self.timeout_seconds},
            )

        if "exception" in error_container:
            raise ExplainabilityError(
                f"SHAP computation failed: {error_container['exception']}",
                details={"error": str(error_container["exception"])},
            )

        return result_container["value"]

    def explain_global(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
        mlflow_run: Any | None = None,
    ) -> dict[str, float]:
        """Compute global SHAP values for the full evaluation dataset.

        Generates summary and bar plots showing feature importance ranked
        by mean absolute SHAP value.

        Args:
            model: Trained tree-based model.
            X: Evaluation dataset (n_samples, n_features).
            feature_names: Optional list of feature names. If None, uses
                column names from DataFrame or generates V1..VN names.
            mlflow_run: Optional MLflow run object for artifact logging.

        Returns:
            JSON-serializable dict mapping feature names to mean |SHAP| values,
            sorted by importance descending.

        Raises:
            ExplainabilityError: If computation fails or exceeds timeout.
        """
        self._ensure_output_dir()

        # Resolve feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"V{i+1}" for i in range(X.shape[1])]

        # Convert to numpy if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        logger.info(
            "Computing global SHAP values for %d samples, %d features...",
            X_array.shape[0],
            X_array.shape[1],
        )

        # Compute SHAP values with timeout
        explainer = self._create_explainer(model)

        def _compute_shap() -> np.ndarray:
            return explainer.shap_values(X_array)

        shap_values = self._compute_with_timeout(_compute_shap)

        # Handle binary classification output (may return list of two arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance mapping sorted by importance
        importance_dict: dict[str, float] = {}
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        for idx in sorted_indices:
            importance_dict[feature_names[idx]] = float(mean_abs_shap[idx])

        # Generate summary plot
        self._generate_summary_plot(shap_values, feature_names, X_array)

        # Generate bar plot (top-N features)
        self._generate_bar_plot(mean_abs_shap, feature_names)

        # Log to MLflow if run is provided
        if mlflow_run is not None:
            self._log_to_mlflow(mlflow_run)

        logger.info(
            "Global SHAP explanation complete. Top feature: %s (mean|SHAP|=%.4f)",
            feature_names[sorted_indices[0]],
            mean_abs_shap[sorted_indices[0]],
        )

        return importance_dict

    def explain_local(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        transaction_index: int | None = None,
        transaction_id: str | None = None,
        transaction_ids: list[str] | None = None,
        feature_names: list[str] | None = None,
        mlflow_run: Any | None = None,
    ) -> SHAPExplanation:
        """Compute local SHAP values for a single transaction.

        Generates waterfall and force plots for the specified transaction.

        Args:
            model: Trained tree-based model.
            X: Dataset containing the transaction (n_samples, n_features).
            transaction_index: Direct index into X for the transaction.
            transaction_id: Transaction ID to look up in transaction_ids.
            transaction_ids: List of transaction IDs corresponding to X rows.
            feature_names: Optional list of feature names.
            mlflow_run: Optional MLflow run object for artifact logging.

        Returns:
            SHAPExplanation with feature-level SHAP values, base value,
            and predicted score.

        Raises:
            ExplainabilityError: If transaction not found, computation fails,
                or exceeds timeout.
        """
        self._ensure_output_dir()

        # Resolve transaction index
        idx = self._resolve_transaction_index(
            transaction_index, transaction_id, transaction_ids, X
        )

        # Resolve feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"V{i+1}" for i in range(X.shape[1])]

        # Get the single instance
        if isinstance(X, pd.DataFrame):
            x_instance = X.iloc[idx : idx + 1].values
        else:
            x_instance = X[idx : idx + 1]

        logger.info("Computing local SHAP values for transaction at index %d...", idx)

        # Compute SHAP values with timeout
        explainer = self._create_explainer(model)

        def _compute_shap_local() -> tuple:
            sv = explainer.shap_values(x_instance)
            base = explainer.expected_value
            return sv, base

        shap_result = self._compute_with_timeout(_compute_shap_local)
        shap_values_local, base_value = shap_result

        # Handle binary classification
        if isinstance(shap_values_local, list):
            shap_values_local = shap_values_local[1]
            base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value

        if isinstance(base_value, np.ndarray):
            base_value = float(base_value.item()) if base_value.size == 1 else float(base_value[0])
        else:
            base_value = float(base_value)

        # Flatten to 1D
        shap_values_1d = shap_values_local.flatten()

        # Compute predicted score (base_value + sum of SHAP values)
        predicted_score = float(base_value + np.sum(shap_values_1d))

        # Build feature->SHAP value mapping
        shap_dict: dict[str, float] = {}
        for i, name in enumerate(feature_names):
            shap_dict[name] = float(shap_values_1d[i])

        # Generate waterfall plot
        self._generate_waterfall_plot(
            shap_values_1d, base_value, feature_names, idx
        )

        # Generate force plot
        self._generate_force_plot(
            shap_values_1d, base_value, feature_names, x_instance, idx
        )

        # Log to MLflow if run is provided
        if mlflow_run is not None:
            self._log_to_mlflow(mlflow_run)

        explanation = SHAPExplanation(
            feature_names=feature_names,
            shap_values=shap_dict,
            base_value=base_value,
            predicted_score=predicted_score,
        )

        logger.info(
            "Local SHAP explanation complete for index %d. Predicted score: %.4f",
            idx,
            predicted_score,
        )

        return explanation

    def _resolve_transaction_index(
        self,
        transaction_index: int | None,
        transaction_id: str | None,
        transaction_ids: list[str] | None,
        X: np.ndarray | pd.DataFrame,
    ) -> int:
        """Resolve a transaction to its row index in X.

        Args:
            transaction_index: Direct index (takes precedence).
            transaction_id: Transaction ID to search for.
            transaction_ids: List mapping rows to transaction IDs.
            X: The dataset.

        Returns:
            Integer index into X.

        Raises:
            ExplainabilityError: If transaction cannot be found.
        """
        if transaction_index is not None:
            n_samples = X.shape[0]
            if transaction_index < 0 or transaction_index >= n_samples:
                raise ExplainabilityError(
                    f"Transaction index {transaction_index} out of bounds "
                    f"(dataset has {n_samples} samples).",
                    details={
                        "transaction_index": transaction_index,
                        "n_samples": n_samples,
                    },
                )
            return transaction_index

        if transaction_id is not None:
            if transaction_ids is None:
                raise ExplainabilityError(
                    "transaction_ids list required when looking up by transaction_id.",
                    details={"transaction_id": transaction_id},
                )
            try:
                return transaction_ids.index(transaction_id)
            except ValueError:
                raise ExplainabilityError(
                    f"Transaction ID '{transaction_id}' not found in dataset.",
                    details={"transaction_id": transaction_id},
                )

        raise ExplainabilityError(
            "Either transaction_index or transaction_id must be provided.",
            details={},
        )

    def _generate_summary_plot(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        X: np.ndarray,
    ) -> Path:
        """Generate and save SHAP summary plot.

        Args:
            shap_values: SHAP values array (n_samples, n_features).
            feature_names: List of feature names.
            X: Original feature values for coloring.

        Returns:
            Path to the saved PNG file.
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            features=X,
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
        plot_path = self.output_dir / "shap_summary_plot.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info("Summary plot saved to %s", plot_path)
        return plot_path

    def _generate_bar_plot(
        self,
        mean_abs_shap: np.ndarray,
        feature_names: list[str],
    ) -> Path:
        """Generate and save SHAP bar plot (top-N features).

        Args:
            mean_abs_shap: Mean absolute SHAP values per feature.
            feature_names: List of feature names.

        Returns:
            Path to the saved PNG file.
        """
        # Sort and select top-N
        sorted_indices = np.argsort(mean_abs_shap)[::-1][: self.top_n_features]
        top_names = [feature_names[i] for i in sorted_indices]
        top_values = mean_abs_shap[sorted_indices]

        plt.figure(figsize=(10, max(6, self.top_n_features * 0.4)))
        plt.barh(range(len(top_names)), top_values[::-1], align="center")
        plt.yticks(range(len(top_names)), top_names[::-1])
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"Top-{self.top_n_features} Feature Importance (SHAP)")
        plt.tight_layout()

        plot_path = self.output_dir / "shap_bar_plot.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info("Bar plot saved to %s", plot_path)
        return plot_path

    def _generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: list[str],
        transaction_idx: int,
    ) -> Path:
        """Generate and save SHAP waterfall plot for a single transaction.

        Args:
            shap_values: 1D SHAP values for the transaction.
            base_value: Expected model output.
            feature_names: List of feature names.
            transaction_idx: Index for file naming.

        Returns:
            Path to the saved PNG file.
        """
        # Create a shap.Explanation object for the waterfall plot
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            feature_names=feature_names,
        )

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plot_path = self.output_dir / f"shap_waterfall_tx_{transaction_idx}.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info("Waterfall plot saved to %s", plot_path)
        return plot_path

    def _generate_force_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: list[str],
        x_instance: np.ndarray,
        transaction_idx: int,
    ) -> Path:
        """Generate and save SHAP force plot for a single transaction.

        Args:
            shap_values: 1D SHAP values for the transaction.
            base_value: Expected model output.
            feature_names: List of feature names.
            x_instance: Feature values for the transaction.
            transaction_idx: Index for file naming.

        Returns:
            Path to the saved PNG file.
        """
        force_plot = shap.force_plot(
            base_value,
            shap_values,
            features=x_instance.flatten(),
            feature_names=feature_names,
            show=False,
            matplotlib=True,
        )
        plot_path = self.output_dir / f"shap_force_tx_{transaction_idx}.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info("Force plot saved to %s", plot_path)
        return plot_path

    def _log_to_mlflow(self, mlflow_run: Any) -> None:
        """Log all plot artifacts in output_dir to MLflow.

        Args:
            mlflow_run: MLflow run context or client instance with
                log_artifacts method.
        """
        try:
            import mlflow

            mlflow.log_artifacts(str(self.output_dir), artifact_path="shap_plots")
            logger.info("SHAP plots logged to MLflow.")
        except Exception as e:
            logger.warning("Failed to log SHAP plots to MLflow: %s", e)

    def to_json(self, explanation: SHAPExplanation) -> dict[str, Any]:
        """Convert a SHAPExplanation to a JSON-serializable dictionary.

        Args:
            explanation: A SHAPExplanation dataclass instance.

        Returns:
            Dictionary with feature_names, shap_values, base_value,
            and predicted_score.
        """
        return {
            "feature_names": explanation.feature_names,
            "shap_values": explanation.shap_values,
            "base_value": explanation.base_value,
            "predicted_score": explanation.predicted_score,
        }
