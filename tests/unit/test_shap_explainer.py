"""Unit tests for the SHAP explainer module.

Mocks SHAP internals for speed while validating the SHAPExplainer class behavior
including global/local explanations, timeout handling, error cases, and plot generation.
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import SHAPExplainer
from src.models.errors import ExplainabilityError
from src.models.explainability import SHAPExplanation


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def feature_names():
    """Standard CCFD feature names."""
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@pytest.fixture
def sample_data(feature_names):
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = len(feature_names)
    X = np.random.randn(n_samples, n_features)
    return X, feature_names


@pytest.fixture
def sample_df(sample_data):
    """Create a pandas DataFrame from sample data."""
    X, names = sample_data
    return pd.DataFrame(X, columns=names)


@pytest.fixture
def mock_model():
    """Create a mock tree-based model."""
    model = MagicMock()
    model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
    return model


@pytest.fixture
def mock_shap_values(sample_data):
    """Generate mock SHAP values matching sample data dimensions."""
    X, _ = sample_data
    np.random.seed(123)
    return np.random.randn(*X.shape) * 0.1


@pytest.fixture
def explainer(tmp_path):
    """Create a SHAPExplainer with a temporary output directory."""
    return SHAPExplainer(
        top_n_features=10,
        timeout_seconds=60,
        output_dir=str(tmp_path / "shap_plots"),
        dpi=150,
    )


# ─── Initialization Tests ────────────────────────────────────────────────────


class TestSHAPExplainerInit:
    """Tests for SHAPExplainer initialization and parameter validation."""

    def test_valid_initialization(self, tmp_path):
        """SHAPExplainer initializes with valid parameters."""
        explainer = SHAPExplainer(
            top_n_features=20,
            timeout_seconds=300,
            output_dir=str(tmp_path / "plots"),
            dpi=200,
        )
        assert explainer.top_n_features == 20
        assert explainer.timeout_seconds == 300
        assert explainer.dpi == 200

    def test_invalid_top_n_features_zero(self, tmp_path):
        """Raises ExplainabilityError when top_n_features is 0."""
        with pytest.raises(ExplainabilityError, match="top_n_features"):
            SHAPExplainer(top_n_features=0, output_dir=str(tmp_path))

    def test_invalid_top_n_features_exceeds_max(self, tmp_path):
        """Raises ExplainabilityError when top_n_features exceeds 30."""
        with pytest.raises(ExplainabilityError, match="top_n_features"):
            SHAPExplainer(top_n_features=31, output_dir=str(tmp_path))

    def test_invalid_timeout_zero(self, tmp_path):
        """Raises ExplainabilityError when timeout_seconds is 0."""
        with pytest.raises(ExplainabilityError, match="timeout_seconds"):
            SHAPExplainer(timeout_seconds=0, output_dir=str(tmp_path))

    def test_invalid_timeout_negative(self, tmp_path):
        """Raises ExplainabilityError when timeout_seconds is negative."""
        with pytest.raises(ExplainabilityError, match="timeout_seconds"):
            SHAPExplainer(timeout_seconds=-10, output_dir=str(tmp_path))

    def test_invalid_dpi_below_minimum(self, tmp_path):
        """Raises ExplainabilityError when dpi is below 150."""
        with pytest.raises(ExplainabilityError, match="dpi"):
            SHAPExplainer(dpi=100, output_dir=str(tmp_path))

    def test_boundary_top_n_features_min(self, tmp_path):
        """top_n_features=1 is valid."""
        explainer = SHAPExplainer(top_n_features=1, output_dir=str(tmp_path))
        assert explainer.top_n_features == 1

    def test_boundary_top_n_features_max(self, tmp_path):
        """top_n_features=30 is valid."""
        explainer = SHAPExplainer(top_n_features=30, output_dir=str(tmp_path))
        assert explainer.top_n_features == 30


# ─── Global Explanation Tests ─────────────────────────────────────────────────


class TestExplainGlobal:
    """Tests for global SHAP explanation computation."""

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_global_explanation_returns_dict(
        self, mock_summary, mock_tree_explainer, explainer, sample_data, mock_model
    ):
        """explain_global returns a dict mapping feature names to mean |SHAP| values."""
        X, feature_names = sample_data
        mock_shap = np.random.randn(*X.shape) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_global(mock_model, X, feature_names=feature_names)

        assert isinstance(result, dict)
        assert len(result) == len(feature_names)
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, float) for v in result.values())

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_global_explanation_sorted_by_importance(
        self, mock_summary, mock_tree_explainer, explainer, sample_data, mock_model
    ):
        """Global explanation dict is sorted by mean |SHAP| descending."""
        X, feature_names = sample_data
        # Create SHAP values with known ordering
        mock_shap = np.zeros_like(X)
        mock_shap[:, 0] = 1.0  # First feature has highest importance
        mock_shap[:, 1] = 0.5  # Second has less
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_global(mock_model, X, feature_names=feature_names)

        values = list(result.values())
        assert values[0] >= values[1]  # Sorted descending

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_global_explanation_handles_binary_classification_list(
        self, mock_summary, mock_tree_explainer, explainer, sample_data, mock_model
    ):
        """Handles SHAP returning a list (binary classification) by using positive class."""
        X, feature_names = sample_data
        # Binary classification returns list of 2 arrays
        shap_class_0 = np.random.randn(*X.shape) * 0.05
        shap_class_1 = np.random.randn(*X.shape) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = [shap_class_0, shap_class_1]
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_global(mock_model, X, feature_names=feature_names)

        assert isinstance(result, dict)
        assert len(result) == len(feature_names)

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_global_explanation_with_dataframe(
        self, mock_summary, mock_tree_explainer, explainer, sample_df, mock_model
    ):
        """explain_global works with pandas DataFrame input."""
        mock_shap = np.random.randn(*sample_df.shape) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_global(mock_model, sample_df)

        assert isinstance(result, dict)
        # Should use DataFrame column names
        assert list(sample_df.columns)[0] in result

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_global_explanation_generates_plots(
        self, mock_summary, mock_tree_explainer, explainer, sample_data, mock_model
    ):
        """explain_global generates summary and bar plot PNG files."""
        X, feature_names = sample_data
        mock_shap = np.random.randn(*X.shape) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer.explain_global(mock_model, X, feature_names=feature_names)

        # Bar plot should be generated (summary_plot is mocked)
        bar_plot_path = explainer.output_dir / "shap_bar_plot.png"
        assert bar_plot_path.exists()

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_global_explanation_auto_feature_names(
        self, mock_summary, mock_tree_explainer, explainer, mock_model
    ):
        """Auto-generates feature names V1..VN when none provided."""
        X = np.random.randn(20, 5)
        mock_shap = np.random.randn(20, 5) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_global(mock_model, X)

        assert "V1" in result
        assert "V5" in result


# ─── Local Explanation Tests ──────────────────────────────────────────────────


class TestExplainLocal:
    """Tests for local SHAP explanation computation."""

    @patch("src.explainability.shap_explainer.shap.force_plot")
    @patch("src.explainability.shap_explainer.shap.plots.waterfall")
    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_local_explanation_returns_shap_explanation(
        self,
        mock_tree_explainer,
        mock_waterfall,
        mock_force,
        explainer,
        sample_data,
        mock_model,
    ):
        """explain_local returns a SHAPExplanation dataclass."""
        X, feature_names = sample_data
        n_features = X.shape[1]
        mock_shap_vals = np.random.randn(1, n_features) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap_vals
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_local(
            mock_model, X, transaction_index=0, feature_names=feature_names
        )

        assert isinstance(result, SHAPExplanation)
        assert result.feature_names == feature_names
        assert len(result.shap_values) == n_features
        assert isinstance(result.base_value, float)
        assert isinstance(result.predicted_score, float)

    @patch("src.explainability.shap_explainer.shap.force_plot")
    @patch("src.explainability.shap_explainer.shap.plots.waterfall")
    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_local_explanation_shap_values_json_serializable(
        self,
        mock_tree_explainer,
        mock_waterfall,
        mock_force,
        explainer,
        sample_data,
        mock_model,
    ):
        """SHAP values in explanation are JSON-serializable floats."""
        X, feature_names = sample_data
        n_features = X.shape[1]
        mock_shap_vals = np.random.randn(1, n_features) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap_vals
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_local(
            mock_model, X, transaction_index=0, feature_names=feature_names
        )

        # All values should be native Python floats
        for v in result.shap_values.values():
            assert isinstance(v, float)

    @patch("src.explainability.shap_explainer.shap.force_plot")
    @patch("src.explainability.shap_explainer.shap.plots.waterfall")
    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_local_explanation_by_transaction_id(
        self,
        mock_tree_explainer,
        mock_waterfall,
        mock_force,
        explainer,
        sample_data,
        mock_model,
    ):
        """explain_local resolves transaction by ID."""
        X, feature_names = sample_data
        n_features = X.shape[1]
        transaction_ids = [f"tx_{i}" for i in range(X.shape[0])]
        mock_shap_vals = np.random.randn(1, n_features) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap_vals
        mock_explainer_instance.expected_value = 0.3
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_local(
            mock_model,
            X,
            transaction_id="tx_5",
            transaction_ids=transaction_ids,
            feature_names=feature_names,
        )

        assert isinstance(result, SHAPExplanation)

    @patch("src.explainability.shap_explainer.shap.force_plot")
    @patch("src.explainability.shap_explainer.shap.plots.waterfall")
    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_local_explanation_handles_binary_list(
        self,
        mock_tree_explainer,
        mock_waterfall,
        mock_force,
        explainer,
        sample_data,
        mock_model,
    ):
        """Handles binary classification list output from shap_values."""
        X, feature_names = sample_data
        n_features = X.shape[1]
        shap_class_0 = np.random.randn(1, n_features) * 0.05
        shap_class_1 = np.random.randn(1, n_features) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = [shap_class_0, shap_class_1]
        mock_explainer_instance.expected_value = [0.4, 0.6]
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_local(
            mock_model, X, transaction_index=0, feature_names=feature_names
        )

        assert isinstance(result, SHAPExplanation)
        assert result.base_value == 0.6  # Positive class base value

    @patch("src.explainability.shap_explainer.shap.force_plot")
    @patch("src.explainability.shap_explainer.shap.plots.waterfall")
    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_local_explanation_predicted_score_correct(
        self,
        mock_tree_explainer,
        mock_waterfall,
        mock_force,
        explainer,
        sample_data,
        mock_model,
    ):
        """Predicted score equals base_value + sum(shap_values)."""
        X, feature_names = sample_data
        n_features = X.shape[1]
        shap_vals = np.array([[0.1, -0.05, 0.2] + [0.0] * (n_features - 3)])
        base_value = 0.5
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = shap_vals
        mock_explainer_instance.expected_value = base_value
        mock_tree_explainer.return_value = mock_explainer_instance

        result = explainer.explain_local(
            mock_model, X, transaction_index=0, feature_names=feature_names
        )

        expected_score = base_value + shap_vals.sum()
        assert abs(result.predicted_score - expected_score) < 1e-10


# ─── Transaction Resolution Tests ────────────────────────────────────────────


class TestTransactionResolution:
    """Tests for transaction ID/index resolution."""

    def test_transaction_not_found_by_id(self, explainer, sample_data, mock_model):
        """Raises ExplainabilityError when transaction ID not in list."""
        X, feature_names = sample_data
        transaction_ids = [f"tx_{i}" for i in range(X.shape[0])]

        with pytest.raises(ExplainabilityError, match="not found"):
            explainer.explain_local(
                mock_model,
                X,
                transaction_id="nonexistent_tx",
                transaction_ids=transaction_ids,
                feature_names=feature_names,
            )

    def test_transaction_id_without_ids_list(self, explainer, sample_data, mock_model):
        """Raises ExplainabilityError when transaction_ids list is not provided."""
        X, feature_names = sample_data

        with pytest.raises(ExplainabilityError, match="transaction_ids list required"):
            explainer.explain_local(
                mock_model,
                X,
                transaction_id="tx_0",
                feature_names=feature_names,
            )

    def test_transaction_index_out_of_bounds(self, explainer, sample_data, mock_model):
        """Raises ExplainabilityError when index is out of bounds."""
        X, feature_names = sample_data

        with pytest.raises(ExplainabilityError, match="out of bounds"):
            explainer.explain_local(
                mock_model,
                X,
                transaction_index=999,
                feature_names=feature_names,
            )

    def test_no_transaction_identifier(self, explainer, sample_data, mock_model):
        """Raises ExplainabilityError when neither index nor ID provided."""
        X, feature_names = sample_data

        with pytest.raises(ExplainabilityError, match="must be provided"):
            explainer.explain_local(
                mock_model,
                X,
                feature_names=feature_names,
            )


# ─── Timeout Tests ────────────────────────────────────────────────────────────


class TestTimeout:
    """Tests for timeout handling in SHAP computation."""

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_global_timeout_raises_error(
        self, mock_tree_explainer, tmp_path, sample_data, mock_model
    ):
        """Raises ExplainabilityError when global computation exceeds timeout."""
        X, feature_names = sample_data
        explainer = SHAPExplainer(
            timeout_seconds=1,
            output_dir=str(tmp_path / "plots"),
        )

        def slow_shap_values(data):
            time.sleep(5)
            return np.zeros_like(data)

        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.side_effect = slow_shap_values
        mock_tree_explainer.return_value = mock_explainer_instance

        with pytest.raises(ExplainabilityError, match="exceeded timeout"):
            explainer.explain_global(mock_model, X, feature_names=feature_names)

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_local_timeout_raises_error(
        self, mock_tree_explainer, tmp_path, sample_data, mock_model
    ):
        """Raises ExplainabilityError when local computation exceeds timeout."""
        X, feature_names = sample_data
        explainer = SHAPExplainer(
            timeout_seconds=1,
            output_dir=str(tmp_path / "plots"),
        )

        def slow_shap_values(data):
            time.sleep(5)
            return np.zeros((1, X.shape[1]))

        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.side_effect = slow_shap_values
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        with pytest.raises(ExplainabilityError, match="exceeded timeout"):
            explainer.explain_local(
                mock_model, X, transaction_index=0, feature_names=feature_names
            )


# ─── MLflow Integration Tests ────────────────────────────────────────────────


class TestMLflowLogging:
    """Tests for MLflow artifact logging."""

    @patch("mlflow.log_artifacts")
    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_mlflow_logging_called(
        self,
        mock_summary,
        mock_tree_explainer,
        mock_log_artifacts,
        explainer,
        sample_data,
        mock_model,
    ):
        """MLflow log_artifacts is called when mlflow_run is provided."""
        X, feature_names = sample_data
        mock_shap = np.random.randn(*X.shape) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        mlflow_run = MagicMock()
        explainer.explain_global(
            mock_model, X, feature_names=feature_names, mlflow_run=mlflow_run
        )

        mock_log_artifacts.assert_called_once()

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_no_mlflow_logging_when_run_is_none(
        self,
        mock_summary,
        mock_tree_explainer,
        explainer,
        sample_data,
        mock_model,
    ):
        """No MLflow logging when mlflow_run is None (no import error)."""
        X, feature_names = sample_data
        mock_shap = np.random.randn(*X.shape) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        # Should not raise even without MLflow
        result = explainer.explain_global(mock_model, X, feature_names=feature_names)
        assert isinstance(result, dict)


# ─── JSON Serialization Tests ─────────────────────────────────────────────────


class TestToJson:
    """Tests for JSON conversion of SHAPExplanation."""

    def test_to_json_returns_dict(self, explainer):
        """to_json returns a JSON-serializable dictionary."""
        explanation = SHAPExplanation(
            feature_names=["V1", "V2", "V3"],
            shap_values={"V1": 0.3, "V2": -0.1, "V3": 0.05},
            base_value=0.5,
            predicted_score=0.75,
        )

        result = explainer.to_json(explanation)

        assert result["feature_names"] == ["V1", "V2", "V3"]
        assert result["shap_values"] == {"V1": 0.3, "V2": -0.1, "V3": 0.05}
        assert result["base_value"] == 0.5
        assert result["predicted_score"] == 0.75

    def test_to_json_serializable(self, explainer):
        """Verify the result can be serialized to JSON."""
        import json

        explanation = SHAPExplanation(
            feature_names=["V1", "V2"],
            shap_values={"V1": 0.123, "V2": -0.456},
            base_value=0.5,
            predicted_score=0.167,
        )

        result = explainer.to_json(explanation)
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


# ─── Plot Generation Tests ────────────────────────────────────────────────────


class TestPlotGeneration:
    """Tests for plot file generation."""

    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    @patch("src.explainability.shap_explainer.shap.summary_plot")
    def test_bar_plot_created(
        self, mock_summary, mock_tree_explainer, explainer, sample_data, mock_model
    ):
        """Bar plot PNG file is created in output directory."""
        X, feature_names = sample_data
        mock_shap = np.random.randn(*X.shape) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer.explain_global(mock_model, X, feature_names=feature_names)

        bar_path = explainer.output_dir / "shap_bar_plot.png"
        assert bar_path.exists()
        # Check file is not empty
        assert bar_path.stat().st_size > 0

    @patch("src.explainability.shap_explainer.shap.force_plot")
    @patch("src.explainability.shap_explainer.shap.plots.waterfall")
    @patch("src.explainability.shap_explainer.shap.TreeExplainer")
    def test_waterfall_plot_created(
        self,
        mock_tree_explainer,
        mock_waterfall,
        mock_force,
        explainer,
        sample_data,
        mock_model,
    ):
        """Waterfall plot PNG file is created for local explanation."""
        X, feature_names = sample_data
        n_features = X.shape[1]
        mock_shap_vals = np.random.randn(1, n_features) * 0.1
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = mock_shap_vals
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer.explain_local(
            mock_model, X, transaction_index=0, feature_names=feature_names
        )

        waterfall_path = explainer.output_dir / "shap_waterfall_tx_0.png"
        # File would be created by the actual waterfall call,
        # but since we mock the waterfall plot, verify the method was called
        mock_waterfall.assert_called_once()

    def test_output_directory_created(self, tmp_path, mock_model):
        """Output directory is created if it doesn't exist."""
        output_path = tmp_path / "nested" / "dir" / "plots"
        explainer = SHAPExplainer(output_dir=str(output_path))
        explainer._ensure_output_dir()
        assert output_path.exists()
