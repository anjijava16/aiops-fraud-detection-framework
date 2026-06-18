"""Unit tests for the LIMEExplainer class."""

import numpy as np
import pytest

from src.explainability.lime_explainer import (
    DEFAULT_PERTURBATION_SAMPLES,
    DEFAULT_TOP_N,
    LOW_FIDELITY_THRESHOLD,
    MAX_PERTURBATION_SAMPLES,
    MAX_TOP_N,
    MIN_PERTURBATION_SAMPLES,
    MIN_TOP_N,
    LIMEExplainer,
)
from src.models.errors import ExplainabilityError
from src.models.explainability import LIMEExplanation


def _make_training_data(n_samples: int = 100, n_features: int = 30) -> np.ndarray:
    """Create synthetic training data for LIME background."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_samples, n_features))


def _make_feature_names(n_features: int = 30) -> list[str]:
    """Create feature names matching CCFD dataset."""
    names = ["Time"]
    names.extend([f"V{i}" for i in range(1, n_features - 1)])
    names.append("Amount")
    return names


def _mock_predict_fn(X: np.ndarray) -> np.ndarray:
    """Mock predict function returning two-class probabilities.

    Uses a simple logistic-like function based on the first feature.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Simple: probability based on sum of first 3 features
    scores = 1.0 / (1.0 + np.exp(-X[:, 0] - 0.5 * X[:, 1]))
    # Return (n_samples, 2) array: [P(not_fraud), P(fraud)]
    return np.column_stack([1 - scores, scores])


def _mock_low_fidelity_predict_fn(X: np.ndarray) -> np.ndarray:
    """Mock predict function designed to produce low fidelity LIME explanations.

    Uses random noise to make the surrogate model fit poorly.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    rng = np.random.default_rng(int(X.sum() * 100) % 2**31)
    scores = rng.random(X.shape[0])
    return np.column_stack([1 - scores, scores])


class TestLIMEExplainerInit:
    """Tests for LIMEExplainer initialization and parameter validation."""

    def test_default_initialization(self):
        """Explainer should initialize with default parameters."""
        data = _make_training_data()
        names = _make_feature_names()
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
        )
        assert explainer.num_samples == DEFAULT_PERTURBATION_SAMPLES
        assert explainer.top_n == DEFAULT_TOP_N

    def test_custom_num_samples(self):
        """Custom num_samples within range should be accepted."""
        data = _make_training_data()
        names = _make_feature_names()
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
            num_samples=2000,
        )
        assert explainer.num_samples == 2000

    def test_custom_top_n(self):
        """Custom top_n within range should be accepted."""
        data = _make_training_data()
        names = _make_feature_names()
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
            top_n=5,
        )
        assert explainer.top_n == 5

    def test_num_samples_below_min_raises(self):
        """num_samples below minimum should raise ExplainabilityError."""
        data = _make_training_data()
        names = _make_feature_names()
        with pytest.raises(ExplainabilityError, match="num_samples must be between"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
                num_samples=999,
            )

    def test_num_samples_above_max_raises(self):
        """num_samples above maximum should raise ExplainabilityError."""
        data = _make_training_data()
        names = _make_feature_names()
        with pytest.raises(ExplainabilityError, match="num_samples must be between"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
                num_samples=50001,
            )

    def test_num_samples_at_min_boundary(self):
        """num_samples at minimum boundary should be accepted."""
        data = _make_training_data()
        names = _make_feature_names()
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
            num_samples=MIN_PERTURBATION_SAMPLES,
        )
        assert explainer.num_samples == MIN_PERTURBATION_SAMPLES

    def test_num_samples_at_max_boundary(self):
        """num_samples at maximum boundary should be accepted."""
        data = _make_training_data()
        names = _make_feature_names()
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
            num_samples=MAX_PERTURBATION_SAMPLES,
        )
        assert explainer.num_samples == MAX_PERTURBATION_SAMPLES

    def test_top_n_below_min_raises(self):
        """top_n below minimum should raise ExplainabilityError."""
        data = _make_training_data()
        names = _make_feature_names()
        with pytest.raises(ExplainabilityError, match="top_n must be between"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
                top_n=0,
            )

    def test_top_n_above_max_raises(self):
        """top_n above maximum should raise ExplainabilityError."""
        data = _make_training_data()
        names = _make_feature_names()
        with pytest.raises(ExplainabilityError, match="top_n must be between"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
                top_n=31,
            )

    def test_top_n_at_min_boundary(self):
        """top_n at minimum boundary should be accepted."""
        data = _make_training_data()
        names = _make_feature_names()
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
            top_n=MIN_TOP_N,
        )
        assert explainer.top_n == MIN_TOP_N

    def test_top_n_at_max_boundary(self):
        """top_n at maximum boundary should be accepted."""
        data = _make_training_data()
        names = _make_feature_names()
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
            top_n=MAX_TOP_N,
        )
        assert explainer.top_n == MAX_TOP_N

    def test_non_integer_num_samples_raises(self):
        """Non-integer num_samples should raise ExplainabilityError."""
        data = _make_training_data()
        names = _make_feature_names()
        with pytest.raises(ExplainabilityError, match="num_samples must be an integer"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
                num_samples=5000.0,
            )

    def test_non_integer_top_n_raises(self):
        """Non-integer top_n should raise ExplainabilityError."""
        data = _make_training_data()
        names = _make_feature_names()
        with pytest.raises(ExplainabilityError, match="top_n must be an integer"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
                top_n=10.0,
            )

    def test_1d_training_data_raises(self):
        """1D training data should raise ExplainabilityError."""
        data = np.array([1.0, 2.0, 3.0])
        names = ["a", "b", "c"]
        with pytest.raises(ExplainabilityError, match="2-dimensional"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
            )

    def test_mismatched_feature_names_raises(self):
        """Feature names length mismatch should raise ExplainabilityError."""
        data = _make_training_data(n_features=30)
        names = ["a", "b", "c"]  # Only 3 names for 30 features
        with pytest.raises(ExplainabilityError, match="Feature names length"):
            LIMEExplainer(
                training_data=data,
                feature_names=names,
                predict_fn=_mock_predict_fn,
            )


class TestLIMEExplainerExplain:
    """Tests for the explain method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data = _make_training_data(n_samples=50, n_features=10)
        self.names = [f"feature_{i}" for i in range(10)]
        self.explainer = LIMEExplainer(
            training_data=self.data,
            feature_names=self.names,
            predict_fn=_mock_predict_fn,
            num_samples=1000,
            top_n=5,
        )

    def test_explain_returns_lime_explanation(self):
        """explain should return a LIMEExplanation dataclass."""
        instance = self.data[0]
        result = self.explainer.explain(instance)
        assert isinstance(result, LIMEExplanation)

    def test_explain_feature_contributions_structure(self):
        """Feature contributions should be list of (str, float) tuples."""
        instance = self.data[0]
        result = self.explainer.explain(instance)
        assert isinstance(result.feature_contributions, list)
        assert len(result.feature_contributions) <= 5
        for item in result.feature_contributions:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_explain_fidelity_score_is_float(self):
        """Fidelity score should be a float value."""
        instance = self.data[0]
        result = self.explainer.explain(instance)
        assert isinstance(result.fidelity_score, float)

    def test_explain_num_perturbation_samples_recorded(self):
        """Number of perturbation samples should be recorded in output."""
        instance = self.data[0]
        result = self.explainer.explain(instance)
        assert result.num_perturbation_samples == 1000

    def test_explain_with_override_num_samples(self):
        """Override num_samples should be used and recorded."""
        instance = self.data[0]
        result = self.explainer.explain(instance, num_samples=2000)
        assert result.num_perturbation_samples == 2000

    def test_explain_with_override_top_n(self):
        """Override top_n should limit feature contributions."""
        instance = self.data[0]
        result = self.explainer.explain(instance, top_n=3)
        assert len(result.feature_contributions) <= 3

    def test_explain_invalid_override_num_samples_raises(self):
        """Invalid override num_samples should raise ExplainabilityError."""
        instance = self.data[0]
        with pytest.raises(ExplainabilityError, match="num_samples must be between"):
            self.explainer.explain(instance, num_samples=500)

    def test_explain_invalid_override_top_n_raises(self):
        """Invalid override top_n should raise ExplainabilityError."""
        instance = self.data[0]
        with pytest.raises(ExplainabilityError, match="top_n must be between"):
            self.explainer.explain(instance, top_n=0)

    def test_explain_2d_instance_raises(self):
        """2D instance should raise ExplainabilityError."""
        instance = self.data[:2]  # 2D array
        with pytest.raises(ExplainabilityError, match="1-dimensional"):
            self.explainer.explain(instance)

    def test_explain_wrong_feature_count_raises(self):
        """Instance with wrong feature count should raise ExplainabilityError."""
        instance = np.array([1.0, 2.0, 3.0])  # Only 3 features, expect 10
        with pytest.raises(ExplainabilityError, match="feature count"):
            self.explainer.explain(instance)

    def test_explain_is_low_confidence_boolean(self):
        """is_low_confidence should be a boolean."""
        instance = self.data[0]
        result = self.explainer.explain(instance)
        assert isinstance(result.is_low_confidence, bool)

    def test_explain_high_fidelity_not_low_confidence(self):
        """High fidelity (>= 0.6) should not be flagged as low confidence."""
        instance = self.data[0]
        result = self.explainer.explain(instance)
        if result.fidelity_score >= LOW_FIDELITY_THRESHOLD:
            assert result.is_low_confidence is False

    def test_explain_transaction_id_in_error_details(self):
        """Transaction ID should appear in error details on failure."""
        instance = np.array([1.0, 2.0, 3.0])  # Wrong size
        with pytest.raises(ExplainabilityError) as exc_info:
            self.explainer.explain(instance, transaction_id="tx-123")
        assert exc_info.value.details["transaction_id"] == "tx-123"


class TestLIMEExplainerExplainById:
    """Tests for the explain_by_id method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data = _make_training_data(n_samples=50, n_features=10)
        self.names = [f"feature_{i}" for i in range(10)]
        self.explainer = LIMEExplainer(
            training_data=self.data,
            feature_names=self.names,
            predict_fn=_mock_predict_fn,
            num_samples=1000,
            top_n=5,
        )
        # Create a dataset mapping
        self.dataset = {
            "tx-001": self.data[0],
            "tx-002": self.data[1],
            "tx-003": self.data[2],
        }

    def test_explain_by_id_returns_explanation(self):
        """explain_by_id should return a LIMEExplanation for valid ID."""
        result = self.explainer.explain_by_id("tx-001", self.dataset)
        assert isinstance(result, LIMEExplanation)

    def test_explain_by_id_not_found_raises(self):
        """Non-existent transaction ID should raise ExplainabilityError."""
        with pytest.raises(ExplainabilityError, match="Transaction not found"):
            self.explainer.explain_by_id("tx-999", self.dataset)

    def test_explain_by_id_error_contains_transaction_id(self):
        """Error details should contain the transaction ID."""
        with pytest.raises(ExplainabilityError) as exc_info:
            self.explainer.explain_by_id("tx-999", self.dataset)
        assert exc_info.value.details["transaction_id"] == "tx-999"

    def test_explain_by_id_with_overrides(self):
        """explain_by_id should pass through num_samples and top_n overrides."""
        result = self.explainer.explain_by_id(
            "tx-001", self.dataset, num_samples=2000, top_n=3
        )
        assert result.num_perturbation_samples == 2000
        assert len(result.feature_contributions) <= 3


class TestLIMEExplainerLowFidelity:
    """Tests for low-fidelity flagging behavior."""

    def test_low_fidelity_flag_when_score_below_threshold(self):
        """Explanation should be flagged as low-confidence when fidelity < 0.6."""
        # Use a random predict function to produce low fidelity
        data = _make_training_data(n_samples=50, n_features=5)
        names = [f"f{i}" for i in range(5)]
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_low_fidelity_predict_fn,
            num_samples=1000,
            top_n=3,
        )
        instance = data[0]
        result = explainer.explain(instance)
        # With a random predict function, fidelity should be low
        if result.fidelity_score < LOW_FIDELITY_THRESHOLD:
            assert result.is_low_confidence is True
        # The key property: is_low_confidence iff fidelity < 0.6
        assert result.is_low_confidence == (result.fidelity_score < LOW_FIDELITY_THRESHOLD)

    def test_fidelity_flag_consistency(self):
        """is_low_confidence should always be consistent with fidelity_score."""
        data = _make_training_data(n_samples=50, n_features=5)
        names = [f"f{i}" for i in range(5)]
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_predict_fn,
            num_samples=1000,
            top_n=3,
        )
        # Test multiple instances
        for i in range(5):
            result = explainer.explain(data[i])
            expected_flag = result.fidelity_score < LOW_FIDELITY_THRESHOLD
            assert result.is_low_confidence == expected_flag, (
                f"Instance {i}: fidelity={result.fidelity_score}, "
                f"is_low_confidence={result.is_low_confidence}, expected={expected_flag}"
            )

    def test_low_confidence_warning_logged(self, caplog):
        """Low confidence should log a warning."""
        import logging

        data = _make_training_data(n_samples=50, n_features=5)
        names = [f"f{i}" for i in range(5)]
        explainer = LIMEExplainer(
            training_data=data,
            feature_names=names,
            predict_fn=_mock_low_fidelity_predict_fn,
            num_samples=1000,
            top_n=3,
        )
        instance = data[0]
        with caplog.at_level(logging.WARNING, logger="src.explainability.lime_explainer"):
            result = explainer.explain(instance, transaction_id="tx-low")
        if result.is_low_confidence:
            assert "low fidelity" in caplog.text.lower()
