"""Unit tests for the SoftVotingEnsemble class."""

import logging

import numpy as np
import pytest

from src.ensemble.soft_voting_ensemble import SoftVotingEnsemble
from src.models.ensemble import EnsemblePrediction
from src.models.errors import EnsembleError


class TestSoftVotingEnsembleInit:
    """Tests for SoftVotingEnsemble initialization and validation."""

    def test_default_weights(self):
        """Default weights should be equal 1/3 for each model."""
        ensemble = SoftVotingEnsemble()
        assert abs(ensemble.weights["xgboost"] - 1 / 3) < 1e-9
        assert abs(ensemble.weights["lightgbm"] - 1 / 3) < 1e-9
        assert abs(ensemble.weights["catboost"] - 1 / 3) < 1e-9

    def test_default_threshold(self):
        """Default threshold should be 0.5."""
        ensemble = SoftVotingEnsemble()
        assert ensemble.decision_threshold == 0.5

    def test_custom_weights(self):
        """Custom weights that sum to 1.0 should be accepted."""
        weights = {"xgboost": 0.5, "lightgbm": 0.3, "catboost": 0.2}
        ensemble = SoftVotingEnsemble(weights=weights)
        assert ensemble.weights == weights

    def test_custom_threshold(self):
        """Custom threshold in (0, 1) should be accepted."""
        ensemble = SoftVotingEnsemble(decision_threshold=0.7)
        assert ensemble.decision_threshold == 0.7

    def test_weights_not_summing_to_one_raises(self):
        """Weights that don't sum to 1.0 should raise EnsembleError."""
        weights = {"xgboost": 0.5, "lightgbm": 0.3, "catboost": 0.3}
        with pytest.raises(EnsembleError, match="sum to 1.0"):
            SoftVotingEnsemble(weights=weights)

    def test_negative_weight_raises(self):
        """Negative weight should raise EnsembleError."""
        weights = {"xgboost": -0.1, "lightgbm": 0.6, "catboost": 0.5}
        with pytest.raises(EnsembleError, match="must be in"):
            SoftVotingEnsemble(weights=weights)

    def test_zero_weight_raises(self):
        """Zero weight should raise EnsembleError."""
        weights = {"xgboost": 0.0, "lightgbm": 0.5, "catboost": 0.5}
        with pytest.raises(EnsembleError, match="must be in"):
            SoftVotingEnsemble(weights=weights)

    def test_weight_equal_to_one_raises(self):
        """Weight of 1.0 should raise EnsembleError (exclusive bounds)."""
        weights = {"xgboost": 1.0, "lightgbm": 0.0, "catboost": 0.0}
        with pytest.raises(EnsembleError, match="must be in"):
            SoftVotingEnsemble(weights=weights)

    def test_missing_model_key_raises(self):
        """Missing a model key in weights should raise EnsembleError."""
        weights = {"xgboost": 0.5, "lightgbm": 0.5}
        with pytest.raises(EnsembleError, match="Weights must be provided"):
            SoftVotingEnsemble(weights=weights)

    def test_extra_model_key_raises(self):
        """Extra model key in weights should raise EnsembleError."""
        weights = {"xgboost": 0.25, "lightgbm": 0.25, "catboost": 0.25, "extra": 0.25}
        with pytest.raises(EnsembleError, match="Weights must be provided"):
            SoftVotingEnsemble(weights=weights)

    def test_threshold_zero_raises(self):
        """Threshold of 0.0 should raise EnsembleError."""
        with pytest.raises(EnsembleError, match="Decision threshold"):
            SoftVotingEnsemble(decision_threshold=0.0)

    def test_threshold_one_raises(self):
        """Threshold of 1.0 should raise EnsembleError."""
        with pytest.raises(EnsembleError, match="Decision threshold"):
            SoftVotingEnsemble(decision_threshold=1.0)

    def test_threshold_negative_raises(self):
        """Negative threshold should raise EnsembleError."""
        with pytest.raises(EnsembleError, match="Decision threshold"):
            SoftVotingEnsemble(decision_threshold=-0.1)


class TestSoftVotingEnsemblePredict:
    """Tests for single-sample prediction."""

    def setup_method(self):
        self.ensemble = SoftVotingEnsemble()

    def test_predict_equal_weights_equal_probs(self):
        """Equal weights and equal probs should produce the same probability."""
        probs = {
            "xgboost": np.array([0.6]),
            "lightgbm": np.array([0.6]),
            "catboost": np.array([0.6]),
        }
        result = self.ensemble.predict(probs)
        assert isinstance(result, EnsemblePrediction)
        assert abs(result.fraud_score - 0.6) < 1e-9
        assert result.is_fraud is True

    def test_predict_weighted_average(self):
        """Weighted average should be computed correctly."""
        weights = {"xgboost": 0.5, "lightgbm": 0.3, "catboost": 0.2}
        ensemble = SoftVotingEnsemble(weights=weights)
        probs = {
            "xgboost": np.array([0.8]),
            "lightgbm": np.array([0.6]),
            "catboost": np.array([0.4]),
        }
        result = ensemble.predict(probs)
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4
        assert abs(result.fraud_score - expected) < 1e-9

    def test_predict_below_threshold_not_fraud(self):
        """Score below threshold should classify as not fraud."""
        probs = {
            "xgboost": np.array([0.2]),
            "lightgbm": np.array([0.3]),
            "catboost": np.array([0.1]),
        }
        result = self.ensemble.predict(probs)
        assert result.is_fraud is False

    def test_predict_at_threshold_is_fraud(self):
        """Score exactly at threshold should classify as fraud."""
        ensemble = SoftVotingEnsemble(decision_threshold=0.5)
        probs = {
            "xgboost": np.array([0.5]),
            "lightgbm": np.array([0.5]),
            "catboost": np.array([0.5]),
        }
        result = ensemble.predict(probs)
        assert result.fraud_score == 0.5
        assert result.is_fraud is True

    def test_predict_model_scores_populated(self):
        """Model scores should contain per-model probabilities."""
        probs = {
            "xgboost": np.array([0.9]),
            "lightgbm": np.array([0.7]),
            "catboost": np.array([0.5]),
        }
        result = self.ensemble.predict(probs)
        assert result.model_scores["xgboost"] == 0.9
        assert result.model_scores["lightgbm"] == 0.7
        assert result.model_scores["catboost"] == 0.5

    def test_predict_model_version_set(self):
        """Model version should be set in prediction result."""
        probs = {
            "xgboost": np.array([0.5]),
            "lightgbm": np.array([0.5]),
            "catboost": np.array([0.5]),
        }
        result = self.ensemble.predict(probs)
        assert result.model_version == "1.0.0"

    def test_predict_missing_model_raises(self):
        """Missing model in probabilities should raise EnsembleError."""
        probs = {
            "xgboost": np.array([0.5]),
            "lightgbm": np.array([0.5]),
        }
        with pytest.raises(EnsembleError, match="failed to produce predictions"):
            self.ensemble.predict(probs)

    def test_predict_none_probability_raises(self):
        """None probability for a model should raise EnsembleError."""
        probs = {
            "xgboost": np.array([0.5]),
            "lightgbm": None,
            "catboost": np.array([0.5]),
        }
        with pytest.raises(EnsembleError, match="produced None"):
            self.ensemble.predict(probs)

    def test_predict_scalar_input(self):
        """Scalar probability values should work correctly."""
        probs = {
            "xgboost": np.float64(0.8),
            "lightgbm": np.float64(0.6),
            "catboost": np.float64(0.4),
        }
        result = self.ensemble.predict(probs)
        expected = (1 / 3) * 0.8 + (1 / 3) * 0.6 + (1 / 3) * 0.4
        assert abs(result.fraud_score - expected) < 1e-9


class TestSoftVotingEnsemblePredictBatch:
    """Tests for batch prediction."""

    def setup_method(self):
        self.ensemble = SoftVotingEnsemble()

    def test_predict_batch_returns_list(self):
        """Batch predict should return a list of EnsemblePrediction."""
        probs = {
            "xgboost": np.array([0.9, 0.1, 0.5]),
            "lightgbm": np.array([0.8, 0.2, 0.6]),
            "catboost": np.array([0.7, 0.3, 0.4]),
        }
        results = self.ensemble.predict_batch(probs)
        assert len(results) == 3
        assert all(isinstance(r, EnsemblePrediction) for r in results)

    def test_predict_batch_correct_scores(self):
        """Batch predictions should compute correct weighted averages."""
        probs = {
            "xgboost": np.array([0.9, 0.1]),
            "lightgbm": np.array([0.9, 0.1]),
            "catboost": np.array([0.9, 0.1]),
        }
        results = self.ensemble.predict_batch(probs)
        assert abs(results[0].fraud_score - 0.9) < 1e-9
        assert abs(results[1].fraud_score - 0.1) < 1e-9

    def test_predict_batch_classification(self):
        """Batch predictions should correctly classify fraud/not fraud."""
        probs = {
            "xgboost": np.array([0.9, 0.1]),
            "lightgbm": np.array([0.9, 0.1]),
            "catboost": np.array([0.9, 0.1]),
        }
        results = self.ensemble.predict_batch(probs)
        assert results[0].is_fraud is True
        assert results[1].is_fraud is False

    def test_predict_batch_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise EnsembleError."""
        probs = {
            "xgboost": np.array([0.9, 0.1, 0.5]),
            "lightgbm": np.array([0.8, 0.2]),
            "catboost": np.array([0.7, 0.3, 0.4]),
        }
        with pytest.raises(EnsembleError, match="mismatched lengths"):
            self.ensemble.predict_batch(probs)

    def test_predict_batch_missing_model_raises(self):
        """Missing model in batch probabilities should raise EnsembleError."""
        probs = {
            "xgboost": np.array([0.9]),
            "catboost": np.array([0.7]),
        }
        with pytest.raises(EnsembleError, match="failed to produce predictions"):
            self.ensemble.predict_batch(probs)

    def test_predict_batch_logs_metrics_when_y_true_provided(self, caplog):
        """Metrics should be logged when y_true is provided."""
        probs = {
            "xgboost": np.array([0.9, 0.1, 0.8, 0.2]),
            "lightgbm": np.array([0.8, 0.2, 0.7, 0.3]),
            "catboost": np.array([0.7, 0.3, 0.9, 0.1]),
        }
        y_true = np.array([1, 0, 1, 0])

        with caplog.at_level(logging.INFO, logger="src.ensemble.soft_voting_ensemble"):
            self.ensemble.predict_batch(probs, y_true=y_true)

        assert "Ensemble metrics" in caplog.text
        assert "F1:" in caplog.text
        assert "Precision:" in caplog.text
        assert "Recall:" in caplog.text
        assert "AUC-ROC:" in caplog.text

    def test_predict_batch_no_logging_without_y_true(self, caplog):
        """No metrics should be logged when y_true is not provided."""
        probs = {
            "xgboost": np.array([0.9, 0.1]),
            "lightgbm": np.array([0.8, 0.2]),
            "catboost": np.array([0.7, 0.3]),
        }

        with caplog.at_level(logging.INFO, logger="src.ensemble.soft_voting_ensemble"):
            self.ensemble.predict_batch(probs)

        assert "Ensemble metrics" not in caplog.text

    def test_predict_batch_single_class_y_true_logs_warning(self, caplog):
        """Single class in y_true should log a warning about AUC-ROC."""
        probs = {
            "xgboost": np.array([0.9, 0.8]),
            "lightgbm": np.array([0.8, 0.7]),
            "catboost": np.array([0.7, 0.6]),
        }
        y_true = np.array([1, 1])  # Only one class

        with caplog.at_level(logging.WARNING, logger="src.ensemble.soft_voting_ensemble"):
            self.ensemble.predict_batch(probs, y_true=y_true)

        assert "AUC-ROC cannot be computed" in caplog.text


class TestSoftVotingEnsembleCustomThreshold:
    """Tests for custom decision threshold behavior."""

    def test_high_threshold_fewer_fraud(self):
        """Higher threshold should result in fewer fraud classifications."""
        ensemble = SoftVotingEnsemble(decision_threshold=0.9)
        probs = {
            "xgboost": np.array([0.85]),
            "lightgbm": np.array([0.85]),
            "catboost": np.array([0.85]),
        }
        result = ensemble.predict(probs)
        assert result.fraud_score == pytest.approx(0.85)
        assert result.is_fraud is False

    def test_low_threshold_more_fraud(self):
        """Lower threshold should result in more fraud classifications."""
        ensemble = SoftVotingEnsemble(decision_threshold=0.1)
        probs = {
            "xgboost": np.array([0.15]),
            "lightgbm": np.array([0.15]),
            "catboost": np.array([0.15]),
        }
        result = ensemble.predict(probs)
        assert result.fraud_score == pytest.approx(0.15)
        assert result.is_fraud is True
