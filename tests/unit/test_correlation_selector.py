"""Unit tests for CorrelationSelector.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.models.errors import PreprocessingError
from src.preprocessing.correlation_selector import CorrelationSelector


class TestCorrelationSelectorInit:
    """Tests for CorrelationSelector initialization and threshold validation."""

    def test_default_threshold(self):
        """Default threshold is 0.95."""
        selector = CorrelationSelector()
        assert selector.threshold == 0.95

    def test_custom_threshold(self):
        """Custom threshold within valid range is accepted."""
        selector = CorrelationSelector(threshold=0.8)
        assert selector.threshold == 0.8

    def test_threshold_lower_bound(self):
        """Threshold at 0.5 (lower bound) is accepted."""
        selector = CorrelationSelector(threshold=0.5)
        assert selector.threshold == 0.5

    def test_threshold_upper_bound(self):
        """Threshold at 1.0 (upper bound) is accepted."""
        selector = CorrelationSelector(threshold=1.0)
        assert selector.threshold == 1.0

    def test_threshold_below_range_raises_error(self):
        """Threshold below 0.5 raises PreprocessingError. (Req 7.5)"""
        with pytest.raises(PreprocessingError, match="between 0.5 and 1.0"):
            CorrelationSelector(threshold=0.49)

    def test_threshold_above_range_raises_error(self):
        """Threshold above 1.0 raises PreprocessingError. (Req 7.5)"""
        with pytest.raises(PreprocessingError, match="between 0.5 and 1.0"):
            CorrelationSelector(threshold=1.01)

    def test_threshold_zero_raises_error(self):
        """Threshold of 0 raises PreprocessingError."""
        with pytest.raises(PreprocessingError):
            CorrelationSelector(threshold=0.0)

    def test_threshold_negative_raises_error(self):
        """Negative threshold raises PreprocessingError."""
        with pytest.raises(PreprocessingError):
            CorrelationSelector(threshold=-0.5)

    def test_threshold_non_numeric_raises_error(self):
        """Non-numeric threshold raises PreprocessingError."""
        with pytest.raises(PreprocessingError, match="must be numeric"):
            CorrelationSelector(threshold="high")  # type: ignore


class TestCorrelationSelectorSelect:
    """Tests for CorrelationSelector.select method."""

    def test_no_correlated_features(self):
        """Features below threshold are all retained. (Req 7.1)"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": rng.standard_normal(100),
                "B": rng.standard_normal(100),
                "C": rng.standard_normal(100),
            }
        )
        selector = CorrelationSelector(threshold=0.95)
        result = selector.select(df)

        assert list(result.columns) == ["A", "B", "C"]
        assert selector.removed_features_ == []

    def test_highly_correlated_pair_removes_lower_variance(self):
        """From a correlated pair, the feature with lower variance is removed. (Req 7.2)"""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(100)
        # A has higher variance, B is nearly identical but scaled down (lower variance)
        df = pd.DataFrame(
            {
                "A": base * 2.0,  # higher variance
                "B": base * 2.0 + rng.standard_normal(100) * 0.01,  # nearly same, slightly different
            }
        )
        # These are highly correlated. A has variance ~4, B has variance ~4 too but with tiny noise.
        # Let's be more explicit:
        df = pd.DataFrame(
            {
                "A": base * 3.0,  # variance ~ 9
                "B": base * 1.0,  # variance ~ 1, same correlation
            }
        )
        selector = CorrelationSelector(threshold=0.5)
        result = selector.select(df)

        # B has lower variance, so B should be removed
        assert "A" in result.columns
        assert "B" not in result.columns
        assert "B" in selector.removed_features_

    def test_equal_variance_removes_later_column(self):
        """When variances are equal, the later column is removed. (Req 7.2)"""
        # Create two perfectly correlated features with equal variance
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame(
            {
                "first": data,
                "second": data,  # identical = correlation 1.0, equal variance
            }
        )
        selector = CorrelationSelector(threshold=0.5)
        result = selector.select(df)

        # second is later in column order, so it should be removed
        assert "first" in result.columns
        assert "second" not in result.columns
        assert "second" in selector.removed_features_

    def test_multiple_correlated_pairs(self):
        """Multiple correlated pairs are handled correctly."""
        rng = np.random.default_rng(42)
        base1 = rng.standard_normal(100)
        base2 = rng.standard_normal(100)

        df = pd.DataFrame(
            {
                "A": base1 * 5.0,  # high variance
                "B": base1 * 1.0,  # low variance, correlated with A
                "C": base2 * 5.0,  # high variance
                "D": base2 * 1.0,  # low variance, correlated with C
                "E": rng.standard_normal(100),  # independent
            }
        )
        selector = CorrelationSelector(threshold=0.9)
        result = selector.select(df)

        # B and D should be removed (lower variance in their pairs)
        assert "A" in result.columns
        assert "B" not in result.columns
        assert "C" in result.columns
        assert "D" not in result.columns
        assert "E" in result.columns

    def test_single_column_returns_unchanged(self):
        """Single column DataFrame is returned unchanged."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        selector = CorrelationSelector(threshold=0.95)
        result = selector.select(df)

        assert list(result.columns) == ["A"]
        assert selector.removed_features_ == []

    def test_empty_dataframe(self):
        """Empty DataFrame is returned unchanged."""
        df = pd.DataFrame()
        selector = CorrelationSelector(threshold=0.95)
        result = selector.select(df)

        assert result.empty

    def test_returns_dataframe(self):
        """select() returns a pandas DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        selector = CorrelationSelector(threshold=0.95)
        result = selector.select(df)

        assert isinstance(result, pd.DataFrame)

    def test_non_numeric_columns_preserved(self):
        """Non-numeric columns pass through unchanged."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "name": ["a", "b", "c", "d", "e"],
                "X": rng.standard_normal(5),
                "Y": rng.standard_normal(5),
            }
        )
        selector = CorrelationSelector(threshold=0.95)
        result = selector.select(df)

        assert "name" in result.columns

    def test_threshold_boundary_not_triggered(self):
        """Correlation exactly at threshold does NOT trigger removal."""
        # correlation must be ABOVE threshold, not equal
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame(
            {
                "A": data,
                "B": data,  # perfect correlation = 1.0
            }
        )
        # Threshold at 1.0: correlation is not > 1.0, so nothing removed
        selector = CorrelationSelector(threshold=1.0)
        result = selector.select(df)

        assert list(result.columns) == ["A", "B"]
        assert selector.removed_features_ == []


class TestCorrelationSelectorLogging:
    """Tests for logging output of CorrelationSelector. (Req 7.3)"""

    def test_logs_removed_features(self, caplog):
        """Removed features are logged with correlation and partner info."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame(
            {
                "A": data * 3.0,
                "B": data * 1.0,
            }
        )
        selector = CorrelationSelector(threshold=0.5)

        with caplog.at_level(logging.INFO):
            selector.select(df)

        # Verify log contains removed feature info
        log_text = caplog.text
        assert "Removed feature" in log_text
        assert "B" in log_text
        assert "A" in log_text

    def test_logs_no_removal_message(self, caplog):
        """When no features removed, an info message is logged."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": rng.standard_normal(50),
                "B": rng.standard_normal(50),
            }
        )
        selector = CorrelationSelector(threshold=0.95)

        with caplog.at_level(logging.INFO):
            selector.select(df)

        assert "No features exceed correlation threshold" in caplog.text

    def test_removal_log_attribute_populated(self):
        """removal_log_ contains details of each removal decision."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame(
            {
                "A": data * 3.0,  # higher variance
                "B": data * 1.0,  # lower variance
            }
        )
        selector = CorrelationSelector(threshold=0.5)
        selector.select(df)

        assert len(selector.removal_log_) == 1
        entry = selector.removal_log_[0]
        assert entry["removed_feature"] == "B"
        assert entry["retained_feature"] == "A"
        assert entry["correlation"] > 0.5
        assert "removed_variance" in entry
        assert "retained_variance" in entry


class TestCorrelationSelectorEdgeCases:
    """Edge case tests for CorrelationSelector."""

    def test_all_features_correlated(self):
        """When all features are correlated, only one remains."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame(
            {
                "A": data * 3.0,
                "B": data * 2.0,
                "C": data * 1.0,
            }
        )
        selector = CorrelationSelector(threshold=0.5)
        result = selector.select(df)

        # A has highest variance, should be retained
        assert "A" in result.columns
        assert result.shape[1] == 1

    def test_constant_feature_zero_variance(self):
        """Constant features (zero variance) result in NaN correlations and are not removed."""
        df = pd.DataFrame(
            {
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
                "varied": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        selector = CorrelationSelector(threshold=0.95)
        result = selector.select(df)

        # NaN correlations should not trigger removal
        assert "constant" in result.columns
        assert "varied" in result.columns

    def test_repeated_select_resets_state(self):
        """Calling select() again resets removed_features_ and removal_log_."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df_corr = pd.DataFrame({"A": data * 3.0, "B": data * 1.0})
        df_uncorr = pd.DataFrame(
            {"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [5.0, 3.0, 1.0, 4.0, 2.0]}
        )

        selector = CorrelationSelector(threshold=0.5)
        selector.select(df_corr)
        assert len(selector.removed_features_) > 0

        selector.select(df_uncorr)
        # After second call with uncorrelated data, state should be reset
        assert selector.removed_features_ == []
        assert selector.removal_log_ == []
