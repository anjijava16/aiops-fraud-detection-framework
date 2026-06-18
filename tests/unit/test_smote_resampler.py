"""Unit tests for SMOTEResampler."""

import logging

import numpy as np
import pytest

from src.models.errors import PreprocessingError
from src.preprocessing.smote_resampler import SMOTEResampler


@pytest.fixture
def imbalanced_data():
    """Create an imbalanced dataset with 100 majority and 10 minority samples."""
    rng = np.random.default_rng(42)
    n_majority = 100
    n_minority = 10
    n_features = 5

    X_majority = rng.standard_normal((n_majority, n_features))
    X_minority = rng.standard_normal((n_minority, n_features)) + 3.0

    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    return X, y


@pytest.fixture
def resampler():
    """Create a default SMOTEResampler."""
    return SMOTEResampler(target_ratio=1.0, random_state=42)


class TestSMOTEResamplerResample:
    """Tests for the resample method."""

    def test_achieves_target_ratio_one_to_one(self, imbalanced_data, resampler):
        """SMOTE should produce equal class counts with target_ratio=1.0."""
        X, y = imbalanced_data
        X_res, y_res = resampler.resample(X, y)

        unique, counts = np.unique(y_res, return_counts=True)
        class_dist = dict(zip(unique, counts))

        assert class_dist[0] == 100  # majority unchanged
        assert class_dist[1] == 100  # minority oversampled to match

    def test_achieves_custom_target_ratio(self, imbalanced_data):
        """SMOTE with target_ratio=0.5 should produce half as many minority as majority."""
        X, y = imbalanced_data
        resampler = SMOTEResampler(target_ratio=0.5, random_state=42)
        X_res, y_res = resampler.resample(X, y)

        unique, counts = np.unique(y_res, return_counts=True)
        class_dist = dict(zip(unique, counts))

        assert class_dist[0] == 100  # majority unchanged
        assert class_dist[1] == 50  # 0.5 * 100

    def test_majority_class_unchanged(self, imbalanced_data, resampler):
        """SMOTE should not modify majority class samples."""
        X, y = imbalanced_data
        X_res, y_res = resampler.resample(X, y)

        # Original majority samples should still be present
        majority_mask_original = y == 0
        majority_mask_resampled = y_res == 0

        assert np.sum(majority_mask_resampled) == np.sum(majority_mask_original)

    def test_output_shapes_consistent(self, imbalanced_data, resampler):
        """X and y should have consistent shapes after resampling."""
        X, y = imbalanced_data
        X_res, y_res = resampler.resample(X, y)

        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]  # features preserved

    def test_returns_numpy_arrays(self, imbalanced_data, resampler):
        """Output should be numpy arrays."""
        X, y = imbalanced_data
        X_res, y_res = resampler.resample(X, y)

        assert isinstance(X_res, np.ndarray)
        assert isinstance(y_res, np.ndarray)

    def test_reproducible_with_same_seed(self, imbalanced_data):
        """Same random_state should produce identical results."""
        X, y = imbalanced_data
        resampler1 = SMOTEResampler(target_ratio=1.0, random_state=123)
        resampler2 = SMOTEResampler(target_ratio=1.0, random_state=123)

        X_res1, y_res1 = resampler1.resample(X, y)
        X_res2, y_res2 = resampler2.resample(X, y)

        np.testing.assert_array_equal(X_res1, X_res2)
        np.testing.assert_array_equal(y_res1, y_res2)


class TestSMOTEResamplerErrors:
    """Tests for error cases."""

    def test_raises_error_when_minority_too_small(self):
        """Should raise PreprocessingError when minority class has fewer than 6 samples."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((105, 5))
        # Only 5 minority samples (below the minimum of 6)
        y = np.concatenate([np.zeros(100), np.ones(5)])

        resampler = SMOTEResampler(target_ratio=1.0, random_state=42)

        with pytest.raises(PreprocessingError) as exc_info:
            resampler.resample(X, y)

        assert "Insufficient minority samples" in str(exc_info.value)
        assert exc_info.value.details["minority_count"] == 5
        assert exc_info.value.details["min_required"] == 6

    def test_raises_error_with_zero_minority_samples(self):
        """Should raise PreprocessingError when there are no minority samples."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = np.zeros(100)  # all majority

        resampler = SMOTEResampler(target_ratio=1.0, random_state=42)

        with pytest.raises(PreprocessingError) as exc_info:
            resampler.resample(X, y)

        assert exc_info.value.details["minority_count"] == 0

    def test_raises_error_with_exactly_five_minority(self):
        """Should raise PreprocessingError when minority has exactly 5 (below 6)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((105, 5))
        y = np.concatenate([np.zeros(100), np.ones(5)])

        resampler = SMOTEResampler(target_ratio=1.0, random_state=42)

        with pytest.raises(PreprocessingError):
            resampler.resample(X, y)


class TestSMOTEResamplerEdgeCases:
    """Tests for edge cases."""

    def test_exactly_six_minority_samples_succeeds(self):
        """Should succeed when minority class has exactly 6 samples."""
        rng = np.random.default_rng(42)
        n_features = 5
        X_majority = rng.standard_normal((100, n_features))
        X_minority = rng.standard_normal((6, n_features)) + 3.0

        X = np.vstack([X_majority, X_minority])
        y = np.concatenate([np.zeros(100), np.ones(6)])

        resampler = SMOTEResampler(target_ratio=1.0, random_state=42)
        X_res, y_res = resampler.resample(X, y)

        unique, counts = np.unique(y_res, return_counts=True)
        class_dist = dict(zip(unique, counts))

        assert class_dist[1] == 100  # minority oversampled to match majority

    def test_no_resampling_when_already_balanced(self):
        """Should return copy without resampling when target already met."""
        rng = np.random.default_rng(42)
        n_features = 5
        X = rng.standard_normal((200, n_features))
        y = np.concatenate([np.zeros(100), np.ones(100)])

        resampler = SMOTEResampler(target_ratio=1.0, random_state=42)
        X_res, y_res = resampler.resample(X, y)

        # Should return copies without modification
        assert X_res.shape == X.shape
        assert y_res.shape == y.shape
        np.testing.assert_array_equal(X_res, X)
        np.testing.assert_array_equal(y_res, y)

    def test_does_not_modify_original_arrays(self, imbalanced_data, resampler):
        """Original arrays should not be modified in-place."""
        X, y = imbalanced_data
        X_orig = X.copy()
        y_orig = y.copy()

        resampler.resample(X, y)

        np.testing.assert_array_equal(X, X_orig)
        np.testing.assert_array_equal(y, y_orig)


class TestSMOTEResamplerLogging:
    """Tests for logging behavior."""

    def test_logs_original_distribution(self, imbalanced_data, resampler, caplog):
        """Should log original class distribution."""
        X, y = imbalanced_data

        with caplog.at_level(logging.INFO):
            resampler.resample(X, y)

        assert "Original class distribution" in caplog.text
        assert "majority(0)=100" in caplog.text
        assert "minority(1)=10" in caplog.text

    def test_logs_resampled_distribution(self, imbalanced_data, resampler, caplog):
        """Should log resampled class distribution."""
        X, y = imbalanced_data

        with caplog.at_level(logging.INFO):
            resampler.resample(X, y)

        assert "Resampled class distribution" in caplog.text
        assert "minority(1)=100" in caplog.text
