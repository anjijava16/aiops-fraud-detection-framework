"""Unit tests for FeatureNormalizer.

Tests cover StandardScaler and MinMaxScaler behavior, fit/transform workflow,
persistence via save_params/load_params, and error handling for unsupported
scaler types and invalid operations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.errors import PreprocessingError
from src.preprocessing.feature_normalizer import FeatureNormalizer


class TestFeatureNormalizerInit:
    """Tests for FeatureNormalizer initialization."""

    def test_init_standard_scaler(self):
        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        assert normalizer.scaler_type == "StandardScaler"
        assert normalizer.is_fitted is False

    def test_init_minmax_scaler(self):
        normalizer = FeatureNormalizer(scaler_type="MinMaxScaler")
        assert normalizer.scaler_type == "MinMaxScaler"
        assert normalizer.is_fitted is False

    def test_init_default_scaler(self):
        normalizer = FeatureNormalizer()
        assert normalizer.scaler_type == "StandardScaler"

    def test_init_unsupported_scaler_raises_error(self):
        with pytest.raises(PreprocessingError) as exc_info:
            FeatureNormalizer(scaler_type="RobustScaler")
        assert "Unsupported scaler type" in str(exc_info.value.message)
        assert exc_info.value.details["scaler_type"] == "RobustScaler"
        assert "StandardScaler" in exc_info.value.details["supported_types"]
        assert "MinMaxScaler" in exc_info.value.details["supported_types"]

    def test_init_empty_string_raises_error(self):
        with pytest.raises(PreprocessingError):
            FeatureNormalizer(scaler_type="")


class TestStandardScaler:
    """Tests for StandardScaler normalization behavior."""

    def test_fit_transform_produces_zero_mean_unit_std(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=50, scale=10, size=(1000, 3))

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_transformed = normalizer.fit_transform(X_train)

        # Requirement 6.4: mean within ±0.001 of 0, std within ±0.01 of 1
        assert X_transformed.shape == X_train.shape
        for col in range(X_transformed.shape[1]):
            assert abs(X_transformed[:, col].mean()) < 0.001
            assert abs(X_transformed[:, col].std() - 1.0) < 0.01

    def test_transform_validation_data_with_fitted_params(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=50, scale=10, size=(500, 2))
        X_val = rng.normal(loc=52, scale=11, size=(100, 2))

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        normalizer.fit_transform(X_train)
        X_val_transformed = normalizer.transform(X_val)

        # Validation transform uses training params, so may not have
        # exactly zero mean, but shape should be preserved
        assert X_val_transformed.shape == X_val.shape

    def test_fit_transform_single_feature(self):
        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_transformed = normalizer.fit_transform(X_train)

        assert abs(X_transformed.mean()) < 0.001
        assert abs(X_transformed.std() - 1.0) < 0.01


class TestMinMaxScaler:
    """Tests for MinMaxScaler normalization behavior."""

    def test_fit_transform_produces_bounded_output(self):
        rng = np.random.default_rng(42)
        X_train = rng.uniform(low=-100, high=200, size=(500, 4))

        normalizer = FeatureNormalizer(scaler_type="MinMaxScaler")
        X_transformed = normalizer.fit_transform(X_train)

        # Requirement 6.5: all values in [0.0, 1.0] on training set
        # Allow floating-point epsilon tolerance
        assert X_transformed.shape == X_train.shape
        assert X_transformed.min() >= -1e-10
        assert X_transformed.max() <= 1.0 + 1e-10

    def test_fit_transform_min_is_zero_max_is_one(self):
        X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        normalizer = FeatureNormalizer(scaler_type="MinMaxScaler")
        X_transformed = normalizer.fit_transform(X_train)

        # On training data, min should be exactly 0 and max exactly 1
        for col in range(X_transformed.shape[1]):
            assert abs(X_transformed[:, col].min() - 0.0) < 1e-10
            assert abs(X_transformed[:, col].max() - 1.0) < 1e-10

    def test_transform_validation_can_exceed_bounds(self):
        """Validation data may have values outside training range."""
        X_train = np.array([[1.0], [2.0], [3.0]])
        X_val = np.array([[0.0], [4.0]])  # outside training range

        normalizer = FeatureNormalizer(scaler_type="MinMaxScaler")
        normalizer.fit_transform(X_train)
        X_val_transformed = normalizer.transform(X_val)

        # Values outside training range will exceed [0, 1]
        assert X_val_transformed[0, 0] < 0.0
        assert X_val_transformed[1, 0] > 1.0


class TestFitTransformWorkflow:
    """Tests for the fit/transform workflow."""

    def test_transform_before_fit_raises_error(self):
        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(PreprocessingError) as exc_info:
            normalizer.transform(X)
        assert "not been fitted" in str(exc_info.value.message)

    def test_fit_transform_marks_as_fitted(self):
        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_train = np.array([[1.0], [2.0], [3.0]])

        assert normalizer.is_fitted is False
        normalizer.fit_transform(X_train)
        assert normalizer.is_fitted is True

    def test_fit_transform_empty_array_raises_error(self):
        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_empty = np.array([]).reshape(0, 3)

        with pytest.raises(PreprocessingError) as exc_info:
            normalizer.fit_transform(X_empty)
        assert "empty data" in str(exc_info.value.message)

    def test_transform_preserves_shape(self):
        rng = np.random.default_rng(42)
        X_train = rng.random((100, 5))
        X_val = rng.random((30, 5))

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_train_t = normalizer.fit_transform(X_train)
        X_val_t = normalizer.transform(X_val)

        assert X_train_t.shape == (100, 5)
        assert X_val_t.shape == (30, 5)


class TestPersistence:
    """Tests for saving and loading scaler parameters."""

    def test_save_and_load_standard_scaler(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=10, scale=3, size=(200, 3))
        X_test = rng.normal(loc=10, scale=3, size=(50, 3))

        # Fit and transform with original normalizer
        normalizer1 = FeatureNormalizer(scaler_type="StandardScaler")
        normalizer1.fit_transform(X_train)
        expected = normalizer1.transform(X_test)

        # Save and reload into a new normalizer
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        normalizer1.save_params(path)

        normalizer2 = FeatureNormalizer(scaler_type="StandardScaler")
        normalizer2.load_params(path)
        actual = normalizer2.transform(X_test)

        # Requirement 6.3: identical output after reload
        np.testing.assert_array_almost_equal(actual, expected)

        # Cleanup
        Path(path).unlink()

    def test_save_and_load_minmax_scaler(self):
        rng = np.random.default_rng(42)
        X_train = rng.uniform(0, 100, size=(200, 2))
        X_test = rng.uniform(0, 100, size=(50, 2))

        normalizer1 = FeatureNormalizer(scaler_type="MinMaxScaler")
        normalizer1.fit_transform(X_train)
        expected = normalizer1.transform(X_test)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        normalizer1.save_params(path)

        normalizer2 = FeatureNormalizer(scaler_type="MinMaxScaler")
        normalizer2.load_params(path)
        actual = normalizer2.transform(X_test)

        np.testing.assert_array_almost_equal(actual, expected)

        Path(path).unlink()

    def test_save_unfitted_scaler_raises_error(self):
        normalizer = FeatureNormalizer(scaler_type="StandardScaler")

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        with pytest.raises(PreprocessingError) as exc_info:
            normalizer.save_params(path)
        assert "not been fitted" in str(exc_info.value.message)

        Path(path).unlink()

    def test_load_nonexistent_file_raises_error(self):
        normalizer = FeatureNormalizer(scaler_type="StandardScaler")

        with pytest.raises(PreprocessingError) as exc_info:
            normalizer.load_params("/nonexistent/path/scaler.joblib")
        assert "not found" in str(exc_info.value.message)

    def test_load_marks_as_fitted(self):
        rng = np.random.default_rng(42)
        X_train = rng.random((50, 2))

        normalizer1 = FeatureNormalizer(scaler_type="StandardScaler")
        normalizer1.fit_transform(X_train)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        normalizer1.save_params(path)

        normalizer2 = FeatureNormalizer(scaler_type="StandardScaler")
        assert normalizer2.is_fitted is False
        normalizer2.load_params(path)
        assert normalizer2.is_fitted is True

        Path(path).unlink()

    def test_save_creates_parent_directories(self):
        rng = np.random.default_rng(42)
        X_train = rng.random((50, 2))

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        normalizer.fit_transform(X_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "scaler.joblib"
            normalizer.save_params(path)
            assert path.exists()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample_standard_scaler(self):
        """Single sample results in zero std which sklearn handles."""
        X_train = np.array([[5.0, 10.0]])

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_transformed = normalizer.fit_transform(X_train)

        # With single sample, sklearn sets std to 1 (no scaling)
        assert X_transformed.shape == (1, 2)

    def test_single_sample_minmax_scaler(self):
        """Single sample with MinMaxScaler: min==max so no scaling."""
        X_train = np.array([[5.0, 10.0]])

        normalizer = FeatureNormalizer(scaler_type="MinMaxScaler")
        X_transformed = normalizer.fit_transform(X_train)

        # With single sample, all values map to 0
        assert X_transformed.shape == (1, 2)

    def test_constant_feature_standard_scaler(self):
        """Constant features have zero variance, sklearn handles gracefully."""
        X_train = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_transformed = normalizer.fit_transform(X_train)

        # Constant column should be all zeros after StandardScaler
        assert np.all(X_transformed[:, 0] == 0.0)

    def test_large_dataset_performance(self):
        """Verify normalizer handles large arrays without error."""
        rng = np.random.default_rng(42)
        X_train = rng.random((10000, 30))

        normalizer = FeatureNormalizer(scaler_type="StandardScaler")
        X_transformed = normalizer.fit_transform(X_train)

        assert X_transformed.shape == (10000, 30)
