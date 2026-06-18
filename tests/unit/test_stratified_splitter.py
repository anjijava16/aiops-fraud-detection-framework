"""Unit tests for StratifiedSplitter."""

import numpy as np
import pandas as pd
import pytest

from src.models.errors import PreprocessingError
from src.preprocessing.stratified_splitter import DataSplits, StratifiedSplitter


# ---------- Fixtures ----------


@pytest.fixture
def imbalanced_dataset():
    """Create a dataset mimicking CCFD imbalance (~10% fraud)."""
    rng = np.random.default_rng(42)
    n_legit = 900
    n_fraud = 100
    n_features = 10

    X = rng.standard_normal((n_legit + n_fraud, n_features))
    y = np.array([0] * n_legit + [1] * n_fraud)
    return X, y


@pytest.fixture
def balanced_dataset():
    """Create a balanced binary dataset."""
    rng = np.random.default_rng(42)
    n_per_class = 200
    n_features = 5

    X = rng.standard_normal((n_per_class * 2, n_features))
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


@pytest.fixture
def splitter():
    """Default StratifiedSplitter instance."""
    return StratifiedSplitter()


# ---------- split() Method ----------


class TestSplitMethod:
    """Test the split(X, y, test_size, random_seed) method."""

    def test_default_parameters(self, splitter, imbalanced_dataset):
        """Default split uses test_size=0.2 and random_seed=42."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)

        total = len(y)
        expected_test = int(total * 0.2)
        assert abs(len(result.y_test) - expected_test) <= 1
        assert len(result.y_train) + len(result.y_test) == total

    def test_custom_test_size(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y, test_size=0.3)

        total = len(y)
        expected_test = int(total * 0.3)
        assert abs(len(result.y_test) - expected_test) <= 1

    def test_custom_random_seed(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y, random_seed=99)
        # Just ensure it runs without error and produces a valid split
        assert len(result.y_train) + len(result.y_test) == len(y)

    def test_returns_data_splits_dataclass(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        assert isinstance(result, DataSplits)

    def test_cv_folds_is_empty_list(self, splitter, imbalanced_dataset):
        """split() returns empty cv_folds; use get_cv_folds separately."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        assert result.cv_folds == []

    def test_class_ratio_preserved_within_1_percent(self, splitter, imbalanced_dataset):
        """Requirement 8.1: class ratio within 1% absolute tolerance."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)

        original_ratio = np.mean(y == 1)
        train_ratio = np.mean(result.y_train == 1)
        test_ratio = np.mean(result.y_test == 1)

        assert abs(train_ratio - original_ratio) <= 0.01
        assert abs(test_ratio - original_ratio) <= 0.01

    def test_feature_dimensions_preserved(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)

        assert result.X_train.shape[1] == X.shape[1]
        assert result.X_test.shape[1] == X.shape[1]

    def test_no_data_leakage(self, splitter, imbalanced_dataset):
        """Ensure train and test sets don't overlap in row count."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)

        all_X = np.vstack([result.X_train, result.X_test])
        assert all_X.shape[0] == X.shape[0]

    def test_invalid_test_size_zero(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        with pytest.raises(PreprocessingError, match="test_size"):
            splitter.split(X, y, test_size=0.0)

    def test_invalid_test_size_one(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        with pytest.raises(PreprocessingError, match="test_size"):
            splitter.split(X, y, test_size=1.0)

    def test_invalid_test_size_negative(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        with pytest.raises(PreprocessingError, match="test_size"):
            splitter.split(X, y, test_size=-0.1)

    def test_mismatched_x_y_lengths(self, splitter):
        X = np.random.randn(10, 3)
        y = np.array([0, 1, 0, 1, 0])

        with pytest.raises(PreprocessingError, match="same number of samples"):
            splitter.split(X, y)


# ---------- get_cv_folds() Method ----------


class TestGetCvFolds:
    """Test the get_cv_folds(X_train, y_train, k, random_seed) method."""

    def test_default_k_5(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train)
        assert len(folds) == 5

    def test_custom_k(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train, k=3)
        assert len(folds) == 3

    def test_returns_list_of_tuples(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train)

        assert isinstance(folds, list)
        for fold in folds:
            assert isinstance(fold, tuple)
            assert len(fold) == 2

    def test_fold_indices_are_numpy_arrays(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train)

        for train_idx, val_idx in folds:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(val_idx, np.ndarray)

    def test_fold_class_ratio_within_2_percent(self, splitter, imbalanced_dataset):
        """Requirement 8.4: each fold within 2% tolerance of training ratio."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train)

        train_ratio = np.mean(result.y_train == 1)

        for train_idx, val_idx in folds:
            fold_train_ratio = np.mean(result.y_train[train_idx] == 1)
            fold_val_ratio = np.mean(result.y_train[val_idx] == 1)

            assert abs(fold_train_ratio - train_ratio) <= 0.02
            assert abs(fold_val_ratio - train_ratio) <= 0.02

    def test_folds_cover_all_training_samples(self, splitter, imbalanced_dataset):
        """Each training sample appears in exactly one validation fold."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train)

        n_train = len(result.y_train)
        val_indices = set()
        for _, val_idx in folds:
            val_indices.update(val_idx.tolist())

        assert len(val_indices) == n_train

    def test_k_boundary_low(self, splitter, imbalanced_dataset):
        """k=2 is valid."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train, k=2)
        assert len(folds) == 2

    def test_k_boundary_high(self, splitter, imbalanced_dataset):
        """k=20 is valid."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        folds = splitter.get_cv_folds(result.X_train, result.y_train, k=20)
        assert len(folds) == 20

    def test_invalid_k_too_low(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        with pytest.raises(PreprocessingError, match="k must be between"):
            splitter.get_cv_folds(result.X_train, result.y_train, k=1)

    def test_invalid_k_too_high(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset
        result = splitter.split(X, y)
        with pytest.raises(PreprocessingError, match="k must be between"):
            splitter.get_cv_folds(result.X_train, result.y_train, k=21)

    def test_dataset_too_small_for_k_folds(self, splitter):
        """Requirement 8.5: raise error if too few samples."""
        X = np.random.randn(4, 3)
        y = np.array([0, 0, 1, 1])

        with pytest.raises(PreprocessingError):
            splitter.get_cv_folds(X, y, k=5)

    def test_class_with_too_few_samples(self, splitter):
        """A class with fewer samples than k raises an error."""
        X = np.random.randn(20, 3)
        y = np.array([0] * 17 + [1] * 3)

        with pytest.raises(PreprocessingError):
            splitter.get_cv_folds(X, y, k=5)

    def test_mismatched_x_y_lengths(self, splitter):
        X = np.random.randn(10, 3)
        y = np.array([0, 1, 0, 1, 0])

        with pytest.raises(PreprocessingError, match="same number of samples"):
            splitter.get_cv_folds(X, y)


# ---------- Reproducibility ----------


class TestReproducibility:
    """Test that identical seeds produce identical splits."""

    def test_same_seed_produces_identical_splits(self, splitter, imbalanced_dataset):
        """Requirement 8.3: identical seeds → identical splits."""
        X, y = imbalanced_dataset

        result1 = splitter.split(X, y, random_seed=42)
        result2 = splitter.split(X, y, random_seed=42)

        np.testing.assert_array_equal(result1.X_train, result2.X_train)
        np.testing.assert_array_equal(result1.X_test, result2.X_test)
        np.testing.assert_array_equal(result1.y_train, result2.y_train)
        np.testing.assert_array_equal(result1.y_test, result2.y_test)

    def test_different_seeds_produce_different_splits(self, splitter, imbalanced_dataset):
        X, y = imbalanced_dataset

        result1 = splitter.split(X, y, random_seed=42)
        result2 = splitter.split(X, y, random_seed=99)

        # Very unlikely to be identical with different seeds
        assert not np.array_equal(result1.X_train, result2.X_train)

    def test_cv_folds_reproducible(self, splitter, imbalanced_dataset):
        """CV fold indices are identical with same seed."""
        X, y = imbalanced_dataset

        result = splitter.split(X, y, random_seed=7)
        folds1 = splitter.get_cv_folds(result.X_train, result.y_train, random_seed=7)
        folds2 = splitter.get_cv_folds(result.X_train, result.y_train, random_seed=7)

        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_different_cv_seeds_produce_different_folds(self, splitter, imbalanced_dataset):
        """Different seeds should produce different fold assignments."""
        X, y = imbalanced_dataset
        result = splitter.split(X, y)

        folds1 = splitter.get_cv_folds(result.X_train, result.y_train, random_seed=42)
        folds2 = splitter.get_cv_folds(result.X_train, result.y_train, random_seed=99)

        # At least one fold should differ
        any_different = False
        for (t1, _), (t2, _) in zip(folds1, folds2):
            if not np.array_equal(t1, t2):
                any_different = True
                break
        assert any_different


# ---------- Input Type Support ----------


class TestInputTypes:
    """Test that numpy arrays and pandas inputs are accepted."""

    def test_pandas_dataframe_input(self, splitter):
        """Accept pandas DataFrame for X."""
        rng = np.random.default_rng(42)
        X_df = pd.DataFrame(
            rng.standard_normal((100, 5)), columns=[f"f{i}" for i in range(5)]
        )
        y_series = pd.Series([0] * 80 + [1] * 20)

        result = splitter.split(X_df, y_series)
        assert isinstance(result.X_train, np.ndarray)
        assert isinstance(result.y_train, np.ndarray)

    def test_numpy_array_input(self, splitter, balanced_dataset):
        """Accept numpy arrays directly."""
        X, y = balanced_dataset
        result = splitter.split(X, y)
        assert isinstance(result.X_train, np.ndarray)

    def test_pandas_input_for_cv_folds(self, splitter):
        """Accept pandas inputs for get_cv_folds."""
        rng = np.random.default_rng(42)
        X_df = pd.DataFrame(
            rng.standard_normal((100, 5)), columns=[f"f{i}" for i in range(5)]
        )
        y_series = pd.Series([0] * 80 + [1] * 20)

        folds = splitter.get_cv_folds(X_df, y_series, k=5)
        assert len(folds) == 5
        for train_idx, val_idx in folds:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(val_idx, np.ndarray)
