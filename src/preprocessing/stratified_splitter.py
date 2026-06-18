"""Stratified train-test splitting with k-fold cross-validation.

Provides reproducible stratified splitting that preserves class distribution
across train/test sets and cross-validation folds.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models.errors import PreprocessingError


@dataclass
class DataSplits:
    """Container for stratified split outputs."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    cv_folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


class StratifiedSplitter:
    """Stratified train-test splitter with k-fold cross-validation support.

    Creates an 80/20 stratified split preserving class ratio within 1% absolute
    tolerance, and supports k-fold stratified CV splits on the training set
    with each fold maintaining class ratio within 2% tolerance.

    Uses scikit-learn's train_test_split with stratify=y and StratifiedKFold
    for cross-validation.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> DataSplits:
        """Perform stratified train-test split.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Accepts numpy arrays or pandas DataFrames.
        y : array-like of shape (n_samples,)
            Target labels. Accepts numpy arrays or pandas Series.
        test_size : float
            Fraction of data reserved for testing. Default 0.2.
        random_seed : int
            Random seed for reproducibility. Default 42.

        Returns
        -------
        DataSplits
            Contains X_train, X_test, y_train, y_test, and cv_folds (empty list).

        Raises
        ------
        PreprocessingError
            If test_size is invalid or dataset is too small.
        """
        if not (0.0 < test_size < 1.0):
            raise PreprocessingError(
                f"test_size must be between 0.0 and 1.0 exclusive, got {test_size}",
                details={"parameter": "test_size", "value": test_size},
            )

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise PreprocessingError(
                f"X and y must have the same number of samples. "
                f"Got X: {X_arr.shape[0]}, y: {y_arr.shape[0]}",
                details={"X_samples": X_arr.shape[0], "y_samples": y_arr.shape[0]},
            )

        # Need at least 2 samples per class to stratify
        unique_classes, class_counts = np.unique(y_arr, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            if count < 2:
                raise PreprocessingError(
                    f"Class {cls} has only {count} sample(s), "
                    f"need at least 2 for stratified splitting.",
                    details={"class": int(cls), "class_count": int(count)},
                )

        # Perform stratified split using scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=random_seed,
            stratify=y_arr,
        )

        return DataSplits(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=[],
        )

    def get_cv_folds(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k: int = 5,
        random_seed: int = 42,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate stratified k-fold cross-validation indices on training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target labels.
        k : int
            Number of cross-validation folds (2-20 inclusive). Default 5.
        random_seed : int
            Random seed for reproducibility. Default 42.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, val_indices) tuples, one per fold.

        Raises
        ------
        PreprocessingError
            If k is outside [2, 20] or dataset is too small for k-fold.
        """
        if not (2 <= k <= 20):
            raise PreprocessingError(
                f"k must be between 2 and 20 inclusive, got {k}",
                details={"parameter": "k", "value": k},
            )

        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise PreprocessingError(
                f"X_train and y_train must have the same number of samples. "
                f"Got X_train: {X_arr.shape[0]}, y_train: {y_arr.shape[0]}",
                details={"X_samples": X_arr.shape[0], "y_samples": y_arr.shape[0]},
            )

        # Validate that each class has enough samples for k folds
        unique_classes, class_counts = np.unique(y_arr, return_counts=True)
        n_classes = len(unique_classes)

        for cls, count in zip(unique_classes, class_counts):
            if count < k:
                raise PreprocessingError(
                    f"Class {cls} has only {count} samples, "
                    f"need at least {k} for {k}-fold CV.",
                    details={
                        "class": int(cls),
                        "class_count": int(count),
                        "k": k,
                    },
                )

        # Also check minimum total samples
        min_required = k * n_classes
        if X_arr.shape[0] < min_required:
            raise PreprocessingError(
                f"Dataset too small for {k}-fold CV with {n_classes} classes. "
                f"Need at least {min_required} samples, got {X_arr.shape[0]}.",
                details={
                    "n_samples": X_arr.shape[0],
                    "k": k,
                    "n_classes": n_classes,
                    "min_required": min_required,
                },
            )

        # Use scikit-learn's StratifiedKFold
        skf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=random_seed,
        )

        folds = []
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            folds.append((train_idx, val_idx))

        return folds
