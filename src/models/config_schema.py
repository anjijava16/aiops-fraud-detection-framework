"""Configuration and preprocessing data models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing layer.

    Attributes:
        smote_target_ratio: Target minority-to-majority class ratio after SMOTE.
        scaler_type: Normalization scaler — "StandardScaler" or "MinMaxScaler".
        correlation_threshold: Absolute correlation above which features are removed.
        test_size: Fraction of data reserved for the test split.
        k_folds: Number of stratified cross-validation folds.
        random_seed: Seed for reproducible splitting.
    """

    smote_target_ratio: float = 1.0
    scaler_type: str = "StandardScaler"  # or "MinMaxScaler"
    correlation_threshold: float = 0.95
    test_size: float = 0.2
    k_folds: int = 5
    random_seed: int = 42


@dataclass
class DataSplits:
    """Container for train/test splits and cross-validation folds.

    Attributes:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_test: Test labels.
        cv_folds: List of (train_indices, val_indices) tuples for k-fold CV.
        removed_features: Feature names removed during correlation-based selection.
        scaler_params: Serializable scaler parameters for inference reproducibility.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    cv_folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    removed_features: list[str] = field(default_factory=list)
    scaler_params: dict[str, Any] = field(default_factory=dict)
