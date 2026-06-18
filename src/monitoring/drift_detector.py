"""Drift detection using Population Stability Index (PSI) and Kolmogorov-Smirnov tests.

Monitors feature distributions for statistical drift against a reference distribution,
using a configurable sliding window of recent predictions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats

from src.models.monitoring import DriftResult

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift using PSI and KS-test on a sliding window of predictions.

    Attributes:
        psi_threshold: PSI value above which a feature is flagged as drifted.
        ks_p_value_threshold: p-value below which a feature is flagged (KS-test).
        psi_bins: Number of bins for PSI calculation.
        window_size: Maximum number of samples in the sliding window.
        min_samples: Minimum samples required before detection runs.
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: list[str],
        psi_threshold: float = 0.2,
        ks_p_value_threshold: float = 0.05,
        psi_bins: int = 10,
        window_size: int = 10_000,
        min_samples: int = 100,
    ) -> None:
        """Initialize the DriftDetector.

        Args:
            reference_data: Reference (training) data array of shape (n_samples, n_features).
            feature_names: List of feature names corresponding to columns.
            psi_threshold: PSI value above which drift is flagged (default 0.2).
            ks_p_value_threshold: KS-test p-value below which drift is flagged (default 0.05).
            psi_bins: Number of bins for PSI histogram (default 10).
            window_size: Sliding window size for recent predictions (default 10,000).
            min_samples: Minimum samples before detection is performed (default 100).
        """
        if reference_data.ndim != 2:
            raise ValueError("reference_data must be 2-dimensional")
        if reference_data.shape[1] != len(feature_names):
            raise ValueError(
                f"reference_data has {reference_data.shape[1]} columns "
                f"but {len(feature_names)} feature names provided"
            )

        self._reference_data = reference_data
        self._feature_names = feature_names
        self._psi_threshold = psi_threshold
        self._ks_p_value_threshold = ks_p_value_threshold
        self._psi_bins = psi_bins
        self._window_size = window_size
        self._min_samples = min_samples
        self._window: list[np.ndarray] = []

    @property
    def psi_threshold(self) -> float:
        return self._psi_threshold

    @property
    def ks_p_value_threshold(self) -> float:
        return self._ks_p_value_threshold

    @property
    def psi_bins(self) -> int:
        return self._psi_bins

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def min_samples(self) -> int:
        return self._min_samples

    @property
    def current_window_size(self) -> int:
        """Return the current number of samples in the sliding window."""
        return len(self._window)

    def add_prediction(self, features: np.ndarray) -> None:
        """Add a single prediction's feature vector to the sliding window.

        Args:
            features: 1-D array of feature values for a single prediction.
        """
        if features.ndim != 1:
            raise ValueError("features must be 1-dimensional")
        if len(features) != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} features, got {len(features)}"
            )

        self._window.append(features.copy())

        # Maintain sliding window size
        if len(self._window) > self._window_size:
            self._window = self._window[-self._window_size:]

    def add_batch(self, features_batch: np.ndarray) -> None:
        """Add a batch of prediction feature vectors to the sliding window.

        Args:
            features_batch: 2-D array of shape (n_samples, n_features).
        """
        if features_batch.ndim != 2:
            raise ValueError("features_batch must be 2-dimensional")
        if features_batch.shape[1] != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} features, "
                f"got {features_batch.shape[1]}"
            )

        for row in features_batch:
            self._window.append(row.copy())

        # Maintain sliding window size
        if len(self._window) > self._window_size:
            self._window = self._window[-self._window_size:]

    def detect(self) -> DriftResult:
        """Run drift detection on the current sliding window.

        Returns:
            DriftResult with per-feature PSI values, KS-test results,
            and list of drifted features.

        Notes:
            If the window contains fewer than min_samples, logs a warning
            and returns a no-drift result.
        """
        if len(self._window) < self._min_samples:
            logger.warning(
                "Drift detection skipped: window has %d samples, "
                "minimum required is %d",
                len(self._window),
                self._min_samples,
            )
            return DriftResult(
                is_drifted=False,
                drifted_features=[],
                psi_values={},
                ks_results={},
                detection_timestamp=datetime.now(timezone.utc).isoformat(),
            )

        current_data = np.array(self._window)
        drifted_features: list[str] = []
        psi_values: dict[str, float] = {}
        ks_results: dict[str, tuple[float, float]] = {}

        for i, feature_name in enumerate(self._feature_names):
            reference_col = self._reference_data[:, i]
            current_col = current_data[:, i]

            # Compute PSI
            psi = self._compute_psi(reference_col, current_col)
            psi_values[feature_name] = psi

            # Compute KS-test
            ks_stat, ks_p_value = stats.ks_2samp(reference_col, current_col)
            ks_results[feature_name] = (float(ks_stat), float(ks_p_value))

            # Flag if drifted by either criterion
            if psi > self._psi_threshold or ks_p_value < self._ks_p_value_threshold:
                drifted_features.append(feature_name)

        return DriftResult(
            is_drifted=len(drifted_features) > 0,
            drifted_features=drifted_features,
            psi_values=psi_values,
            ks_results=ks_results,
            detection_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_psi(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Compute Population Stability Index between reference and current distributions.

        PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)

        where P_i = proportion in bin i for current, Q_i = proportion for reference.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            PSI value (non-negative float).
        """
        # Create bins based on reference distribution
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        # Handle edge case where all values are the same
        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, self._psi_bins + 1)
        # Ensure first and last bins capture all values
        bin_edges[0] = min_val - 1e-10
        bin_edges[-1] = max_val + 1e-10

        # Compute proportions per bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions, avoid division by zero
        epsilon = 1e-10
        ref_proportions = (ref_counts + epsilon) / (len(reference) + epsilon * self._psi_bins)
        cur_proportions = (cur_counts + epsilon) / (len(current) + epsilon * self._psi_bins)

        # PSI formula
        psi = np.sum(
            (cur_proportions - ref_proportions)
            * np.log(cur_proportions / ref_proportions)
        )

        return float(psi)

    def reset_window(self) -> None:
        """Clear the sliding window."""
        self._window = []
