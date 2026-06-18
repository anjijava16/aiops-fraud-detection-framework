"""SMOTE oversampling for handling class imbalance in fraud detection.

Applies Synthetic Minority Over-sampling Technique (SMOTE) to the minority class
(Class=1) in the training split only, leaving validation and test splits unchanged.
"""

import logging

import numpy as np
from imblearn.over_sampling import SMOTE

from src.models.errors import PreprocessingError

logger = logging.getLogger(__name__)

# Minimum number of minority samples required for default SMOTE (k_neighbors=5)
MIN_MINORITY_SAMPLES = 6


class SMOTEResampler:
    """Applies SMOTE oversampling to balance class distributions in training data.

    Parameters
    ----------
    target_ratio : float
        Desired ratio of minority to majority class samples after resampling.
        A value of 1.0 means equal number of samples in both classes (1:1 ratio).
    k_neighbors : int
        Number of nearest neighbors used by SMOTE algorithm.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        target_ratio: float = 1.0,
        k_neighbors: int = 5,
        random_state: int | None = 42,
    ):
        self.target_ratio = target_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to the minority class.

        Parameters
        ----------
        X : np.ndarray
            Feature array of shape (n_samples, n_features).
        y : np.ndarray
            Label array of shape (n_samples,) with binary values (0 and 1).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Resampled (X, y) arrays with the minority class oversampled.

        Raises
        ------
        PreprocessingError
            If the minority class has fewer than MIN_MINORITY_SAMPLES samples.
        """
        # Compute original class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))

        minority_class = 1
        majority_class = 0

        minority_count = int(class_dist.get(minority_class, 0))
        majority_count = int(class_dist.get(majority_class, 0))

        logger.info(
            "Original class distribution: majority(0)=%d, minority(1)=%d",
            majority_count,
            minority_count,
        )

        # Validate minimum minority samples
        if minority_count < MIN_MINORITY_SAMPLES:
            raise PreprocessingError(
                f"Insufficient minority samples for SMOTE: found {minority_count}, "
                f"minimum required is {MIN_MINORITY_SAMPLES} (k_neighbors={self.k_neighbors})",
                details={
                    "minority_count": minority_count,
                    "min_required": MIN_MINORITY_SAMPLES,
                    "k_neighbors": self.k_neighbors,
                },
            )

        # Calculate target minority count based on ratio
        target_minority_count = int(majority_count * self.target_ratio)

        # Only resample if target exceeds current minority count
        if target_minority_count <= minority_count:
            logger.info(
                "Target minority count (%d) does not exceed current count (%d). "
                "No resampling needed.",
                target_minority_count,
                minority_count,
            )
            return X.copy(), y.copy()

        # Configure SMOTE with target sampling strategy
        sampling_strategy = {minority_class: target_minority_count}

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Log resampled distribution
        unique_new, counts_new = np.unique(y_resampled, return_counts=True)
        new_dist = dict(zip(unique_new, counts_new))

        logger.info(
            "Resampled class distribution: majority(0)=%d, minority(1)=%d",
            int(new_dist.get(majority_class, 0)),
            int(new_dist.get(minority_class, 0)),
        )

        return X_resampled, y_resampled
