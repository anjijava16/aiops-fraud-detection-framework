"""Correlation-based feature selection for the preprocessing layer.

Computes the Pearson correlation matrix for all numeric features and removes
one feature from each highly-correlated pair based on variance (lower variance
is removed). If variances are equal, the feature appearing later in column order
is removed.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

import logging
from typing import Optional

import pandas as pd

from src.models.errors import PreprocessingError

logger = logging.getLogger(__name__)


class CorrelationSelector:
    """Remove features with high pairwise Pearson correlation.

    Parameters
    ----------
    threshold : float, default=0.95
        Absolute correlation threshold above which one feature in a pair
        is removed. Must be in [0.5, 1.0].

    Attributes
    ----------
    removed_features_ : list[str]
        Features removed after calling `select`.
    removal_log_ : list[dict]
        Detailed log of each removal decision.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self._validate_threshold(threshold)
        self.threshold = threshold
        self.removed_features_: list[str] = []
        self.removal_log_: list[dict] = []

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        """Validate that threshold is within [0.5, 1.0].

        Raises
        ------
        PreprocessingError
            If threshold is outside the valid range.
        """
        if not isinstance(threshold, (int, float)):
            raise PreprocessingError(
                f"Correlation threshold must be numeric, got {type(threshold).__name__}",
                details={"threshold": threshold, "valid_range": "[0.5, 1.0]"},
            )
        if threshold < 0.5 or threshold > 1.0:
            raise PreprocessingError(
                f"Correlation threshold must be between 0.5 and 1.0 inclusive, got {threshold}",
                details={"threshold": threshold, "valid_range": "[0.5, 1.0]"},
            )

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features by removing highly correlated columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with numeric features.

        Returns
        -------
        pd.DataFrame
            DataFrame with highly correlated features removed.
        """
        self.removed_features_ = []
        self.removal_log_ = []

        # Select only numeric columns
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            logger.info("Fewer than 2 numeric features; no correlation filtering applied.")
            return df

        # Compute Pearson correlation matrix
        corr_matrix = numeric_df.corr(method="pearson")

        # Compute variance for each feature
        variances = numeric_df.var()

        # Track features to remove
        features_to_remove: set[str] = set()

        # Get column order for tiebreaking
        columns = list(numeric_df.columns)

        # Iterate over upper triangle of correlation matrix
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                feat_a = columns[i]
                feat_b = columns[j]

                # Skip if either feature already marked for removal
                if feat_a in features_to_remove or feat_b in features_to_remove:
                    continue

                corr_value = abs(corr_matrix.loc[feat_a, feat_b])

                if corr_value > self.threshold:
                    var_a = variances[feat_a]
                    var_b = variances[feat_b]

                    # Remove feature with lower variance
                    # If equal variance, remove later-in-column-order feature
                    if var_a < var_b:
                        removed = feat_a
                        retained = feat_b
                    elif var_b < var_a:
                        removed = feat_b
                        retained = feat_a
                    else:
                        # Equal variance: remove later in column order
                        # feat_b is always later since j > i
                        removed = feat_b
                        retained = feat_a

                    features_to_remove.add(removed)

                    removal_info = {
                        "removed_feature": removed,
                        "retained_feature": retained,
                        "correlation": float(corr_value),
                        "removed_variance": float(variances[removed]),
                        "retained_variance": float(variances[retained]),
                    }
                    self.removal_log_.append(removal_info)

                    logger.info(
                        "Removed feature '%s' (variance=%.6f) correlated with '%s' "
                        "(variance=%.6f) at |r|=%.4f (threshold=%.4f)",
                        removed,
                        variances[removed],
                        retained,
                        variances[retained],
                        corr_value,
                        self.threshold,
                    )

        self.removed_features_ = sorted(features_to_remove)

        if self.removed_features_:
            logger.info(
                "Correlation selector removed %d feature(s): %s",
                len(self.removed_features_),
                self.removed_features_,
            )
        else:
            logger.info("No features exceed correlation threshold %.4f.", self.threshold)

        # Return DataFrame with removed features dropped
        return df.drop(columns=list(features_to_remove))
