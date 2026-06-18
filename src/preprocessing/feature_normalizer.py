"""Feature normalization using StandardScaler or MinMaxScaler.

Supports fitting on training data only and applying the same transformation
to validation/test data. Scaler parameters can be persisted and reloaded
for consistent normalization during inference.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.models.errors import PreprocessingError

logger = logging.getLogger(__name__)

SUPPORTED_SCALERS = {"StandardScaler", "MinMaxScaler"}


class FeatureNormalizer:
    """Normalize features using StandardScaler or MinMaxScaler.

    The normalizer fits on training data only and transforms both training
    and validation/test splits using the same fitted parameters. Scaler
    parameters can be saved and loaded for inference consistency.

    Parameters
    ----------
    scaler_type : str
        Type of scaler to use. Must be 'StandardScaler' or 'MinMaxScaler'.

    Raises
    ------
    PreprocessingError
        If an unsupported scaler type is provided.
    """

    def __init__(self, scaler_type: str = "StandardScaler") -> None:
        if scaler_type not in SUPPORTED_SCALERS:
            raise PreprocessingError(
                f"Unsupported scaler type '{scaler_type}'. "
                f"Supported types: {sorted(SUPPORTED_SCALERS)}",
                details={
                    "scaler_type": scaler_type,
                    "supported_types": sorted(SUPPORTED_SCALERS),
                },
            )

        self.scaler_type = scaler_type
        self._scaler = self._create_scaler(scaler_type)
        self._is_fitted = False

        logger.info("FeatureNormalizer initialized with scaler_type='%s'", scaler_type)

    @staticmethod
    def _create_scaler(scaler_type: str) -> StandardScaler | MinMaxScaler:
        """Create a scaler instance based on the type string."""
        if scaler_type == "StandardScaler":
            return StandardScaler()
        else:
            return MinMaxScaler()

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit the scaler on training data and transform it.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed training data with the same shape as input.

        Raises
        ------
        PreprocessingError
            If the input data is invalid (empty, non-numeric, etc.).
        """
        if X_train.size == 0:
            raise PreprocessingError(
                "Cannot fit scaler on empty data.",
                details={"shape": X_train.shape},
            )

        try:
            X_transformed = self._scaler.fit_transform(X_train)
            self._is_fitted = True
            logger.info(
                "Scaler fitted and transformed training data: shape=%s, scaler_type='%s'",
                X_train.shape,
                self.scaler_type,
            )
            return X_transformed
        except Exception as e:
            raise PreprocessingError(
                f"Failed to fit and transform data: {e}",
                details={"scaler_type": self.scaler_type, "shape": X_train.shape},
            ) from e

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the already-fitted scaler.

        Parameters
        ----------
        X : np.ndarray
            Feature array of shape (n_samples, n_features) to transform.

        Returns
        -------
        np.ndarray
            Transformed data with the same shape as input.

        Raises
        ------
        PreprocessingError
            If the scaler has not been fitted yet or if transform fails.
        """
        if not self._is_fitted:
            raise PreprocessingError(
                "Scaler has not been fitted. Call fit_transform() first.",
                details={"scaler_type": self.scaler_type},
            )

        try:
            X_transformed = self._scaler.transform(X)
            logger.debug(
                "Transformed data: shape=%s, scaler_type='%s'",
                X.shape,
                self.scaler_type,
            )
            return X_transformed
        except Exception as e:
            raise PreprocessingError(
                f"Failed to transform data: {e}",
                details={"scaler_type": self.scaler_type, "shape": X.shape},
            ) from e

    def save_params(self, path: str | Path) -> None:
        """Persist fitted scaler parameters to disk using joblib.

        Parameters
        ----------
        path : str or Path
            File path where the scaler will be saved.

        Raises
        ------
        PreprocessingError
            If the scaler has not been fitted or if serialization fails.
        """
        if not self._is_fitted:
            raise PreprocessingError(
                "Cannot save parameters: scaler has not been fitted.",
                details={"scaler_type": self.scaler_type},
            )

        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"scaler_type": self.scaler_type, "scaler": self._scaler},
                path,
            )
            logger.info("Scaler parameters saved to '%s'", path)
        except Exception as e:
            raise PreprocessingError(
                f"Failed to save scaler parameters: {e}",
                details={"path": str(path), "scaler_type": self.scaler_type},
            ) from e

    def load_params(self, path: str | Path) -> None:
        """Load previously saved scaler parameters from disk.

        Parameters
        ----------
        path : str or Path
            File path from which to load the scaler.

        Raises
        ------
        PreprocessingError
            If the file does not exist or loading fails.
        """
        path = Path(path)
        if not path.exists():
            raise PreprocessingError(
                f"Scaler file not found: '{path}'",
                details={"path": str(path)},
            )

        try:
            data = joblib.load(path)
            self.scaler_type = data["scaler_type"]
            self._scaler = data["scaler"]
            self._is_fitted = True
            logger.info(
                "Scaler parameters loaded from '%s' (type='%s')",
                path,
                self.scaler_type,
            )
        except Exception as e:
            raise PreprocessingError(
                f"Failed to load scaler parameters: {e}",
                details={"path": str(path)},
            ) from e

    @property
    def is_fitted(self) -> bool:
        """Whether the scaler has been fitted."""
        return self._is_fitted
