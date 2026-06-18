"""LIME local surrogate explainer for fraud detection predictions.

Provides model-agnostic local explanations using LIME's tabular explainer,
enabling regulatory cross-validation against SHAP explanations.
"""

import logging
from typing import Any, Callable

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from src.models.errors import ExplainabilityError
from src.models.explainability import LIMEExplanation

logger = logging.getLogger(__name__)

# Validation constants
MIN_PERTURBATION_SAMPLES = 1000
MAX_PERTURBATION_SAMPLES = 50000
DEFAULT_PERTURBATION_SAMPLES = 5000
MIN_TOP_N = 1
MAX_TOP_N = 30
DEFAULT_TOP_N = 10
LOW_FIDELITY_THRESHOLD = 0.6


class LIMEExplainer:
    """LIME-based local surrogate explainer for individual predictions.

    Trains a local interpretable model around a specific prediction to
    explain feature contributions using LIME's tabular explainer.

    Attributes:
        training_data: Background dataset used for perturbation generation.
        feature_names: List of feature names corresponding to columns.
        num_samples: Number of perturbation samples for LIME.
        top_n: Number of top feature contributions to return.
        predict_fn: Model prediction function accepting numpy array, returning probabilities.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: list[str],
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_samples: int = DEFAULT_PERTURBATION_SAMPLES,
        top_n: int = DEFAULT_TOP_N,
    ):
        """Initialize the LIME explainer.

        Args:
            training_data: Background dataset (n_samples, n_features) for perturbation.
            feature_names: Feature names matching training_data columns.
            predict_fn: Model prediction function. Should accept a 2D numpy array
                and return an array of shape (n_samples, n_classes) with probabilities.
            num_samples: Number of perturbation samples (default 5000, range 1000-50000).
            top_n: Number of top features to return (default 10, range 1-30).

        Raises:
            ExplainabilityError: If parameters are out of valid range.
        """
        self._validate_num_samples(num_samples)
        self._validate_top_n(top_n)

        if training_data.ndim != 2:
            raise ExplainabilityError(
                "Training data must be 2-dimensional.",
                details={"ndim": training_data.ndim},
            )

        if len(feature_names) != training_data.shape[1]:
            raise ExplainabilityError(
                "Feature names length must match training data columns.",
                details={
                    "feature_names_length": len(feature_names),
                    "training_data_columns": training_data.shape[1],
                },
            )

        self.training_data = training_data
        self.feature_names = feature_names
        self.predict_fn = predict_fn
        self.num_samples = num_samples
        self.top_n = top_n

        # Initialize the LIME tabular explainer
        self._explainer = LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            mode="classification",
            discretize_continuous=True,
        )

        logger.info(
            "LIMEExplainer initialized with %d perturbation samples, top-%d features",
            self.num_samples,
            self.top_n,
        )

    def explain(
        self,
        instance: np.ndarray,
        transaction_id: str | None = None,
        num_samples: int | None = None,
        top_n: int | None = None,
    ) -> LIMEExplanation:
        """Generate a LIME explanation for a single instance.

        Args:
            instance: Feature vector for the transaction to explain (1D array).
            transaction_id: Optional transaction ID (for error messaging).
            num_samples: Override default perturbation samples for this explanation.
            top_n: Override default top-N features for this explanation.

        Returns:
            LIMEExplanation with feature contributions, fidelity score, and
            low-confidence flag.

        Raises:
            ExplainabilityError: If explanation computation fails.
        """
        samples = num_samples if num_samples is not None else self.num_samples
        n_features = top_n if top_n is not None else self.top_n

        # Validate overrides
        if num_samples is not None:
            self._validate_num_samples(samples)
        if top_n is not None:
            self._validate_top_n(n_features)

        # Ensure instance is 1D
        if instance.ndim != 1:
            raise ExplainabilityError(
                "Instance must be a 1-dimensional array.",
                details={"ndim": instance.ndim, "transaction_id": transaction_id},
            )

        if len(instance) != len(self.feature_names):
            raise ExplainabilityError(
                "Instance feature count does not match expected features.",
                details={
                    "instance_features": len(instance),
                    "expected_features": len(self.feature_names),
                    "transaction_id": transaction_id,
                },
            )

        try:
            explanation = self._explainer.explain_instance(
                data_row=instance,
                predict_fn=self.predict_fn,
                num_features=n_features,
                num_samples=samples,
            )

            # Extract feature contributions (feature_name, weight) pairs
            feature_contributions = explanation.as_list()

            # Extract the local fidelity score (R² of the surrogate model)
            fidelity_score = explanation.score

            # Determine low-confidence flag
            is_low_confidence = fidelity_score < LOW_FIDELITY_THRESHOLD

            if is_low_confidence:
                logger.warning(
                    "LIME explanation has low fidelity (R²=%.4f < %.1f) for transaction %s",
                    fidelity_score,
                    LOW_FIDELITY_THRESHOLD,
                    transaction_id or "unknown",
                )

            result = LIMEExplanation(
                feature_contributions=feature_contributions,
                fidelity_score=fidelity_score,
                is_low_confidence=is_low_confidence,
                num_perturbation_samples=samples,
            )

            logger.info(
                "LIME explanation generated for transaction %s: fidelity=%.4f, "
                "low_confidence=%s, top_%d features",
                transaction_id or "unknown",
                fidelity_score,
                is_low_confidence,
                n_features,
            )

            return result

        except Exception as e:
            raise ExplainabilityError(
                f"LIME explanation failed: {str(e)}",
                details={"transaction_id": transaction_id, "error_type": type(e).__name__},
            ) from e

    def explain_by_id(
        self,
        transaction_id: str,
        dataset: dict[str, np.ndarray],
        num_samples: int | None = None,
        top_n: int | None = None,
    ) -> LIMEExplanation:
        """Generate a LIME explanation for a transaction by its ID.

        Looks up the transaction in the provided dataset and generates an explanation.

        Args:
            transaction_id: The transaction ID to explain.
            dataset: Mapping of transaction_id -> feature vector (1D numpy array).
            num_samples: Override default perturbation samples.
            top_n: Override default top-N features.

        Returns:
            LIMEExplanation with feature contributions, fidelity, and confidence flag.

        Raises:
            ExplainabilityError: If transaction not found or explanation fails.
        """
        if transaction_id not in dataset:
            raise ExplainabilityError(
                f"Transaction not found: {transaction_id}",
                details={"transaction_id": transaction_id},
            )

        instance = dataset[transaction_id]
        return self.explain(
            instance=instance,
            transaction_id=transaction_id,
            num_samples=num_samples,
            top_n=top_n,
        )

    @staticmethod
    def _validate_num_samples(num_samples: int) -> None:
        """Validate perturbation samples parameter.

        Raises:
            ExplainabilityError: If num_samples is out of range.
        """
        if not isinstance(num_samples, int):
            raise ExplainabilityError(
                f"num_samples must be an integer, got {type(num_samples).__name__}.",
                details={"num_samples": num_samples},
            )
        if num_samples < MIN_PERTURBATION_SAMPLES or num_samples > MAX_PERTURBATION_SAMPLES:
            raise ExplainabilityError(
                f"num_samples must be between {MIN_PERTURBATION_SAMPLES} and "
                f"{MAX_PERTURBATION_SAMPLES}, got {num_samples}.",
                details={
                    "num_samples": num_samples,
                    "valid_range": f"[{MIN_PERTURBATION_SAMPLES}, {MAX_PERTURBATION_SAMPLES}]",
                },
            )

    @staticmethod
    def _validate_top_n(top_n: int) -> None:
        """Validate top-N features parameter.

        Raises:
            ExplainabilityError: If top_n is out of range.
        """
        if not isinstance(top_n, int):
            raise ExplainabilityError(
                f"top_n must be an integer, got {type(top_n).__name__}.",
                details={"top_n": top_n},
            )
        if top_n < MIN_TOP_N or top_n > MAX_TOP_N:
            raise ExplainabilityError(
                f"top_n must be between {MIN_TOP_N} and {MAX_TOP_N}, got {top_n}.",
                details={
                    "top_n": top_n,
                    "valid_range": f"[{MIN_TOP_N}, {MAX_TOP_N}]",
                },
            )
