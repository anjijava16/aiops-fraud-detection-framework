"""FastAPI explanation endpoint - POST /explain returns SHAP values, base_value, fraud_score.

Provides the ExplanationRouter with a /explain endpoint that returns sorted SHAP
feature contributions for a given transaction.

Validates: Requirements 22.1, 22.2
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field


class ExplanationRequest(BaseModel):
    """Request schema for the /explain endpoint.

    All 30 numeric fields (Time, V1-V28, Amount) are required.
    """

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0.0)


class ExplanationResponse(BaseModel):
    """Response schema for the /explain endpoint."""

    shap_values: dict[str, float]  # Sorted by |value| desc
    base_value: float
    fraud_score: float = Field(ge=0.0, le=1.0)
    model_version: str


class ExplanationRouter:
    """Router for the /explain endpoint.

    Returns sorted SHAP values (by absolute magnitude descending), base_value,
    and fraud_score for a given transaction input.

    Usage:
        explainer = ExplanationRouter(model=ensemble_model, shap_explainer=shap_engine)
        app.include_router(explainer.router)
    """

    def __init__(
        self,
        model: Any = None,
        shap_explainer: Any = None,
        model_version: str = "0.0.0",
    ) -> None:
        """Initialize the ExplanationRouter.

        Args:
            model: The ensemble model for inference.
            shap_explainer: SHAP explainer instance.
            model_version: Current model version string.
        """
        self._model = model
        self._shap_explainer = shap_explainer
        self._model_version = model_version
        self._router = APIRouter(tags=["explanation"])
        self._setup_routes()

    @property
    def router(self) -> APIRouter:
        """Return the FastAPI APIRouter."""
        return self._router

    def set_model(self, model: Any, version: str) -> None:
        """Update the model and version.

        Args:
            model: New model instance.
            version: New model version string.
        """
        self._model = model
        self._model_version = version

    def set_shap_explainer(self, explainer: Any) -> None:
        """Update the SHAP explainer.

        Args:
            explainer: New SHAP explainer instance.
        """
        self._shap_explainer = explainer

    def _setup_routes(self) -> None:
        """Register the /explain endpoint on the router."""

        @self._router.post(
            "/explain",
            response_model=ExplanationResponse,
            status_code=status.HTTP_200_OK,
        )
        async def explain(
            request: ExplanationRequest,
            response: Response,
        ) -> ExplanationResponse:
            """Generate SHAP explanation for a transaction.

            Returns sorted SHAP values, base_value, and fraud_score.
            """
            if self._model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"error": "Model not loaded"},
                )

            if self._shap_explainer is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"error": "SHAP explainer not available"},
                )

            features = request.model_dump()

            # Run inference for fraud_score
            try:
                import numpy as np

                feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
                feature_values = [features[name] for name in feature_names]
                X = np.array([feature_values])

                # Get prediction
                try:
                    prediction = self._model.predict_single(features)
                    fraud_score = float(prediction.fraud_score)
                except AttributeError:
                    fraud_score = float(self._model.predict_proba(X)[0])

                # Get SHAP explanation
                try:
                    explanation = self._shap_explainer.explain_local(self._model, X[0])
                    shap_values = explanation.shap_values
                    base_value = explanation.base_value
                except AttributeError:
                    # Fallback: compute SHAP values directly
                    shap_result = self._shap_explainer(X)
                    raw_values = shap_result.values[0] if hasattr(shap_result, 'values') else shap_result[0]
                    base_value = float(shap_result.base_values[0]) if hasattr(shap_result, 'base_values') else 0.0
                    shap_values = {
                        name: float(val)
                        for name, val in zip(feature_names, raw_values)
                    }

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": f"Explanation generation failed: {str(e)}"},
                )

            # Sort SHAP values by absolute magnitude descending
            sorted_shap = dict(
                sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            )

            response.headers["X-Model-Version"] = self._model_version

            return ExplanationResponse(
                shap_values=sorted_shap,
                base_value=base_value,
                fraud_score=fraud_score,
                model_version=self._model_version,
            )
