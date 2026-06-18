"""FastAPI prediction endpoint - POST /predict returns fraud_score, is_fraud, X-Model-Version.

Provides the PredictionRouter with a /predict endpoint that accepts transaction
features, runs inference through the ensemble model, and returns fraud assessment.

Validates: Requirements 21.1, 21.2, 21.3
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for the /predict endpoint.

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


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    fraud_score: float = Field(ge=0.0, le=1.0)
    is_fraud: bool
    model_version: str


class PredictionRouter:
    """Router for the /predict endpoint.

    Accepts transaction features, runs inference through the ensemble model,
    and returns the fraud score, binary classification, and model version.
    Sets the X-Model-Version response header.

    Usage:
        predictor = PredictionRouter(model=ensemble_model)
        app.include_router(predictor.router)
    """

    def __init__(self, model: Any = None, model_version: str = "0.0.0") -> None:
        """Initialize the PredictionRouter.

        Args:
            model: The ensemble model for inference. If None, a stub is used.
            model_version: Current model version string.
        """
        self._model = model
        self._model_version = model_version
        self._router = APIRouter(tags=["prediction"])
        self._setup_routes()

    @property
    def router(self) -> APIRouter:
        """Return the FastAPI APIRouter."""
        return self._router

    @property
    def model_version(self) -> str:
        """Return the current model version."""
        return self._model_version

    @model_version.setter
    def model_version(self, version: str) -> None:
        """Update the model version."""
        self._model_version = version

    def set_model(self, model: Any, version: str) -> None:
        """Update the model and its version.

        Args:
            model: New model instance.
            version: New model version string.
        """
        self._model = model
        self._model_version = version

    def _setup_routes(self) -> None:
        """Register the /predict endpoint on the router."""

        @self._router.post(
            "/predict",
            response_model=PredictionResponse,
            status_code=status.HTTP_200_OK,
        )
        async def predict(
            request: PredictionRequest,
            response: Response,
        ) -> PredictionResponse:
            """Run fraud inference on transaction features.

            Returns fraud_score, is_fraud flag, and sets X-Model-Version header.
            """
            if self._model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"error": "Model not loaded"},
                )

            # Extract features as dict
            features = request.model_dump()

            # Run inference
            try:
                prediction = self._model.predict_single(features)
                fraud_score = float(prediction.fraud_score)
                is_fraud = bool(prediction.is_fraud)
            except AttributeError:
                # Fallback for models with different interface
                try:
                    import numpy as np

                    feature_values = [
                        features["Time"],
                        *[features[f"V{i}"] for i in range(1, 29)],
                        features["Amount"],
                    ]
                    X = np.array([feature_values])
                    fraud_score = float(self._model.predict_proba(X)[0])
                    is_fraud = fraud_score >= 0.5
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail={"error": f"Inference failed: {str(e)}"},
                    )

            # Set model version header
            response.headers["X-Model-Version"] = self._model_version

            return PredictionResponse(
                fraud_score=fraud_score,
                is_fraud=is_fraud,
                model_version=self._model_version,
            )
