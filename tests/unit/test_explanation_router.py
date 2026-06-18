"""Unit tests for ExplanationRouter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.explanation_router import (
    ExplanationRequest,
    ExplanationResponse,
    ExplanationRouter,
)


def _make_valid_payload() -> dict:
    """Create a valid explanation request payload."""
    payload = {"Time": 0.0, "Amount": 100.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.1 * i
    return payload


class TestExplanationResponse:
    """Tests for ExplanationResponse schema."""

    def test_valid_response(self):
        """Test creating a valid explanation response."""
        resp = ExplanationResponse(
            shap_values={"V14": -0.5, "V12": 0.3, "V10": -0.1},
            base_value=0.02,
            fraud_score=0.85,
            model_version="1.0.0",
        )
        assert resp.shap_values["V14"] == -0.5
        assert resp.base_value == 0.02
        assert resp.fraud_score == 0.85


class TestExplanationRouter:
    """Tests for ExplanationRouter endpoint."""

    def _create_app(self, model=None, shap_explainer=None, model_version="1.0.0"):
        """Create a test app with the explanation router."""
        router = ExplanationRouter(
            model=model,
            shap_explainer=shap_explainer,
            model_version=model_version,
        )
        app = FastAPI()
        app.include_router(router.router)
        return TestClient(app)

    def test_explain_success(self):
        """Test successful explanation returns sorted SHAP values."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.fraud_score = 0.85
        mock_model.predict_single.return_value = mock_prediction

        mock_explainer = MagicMock()
        mock_explanation = MagicMock()
        mock_explanation.shap_values = {"V14": -0.5, "V12": 0.3, "V10": -0.1, "V1": 0.05}
        mock_explanation.base_value = 0.02
        mock_explainer.explain_local.return_value = mock_explanation

        client = self._create_app(
            model=mock_model,
            shap_explainer=mock_explainer,
            model_version="2.0.0",
        )
        payload = _make_valid_payload()

        response = client.post("/explain", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["fraud_score"] == 0.85
        assert data["base_value"] == 0.02
        assert data["model_version"] == "2.0.0"

        # SHAP values should be sorted by absolute magnitude
        shap_keys = list(data["shap_values"].keys())
        assert shap_keys[0] == "V14"  # |−0.5| is largest
        assert shap_keys[1] == "V12"  # |0.3|

    def test_explain_no_model_returns_503(self):
        """Test that missing model returns 503."""
        client = self._create_app(model=None, shap_explainer=MagicMock())
        payload = _make_valid_payload()

        response = client.post("/explain", json=payload)
        assert response.status_code == 503

    def test_explain_no_shap_explainer_returns_503(self):
        """Test that missing SHAP explainer returns 503."""
        client = self._create_app(model=MagicMock(), shap_explainer=None)
        payload = _make_valid_payload()

        response = client.post("/explain", json=payload)
        assert response.status_code == 503

    def test_explain_invalid_payload_returns_422(self):
        """Test that invalid payload returns 422."""
        client = self._create_app(model=MagicMock(), shap_explainer=MagicMock())

        response = client.post("/explain", json={"Time": 0.0})
        assert response.status_code == 422

    def test_set_model_updates_version(self):
        """Test that set_model updates the model and version."""
        router = ExplanationRouter(model=None, model_version="0.0.0")
        mock_model = MagicMock()
        router.set_model(mock_model, "3.0.0")
        assert router._model_version == "3.0.0"
