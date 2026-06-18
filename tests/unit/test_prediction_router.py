"""Unit tests for PredictionRouter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.prediction_router import PredictionRequest, PredictionResponse, PredictionRouter


def _make_valid_payload() -> dict:
    """Create a valid prediction request payload."""
    payload = {"Time": 0.0, "Amount": 100.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.1 * i
    return payload


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    def test_valid_request(self):
        """Test creating a valid prediction request."""
        payload = _make_valid_payload()
        request = PredictionRequest(**payload)
        assert request.Time == 0.0
        assert request.Amount == 100.0
        assert request.V1 == 0.1

    def test_negative_amount_rejected(self):
        """Test that negative Amount is rejected."""
        payload = _make_valid_payload()
        payload["Amount"] = -1.0
        with pytest.raises(Exception):
            PredictionRequest(**payload)


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """Test creating a valid prediction response."""
        resp = PredictionResponse(fraud_score=0.75, is_fraud=True, model_version="1.0.0")
        assert resp.fraud_score == 0.75
        assert resp.is_fraud is True
        assert resp.model_version == "1.0.0"


class TestPredictionRouter:
    """Tests for PredictionRouter endpoint."""

    def _create_app_with_model(self, model=None, model_version="1.0.0"):
        """Create a test app with the prediction router."""
        router = PredictionRouter(model=model, model_version=model_version)
        app = FastAPI()
        app.include_router(router.router)
        return TestClient(app)

    def test_predict_success(self):
        """Test successful prediction returns fraud_score and is_fraud."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.fraud_score = 0.85
        mock_prediction.is_fraud = True
        mock_model.predict_single.return_value = mock_prediction

        client = self._create_app_with_model(model=mock_model, model_version="2.0.0")
        payload = _make_valid_payload()

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["fraud_score"] == 0.85
        assert data["is_fraud"] is True
        assert data["model_version"] == "2.0.0"

    def test_predict_sets_model_version_header(self):
        """Test that X-Model-Version header is set in the response."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.fraud_score = 0.3
        mock_prediction.is_fraud = False
        mock_model.predict_single.return_value = mock_prediction

        client = self._create_app_with_model(model=mock_model, model_version="3.1.0")
        payload = _make_valid_payload()

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.headers.get("x-model-version") == "3.1.0"

    def test_predict_no_model_returns_503(self):
        """Test that missing model returns 503."""
        client = self._create_app_with_model(model=None)
        payload = _make_valid_payload()

        response = client.post("/predict", json=payload)

        assert response.status_code == 503

    def test_predict_invalid_payload_returns_422(self):
        """Test that invalid payload returns 422."""
        mock_model = MagicMock()
        client = self._create_app_with_model(model=mock_model)

        # Missing required fields
        response = client.post("/predict", json={"Time": 0.0})

        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self):
        """Test that a missing field returns 422 with field info."""
        mock_model = MagicMock()
        client = self._create_app_with_model(model=mock_model)

        payload = _make_valid_payload()
        del payload["V14"]

        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_set_model_updates_version(self):
        """Test that set_model updates the model and version."""
        router = PredictionRouter(model=None, model_version="0.0.0")
        mock_model = MagicMock()

        router.set_model(mock_model, "2.0.0")

        assert router.model_version == "2.0.0"
