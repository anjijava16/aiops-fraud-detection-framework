"""Unit tests for the FastAPI app factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


def _make_valid_payload() -> dict:
    """Create a valid prediction request payload."""
    payload = {"Time": 0.0, "Amount": 100.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.1 * i
    return payload


class TestApp:
    """Tests for the FastAPI application factory."""

    def test_create_app_returns_fastapi(self):
        """Test that create_app returns a FastAPI instance."""
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_health_live_no_auth_required(self):
        """Test that /health/live is accessible without auth."""
        app = create_app(auth_enabled=True, api_keys={"test-key"})
        client = TestClient(app)

        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_health_ready_no_auth_required(self):
        """Test that /health/ready is accessible without auth."""
        app = create_app(auth_enabled=True, api_keys={"test-key"})
        client = TestClient(app)

        response = client.get("/health/ready")
        # 503 because model is None (check fails)
        assert response.status_code in (200, 503)

    def test_predict_requires_auth(self):
        """Test that /predict requires authentication."""
        app = create_app(auth_enabled=True, api_keys={"valid-key"})
        client = TestClient(app)
        payload = _make_valid_payload()

        # No auth header
        response = client.post("/predict", json=payload)
        assert response.status_code == 401

    def test_predict_with_valid_auth(self):
        """Test that /predict works with valid auth (even if model is None -> 503)."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.fraud_score = 0.5
        mock_prediction.is_fraud = False
        mock_model.predict_single.return_value = mock_prediction

        app = create_app(
            model=mock_model,
            model_version="1.0.0",
            auth_enabled=True,
            api_keys={"valid-key"},
        )
        client = TestClient(app)
        payload = _make_valid_payload()

        response = client.post(
            "/predict",
            json=payload,
            headers={"x-api-key": "valid-key"},
        )
        assert response.status_code == 200

    def test_predict_with_invalid_auth(self):
        """Test that /predict rejects invalid auth."""
        app = create_app(auth_enabled=True, api_keys={"valid-key"})
        client = TestClient(app)
        payload = _make_valid_payload()

        response = client.post(
            "/predict",
            json=payload,
            headers={"x-api-key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_predict_no_auth_mode(self):
        """Test that /predict works without auth when disabled."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.fraud_score = 0.3
        mock_prediction.is_fraud = False
        mock_model.predict_single.return_value = mock_prediction

        app = create_app(model=mock_model, model_version="1.0.0", auth_enabled=False)
        client = TestClient(app)
        payload = _make_valid_payload()

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
