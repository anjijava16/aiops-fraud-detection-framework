"""Unit tests for HealthRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.health_router import HealthResponse, HealthRouter


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_alive_response(self):
        """Test creating an alive health response."""
        resp = HealthResponse(status="alive")
        assert resp.status == "alive"
        assert resp.failing_components is None

    def test_not_ready_response(self):
        """Test creating a not_ready response with failing components."""
        resp = HealthResponse(status="not_ready", failing_components=["model", "mlflow"])
        assert resp.status == "not_ready"
        assert resp.failing_components == ["model", "mlflow"]


class TestHealthRouter:
    """Tests for HealthRouter endpoints."""

    def _create_app(self):
        """Create a test app with the health router."""
        health = HealthRouter()
        app = FastAPI()
        app.include_router(health.router)
        return TestClient(app), health

    def test_liveness_returns_alive(self):
        """Test that /health/live always returns 200 with status 'alive'."""
        client, _ = self._create_app()

        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_all_checks_pass(self):
        """Test that /health/ready returns 200 when all checks pass."""
        client, health = self._create_app()
        health.register_check("model", lambda: True)
        health.register_check("mlflow", lambda: True)

        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_readiness_check_fails(self):
        """Test that /health/ready returns 503 when a check fails."""
        client, health = self._create_app()
        health.register_check("model", lambda: True)
        health.register_check("database", lambda: False)

        response = client.get("/health/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert "database" in data["failing_components"]

    def test_readiness_check_raises_exception(self):
        """Test that a check raising an exception is treated as failure."""
        client, health = self._create_app()
        health.register_check("flaky", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        response = client.get("/health/ready")

        assert response.status_code == 503
        data = response.json()
        assert "flaky" in data["failing_components"]

    def test_readiness_no_checks_returns_ready(self):
        """Test that /health/ready returns 200 when no checks are registered."""
        client, _ = self._create_app()

        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_register_and_unregister_check(self):
        """Test registering and unregistering readiness checks."""
        client, health = self._create_app()
        health.register_check("temp", lambda: False)

        response = client.get("/health/ready")
        assert response.status_code == 503

        health.unregister_check("temp")

        response = client.get("/health/ready")
        assert response.status_code == 200
