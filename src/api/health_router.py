"""FastAPI health endpoints - /health/live and /health/ready.

Provides HealthRouter with liveness and readiness probes for container
orchestration and load balancer health checks.

Validates: Requirements 23.1, 23.2
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, status
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response schema for health endpoints."""

    status: str  # "alive", "ready", "not_ready"
    failing_components: list[str] | None = None


class HealthRouter:
    """Router for /health/live and /health/ready endpoints.

    /health/live - Simple liveness probe (always returns 200 if app is running).
    /health/ready - Readiness probe that checks critical components (model, dependencies).

    Usage:
        health = HealthRouter()
        health.register_check("model", lambda: model is not None)
        health.register_check("mlflow", lambda: mlflow_client.is_connected())
        app.include_router(health.router)
    """

    def __init__(self) -> None:
        """Initialize the HealthRouter with empty readiness checks."""
        self._router = APIRouter(tags=["health"])
        self._readiness_checks: dict[str, Callable[[], bool]] = {}
        self._setup_routes()

    @property
    def router(self) -> APIRouter:
        """Return the FastAPI APIRouter."""
        return self._router

    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a readiness check function.

        Args:
            name: Name of the component being checked.
            check_fn: Callable that returns True if the component is ready.
        """
        self._readiness_checks[name] = check_fn

    def unregister_check(self, name: str) -> None:
        """Remove a readiness check.

        Args:
            name: Name of the component check to remove.
        """
        self._readiness_checks.pop(name, None)

    def _setup_routes(self) -> None:
        """Register the /health/live and /health/ready endpoints."""

        @self._router.get(
            "/health/live",
            response_model=HealthResponse,
            status_code=status.HTTP_200_OK,
        )
        async def liveness() -> HealthResponse:
            """Liveness probe - returns 200 if the application process is running."""
            return HealthResponse(status="alive")

        @self._router.get(
            "/health/ready",
            response_model=HealthResponse,
            responses={
                200: {"description": "Service is ready"},
                503: {"description": "Service is not ready"},
            },
        )
        async def readiness() -> HealthResponse:
            """Readiness probe - checks all registered components.

            Returns 200 if all checks pass, 503 with failing components otherwise.
            """
            failing: list[str] = []

            for name, check_fn in self._readiness_checks.items():
                try:
                    if not check_fn():
                        failing.append(name)
                except Exception:
                    failing.append(name)

            if failing:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=HealthResponse(
                        status="not_ready",
                        failing_components=failing,
                    ).model_dump(),
                )

            return HealthResponse(status="ready")
