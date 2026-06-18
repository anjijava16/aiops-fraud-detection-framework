"""FastAPI application factory - wires all routers and middleware into the app.

Creates and configures the FastAPI application with prediction, explanation,
and health routers, plus authentication and rate limiting middleware.

Validates: Requirements 21.1, 22.1, 23.1, 24.1, 24.2
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Request, Response

from src.api.auth_middleware import AuthMiddleware
from src.api.explanation_router import ExplanationRouter
from src.api.health_router import HealthRouter
from src.api.prediction_router import PredictionRouter
from src.api.rate_limiter import RateLimiter


def create_app(
    model: Any = None,
    shap_explainer: Any = None,
    model_version: str = "0.0.0",
    api_keys: set[str] | None = None,
    requests_per_window: int = 100,
    window_seconds: int = 60,
    auth_enabled: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model: Ensemble model for inference.
        shap_explainer: SHAP explainer for /explain endpoint.
        model_version: Current model version string.
        api_keys: Set of valid API keys for authentication.
        requests_per_window: Rate limit max requests per window.
        window_seconds: Rate limit window duration in seconds.
        auth_enabled: Whether to enable API key authentication.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="AIOps Fraud Detection API",
        description="Real-time fraud detection with explainability",
        version=model_version,
    )

    # Initialize components
    auth = AuthMiddleware(api_keys=api_keys)
    rate_limiter = RateLimiter(
        requests_per_window=requests_per_window,
        window_seconds=window_seconds,
    )

    prediction_router = PredictionRouter(model=model, model_version=model_version)
    explanation_router = ExplanationRouter(
        model=model,
        shap_explainer=shap_explainer,
        model_version=model_version,
    )
    health_router = HealthRouter()

    # Register readiness checks
    health_router.register_check("model", lambda: model is not None)

    # Store components on app state for access
    app.state.auth = auth
    app.state.rate_limiter = rate_limiter
    app.state.prediction_router = prediction_router
    app.state.explanation_router = explanation_router
    app.state.health_router = health_router

    # Health endpoints (no auth required)
    app.include_router(health_router.router)

    if auth_enabled:
        # Protected endpoints with auth + rate limiting
        @app.middleware("http")
        async def auth_rate_limit_middleware(request: Request, call_next):
            """Apply auth and rate limiting to /predict and /explain endpoints."""
            # Skip auth for health endpoints
            if request.url.path.startswith("/health"):
                return await call_next(request)

            # Skip auth for docs
            if request.url.path in ("/docs", "/openapi.json", "/redoc"):
                return await call_next(request)

            # Authenticate
            api_key = request.headers.get("x-api-key")
            if api_key is None or api_key not in auth.api_keys:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={"detail": {"error": "unauthorized"}},
                )

            # Rate limit check
            response = Response()
            try:
                rate_limiter.check(api_key, response)
            except Exception as exc:
                from fastapi.responses import JSONResponse

                if hasattr(exc, "status_code") and exc.status_code == 429:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": {"error": "rate limit exceeded"}},
                        headers=dict(exc.headers) if hasattr(exc, "headers") else {},
                    )
                raise

            # Proceed with request
            actual_response = await call_next(request)

            # Copy rate limit headers
            for key, value in response.headers.items():
                if key.startswith("X-RateLimit"):
                    actual_response.headers[key] = value

            return actual_response

    # Include routers
    app.include_router(prediction_router.router)
    app.include_router(explanation_router.router)

    return app
