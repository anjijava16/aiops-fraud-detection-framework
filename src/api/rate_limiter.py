"""Fixed-window rate limiter for FastAPI with per-API-key counters.

Implements configurable rate limiting (default: 100 requests per 60-second window)
with proper HTTP headers (X-RateLimit-Limit, X-RateLimit-Remaining) and HTTP 429
responses with Retry-After header when limits are exceeded.

Validates: Requirements 24.2, 24.3, 24.4
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Response, status


@dataclass
class RateLimitWindow:
    """Tracks a rate limit window for a single API key."""

    window_start: float
    request_count: int = 0


class RateLimiter:
    """Fixed-window rate limiter that maintains independent counters per API key.

    Each API key gets its own request counter and window. When the window expires,
    the counter resets. Supports a minimum of 50 concurrently active API keys.

    Usage as a FastAPI dependency:
        rate_limiter = RateLimiter(requests_per_window=100, window_seconds=60)
        auth = AuthMiddleware(api_keys={"key1", "key2"})

        @app.get("/endpoint")
        async def endpoint(
            request: Request,
            response: Response,
            api_key: str = Depends(auth),
        ):
            rate_limiter.check(api_key, response)
            return {"data": "value"}
    """

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int | float = 60,
    ) -> None:
        """Initialize the RateLimiter.

        Args:
            requests_per_window: Maximum number of requests allowed per window.
            window_seconds: Duration of the rate limit window in seconds.
        """
        self._requests_per_window = requests_per_window
        self._window_seconds = float(window_seconds)
        self._windows: dict[str, RateLimitWindow] = {}

    @property
    def requests_per_window(self) -> int:
        """Return the configured max requests per window."""
        return self._requests_per_window

    @property
    def window_seconds(self) -> float:
        """Return the configured window duration in seconds."""
        return self._window_seconds

    def _get_or_create_window(self, api_key: str, now: float) -> RateLimitWindow:
        """Get the current window for an API key, creating or resetting if expired.

        Args:
            api_key: The API key to look up.
            now: Current timestamp.

        Returns:
            The active RateLimitWindow for this key.
        """
        window = self._windows.get(api_key)

        if window is None or (now - window.window_start) >= self._window_seconds:
            # Create a new window or reset an expired one
            window = RateLimitWindow(window_start=now, request_count=0)
            self._windows[api_key] = window

        return window

    def check(self, api_key: str, response: Response) -> None:
        """Check rate limit for the given API key and set response headers.

        This method should be called after authentication succeeds. It increments
        the request counter, sets rate limit headers, and raises HTTP 429 if the
        limit is exceeded.

        Args:
            api_key: The authenticated API key.
            response: The FastAPI Response object to set headers on.

        Raises:
            HTTPException: 429 with Retry-After header if rate limit is exceeded.
        """
        now = time.time()
        window = self._get_or_create_window(api_key, now)

        # Check if limit would be exceeded
        if window.request_count >= self._requests_per_window:
            retry_after = int(
                self._window_seconds - (now - window.window_start)
            )
            # Ensure retry_after is at least 1 second
            retry_after = max(1, retry_after)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "rate limit exceeded"},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self._requests_per_window),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Increment counter
        window.request_count += 1

        # Set rate limit headers on the response
        remaining = self._requests_per_window - window.request_count
        response.headers["X-RateLimit-Limit"] = str(self._requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

    def get_remaining(self, api_key: str) -> int:
        """Get the number of remaining requests for an API key in the current window.

        Args:
            api_key: The API key to check.

        Returns:
            Number of remaining requests, or the full limit if no window exists.
        """
        now = time.time()
        window = self._windows.get(api_key)

        if window is None or (now - window.window_start) >= self._window_seconds:
            return self._requests_per_window

        return max(0, self._requests_per_window - window.request_count)

    def reset(self, api_key: str) -> None:
        """Reset the rate limit window for a specific API key.

        Args:
            api_key: The API key to reset.
        """
        self._windows.pop(api_key, None)

    def reset_all(self) -> None:
        """Reset all rate limit windows."""
        self._windows.clear()
