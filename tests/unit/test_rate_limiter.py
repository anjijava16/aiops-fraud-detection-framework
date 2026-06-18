"""Unit tests for the RateLimiter fixed-window rate limiting.

Validates: Requirements 24.2, 24.3, 24.4
"""

import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from src.api.rate_limiter import RateLimiter


class FakeResponse:
    """Fake response object to capture header assignments."""

    def __init__(self):
        self.headers = {}


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_default_config(self):
        """RateLimiter defaults to 100 requests per 60-second window."""
        limiter = RateLimiter()
        assert limiter.requests_per_window == 100
        assert limiter.window_seconds == 60.0

    def test_custom_config(self):
        """RateLimiter accepts custom configuration."""
        limiter = RateLimiter(requests_per_window=50, window_seconds=30)
        assert limiter.requests_per_window == 50
        assert limiter.window_seconds == 30.0


class TestRateLimiterHeaders:
    """Tests for rate limit response headers."""

    def test_sets_limit_header(self):
        """Successful request includes X-RateLimit-Limit header."""
        limiter = RateLimiter(requests_per_window=100, window_seconds=60)
        response = FakeResponse()

        limiter.check("key-1", response)

        assert response.headers["X-RateLimit-Limit"] == "100"

    def test_sets_remaining_header(self):
        """Successful request includes X-RateLimit-Remaining header."""
        limiter = RateLimiter(requests_per_window=100, window_seconds=60)
        response = FakeResponse()

        limiter.check("key-1", response)

        assert response.headers["X-RateLimit-Remaining"] == "99"

    def test_remaining_decrements(self):
        """X-RateLimit-Remaining decrements with each request."""
        limiter = RateLimiter(requests_per_window=5, window_seconds=60)

        for i in range(5):
            response = FakeResponse()
            limiter.check("key-1", response)
            assert response.headers["X-RateLimit-Remaining"] == str(4 - i)


class TestRateLimiterEnforcement:
    """Tests for rate limit enforcement and HTTP 429."""

    def test_allows_requests_within_limit(self):
        """Requests within the limit are allowed."""
        limiter = RateLimiter(requests_per_window=3, window_seconds=60)

        for _ in range(3):
            response = FakeResponse()
            limiter.check("key-1", response)  # Should not raise

    def test_rejects_request_over_limit(self):
        """Request exceeding the limit returns HTTP 429."""
        limiter = RateLimiter(requests_per_window=3, window_seconds=60)

        # Consume all 3 allowed requests
        for _ in range(3):
            response = FakeResponse()
            limiter.check("key-1", response)

        # 4th request should be rejected
        response = FakeResponse()
        with pytest.raises(HTTPException) as exc_info:
            limiter.check("key-1", response)

        assert exc_info.value.status_code == 429

    def test_429_includes_retry_after_header(self):
        """HTTP 429 response includes a Retry-After header."""
        limiter = RateLimiter(requests_per_window=1, window_seconds=60)

        response = FakeResponse()
        limiter.check("key-1", response)  # Use the single allowed request

        with pytest.raises(HTTPException) as exc_info:
            response = FakeResponse()
            limiter.check("key-1", response)

        assert "Retry-After" in exc_info.value.headers
        retry_after = int(exc_info.value.headers["Retry-After"])
        assert 1 <= retry_after <= 60

    def test_429_includes_rate_limit_headers(self):
        """HTTP 429 response includes rate limit headers."""
        limiter = RateLimiter(requests_per_window=1, window_seconds=60)

        response = FakeResponse()
        limiter.check("key-1", response)

        with pytest.raises(HTTPException) as exc_info:
            response = FakeResponse()
            limiter.check("key-1", response)

        assert exc_info.value.headers["X-RateLimit-Limit"] == "1"
        assert exc_info.value.headers["X-RateLimit-Remaining"] == "0"


class TestRateLimiterKeyIsolation:
    """Tests for independent counters per API key."""

    def test_independent_counters(self):
        """Each API key maintains an independent request counter."""
        limiter = RateLimiter(requests_per_window=3, window_seconds=60)

        # key-1 uses all its requests
        for _ in range(3):
            response = FakeResponse()
            limiter.check("key-1", response)

        # key-2 should still have full quota
        response = FakeResponse()
        limiter.check("key-2", response)
        assert response.headers["X-RateLimit-Remaining"] == "2"

    def test_50_concurrent_keys(self):
        """Supports 50 concurrently active API keys with independent counters."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)

        # Make 1 request for each of 50 keys
        for i in range(50):
            response = FakeResponse()
            limiter.check(f"key-{i}", response)
            assert response.headers["X-RateLimit-Remaining"] == "9"

    def test_one_key_exhausted_others_unaffected(self):
        """Exhausting one key's limit does not affect other keys."""
        limiter = RateLimiter(requests_per_window=2, window_seconds=60)

        # Exhaust key-1
        for _ in range(2):
            response = FakeResponse()
            limiter.check("key-1", response)

        # key-1 is now limited
        with pytest.raises(HTTPException) as exc_info:
            response = FakeResponse()
            limiter.check("key-1", response)
        assert exc_info.value.status_code == 429

        # key-2 should be unaffected
        response = FakeResponse()
        limiter.check("key-2", response)
        assert response.headers["X-RateLimit-Remaining"] == "1"


class TestRateLimiterWindowReset:
    """Tests for window expiration and reset behavior."""

    @patch("src.api.rate_limiter.time.time")
    def test_window_resets_after_expiration(self, mock_time):
        """Counter resets when the window expires."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter(requests_per_window=2, window_seconds=60)

        # Use both requests in the first window
        for _ in range(2):
            response = FakeResponse()
            limiter.check("key-1", response)

        # Advance time past the window
        mock_time.return_value = 1061.0

        # Should work again in new window
        response = FakeResponse()
        limiter.check("key-1", response)
        assert response.headers["X-RateLimit-Remaining"] == "1"

    def test_manual_reset(self):
        """Manual reset clears the window for a key."""
        limiter = RateLimiter(requests_per_window=2, window_seconds=60)

        # Use both requests
        for _ in range(2):
            response = FakeResponse()
            limiter.check("key-1", response)

        # Reset and try again
        limiter.reset("key-1")
        response = FakeResponse()
        limiter.check("key-1", response)
        assert response.headers["X-RateLimit-Remaining"] == "1"

    def test_reset_all(self):
        """reset_all clears all windows."""
        limiter = RateLimiter(requests_per_window=1, window_seconds=60)

        # Use request for two keys
        response = FakeResponse()
        limiter.check("key-1", response)
        response = FakeResponse()
        limiter.check("key-2", response)

        # Both should be limited
        with pytest.raises(HTTPException):
            response = FakeResponse()
            limiter.check("key-1", response)

        # Reset all
        limiter.reset_all()

        # Both should work again
        response = FakeResponse()
        limiter.check("key-1", response)
        response = FakeResponse()
        limiter.check("key-2", response)


class TestRateLimiterGetRemaining:
    """Tests for get_remaining utility method."""

    def test_full_remaining_no_requests(self):
        """get_remaining returns full limit when no requests made."""
        limiter = RateLimiter(requests_per_window=100, window_seconds=60)
        assert limiter.get_remaining("key-1") == 100

    def test_remaining_after_requests(self):
        """get_remaining decreases after requests."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)

        for _ in range(3):
            response = FakeResponse()
            limiter.check("key-1", response)

        assert limiter.get_remaining("key-1") == 7

    @patch("src.api.rate_limiter.time.time")
    def test_remaining_resets_after_window(self, mock_time):
        """get_remaining returns full limit after window expires."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)

        response = FakeResponse()
        limiter.check("key-1", response)

        # Advance past window
        mock_time.return_value = 1061.0
        assert limiter.get_remaining("key-1") == 10
