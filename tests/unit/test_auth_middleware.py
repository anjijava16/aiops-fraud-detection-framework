"""Unit tests for the AuthMiddleware API key validation.

Validates: Requirements 24.1, 24.4, 24.5
"""

import pytest
from fastapi import HTTPException

from src.api.auth_middleware import AuthMiddleware


class TestAuthMiddlewareInit:
    """Tests for AuthMiddleware initialization."""

    def test_init_with_keys(self):
        """AuthMiddleware stores the provided API keys."""
        keys = {"key-1", "key-2", "key-3"}
        auth = AuthMiddleware(api_keys=keys)
        assert auth.api_keys == keys

    def test_init_with_none(self):
        """AuthMiddleware with None defaults to empty set."""
        auth = AuthMiddleware(api_keys=None)
        assert auth.api_keys == set()

    def test_init_no_args(self):
        """AuthMiddleware with no args defaults to empty set."""
        auth = AuthMiddleware()
        assert auth.api_keys == set()


class TestAuthMiddlewareKeyManagement:
    """Tests for add/remove key operations."""

    def test_add_key(self):
        """Adding a key makes it available for authentication."""
        auth = AuthMiddleware()
        auth.add_key("new-key")
        assert "new-key" in auth.api_keys

    def test_remove_key(self):
        """Removing a key prevents authentication with it."""
        auth = AuthMiddleware(api_keys={"key-to-remove"})
        auth.remove_key("key-to-remove")
        assert "key-to-remove" not in auth.api_keys

    def test_remove_nonexistent_key(self):
        """Removing a nonexistent key does not raise."""
        auth = AuthMiddleware(api_keys={"existing-key"})
        auth.remove_key("nonexistent-key")  # Should not raise


class TestAuthMiddlewareValidation:
    """Tests for the __call__ authentication validation."""

    @pytest.fixture
    def auth(self):
        """Create an AuthMiddleware with test keys."""
        return AuthMiddleware(api_keys={"valid-key-1", "valid-key-2", "valid-key-3"})

    @pytest.mark.asyncio
    async def test_valid_key_returns_key(self, auth):
        """A valid API key passes authentication and returns the key."""
        result = await auth(x_api_key="valid-key-1")
        assert result == "valid-key-1"

    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self, auth):
        """Missing X-API-Key header returns HTTP 401."""
        with pytest.raises(HTTPException) as exc_info:
            await auth(x_api_key=None)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == {"error": "unauthorized"}

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self, auth):
        """Invalid X-API-Key returns HTTP 401."""
        with pytest.raises(HTTPException) as exc_info:
            await auth(x_api_key="invalid-key")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == {"error": "unauthorized"}

    @pytest.mark.asyncio
    async def test_empty_string_key_returns_401(self, auth):
        """Empty string X-API-Key returns HTTP 401."""
        with pytest.raises(HTTPException) as exc_info:
            await auth(x_api_key="")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == {"error": "unauthorized"}

    @pytest.mark.asyncio
    async def test_authentication_before_rate_limit(self, auth):
        """Authentication validates the key without any rate-limit side effects."""
        # Auth should simply validate the key - no rate limit interaction
        result = await auth(x_api_key="valid-key-2")
        assert result == "valid-key-2"


class TestAuthMiddlewareConcurrentKeys:
    """Tests for supporting 50+ concurrent API keys."""

    @pytest.mark.asyncio
    async def test_supports_50_concurrent_keys(self):
        """AuthMiddleware supports a minimum of 50 concurrently active API keys."""
        keys = {f"api-key-{i}" for i in range(50)}
        auth = AuthMiddleware(api_keys=keys)

        # Validate each of the 50 keys works
        for key in keys:
            result = await auth(x_api_key=key)
            assert result == key

    @pytest.mark.asyncio
    async def test_supports_100_concurrent_keys(self):
        """AuthMiddleware supports well above 50 concurrent keys."""
        keys = {f"api-key-{i}" for i in range(100)}
        auth = AuthMiddleware(api_keys=keys)

        # All 100 keys should authenticate successfully
        for key in keys:
            result = await auth(x_api_key=key)
            assert result == key
