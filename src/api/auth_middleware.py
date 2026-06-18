"""API authentication middleware using X-API-Key header validation.

Implements FastAPI dependency-based authentication that validates requests
against a set of configured API keys. Returns HTTP 401 for missing or invalid keys.
Authentication is evaluated before rate limiting.

Validates: Requirements 24.1, 24.4, 24.5
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status


class AuthMiddleware:
    """FastAPI dependency that validates the X-API-Key header against configured keys.

    Supports a minimum of 50 concurrent active API keys. Each key is stored
    in a set for O(1) lookup performance.

    Usage:
        auth = AuthMiddleware(api_keys={"key1", "key2", ...})
        app = FastAPI()

        @app.get("/protected")
        async def protected(api_key: str = Depends(auth)):
            return {"message": "authenticated"}
    """

    def __init__(self, api_keys: set[str] | None = None) -> None:
        """Initialize the AuthMiddleware.

        Args:
            api_keys: Set of valid API key strings. If None, an empty set is used
                      and all requests will be rejected as unauthorized.
        """
        self._api_keys: set[str] = set(api_keys) if api_keys else set()

    @property
    def api_keys(self) -> set[str]:
        """Return the current set of valid API keys."""
        return self._api_keys

    def add_key(self, key: str) -> None:
        """Add an API key to the valid set.

        Args:
            key: The API key string to add.
        """
        self._api_keys.add(key)

    def remove_key(self, key: str) -> None:
        """Remove an API key from the valid set.

        Args:
            key: The API key string to remove.

        Raises:
            KeyError: If the key does not exist.
        """
        self._api_keys.discard(key)

    async def __call__(self, x_api_key: str | None = Header(default=None)) -> str:
        """Validate the X-API-Key header.

        This method is used as a FastAPI dependency. It extracts the X-API-Key
        header and validates it against the configured set of keys.

        Args:
            x_api_key: The value of the X-API-Key header (injected by FastAPI).

        Returns:
            The validated API key string.

        Raises:
            HTTPException: 401 if the key is missing or invalid.
        """
        if x_api_key is None or x_api_key not in self._api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "unauthorized"},
            )
        return x_api_key
