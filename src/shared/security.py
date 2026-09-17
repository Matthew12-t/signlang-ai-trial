"""Authentication helpers for calls between internal services."""

from __future__ import annotations

import hmac

from fastapi import Header, Request

from src.shared.errors import ServiceError


async def require_internal_api_key(
    request: Request,
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
) -> None:
    expected = request.app.state.settings.internal_api_key
    if expected is None:
        return
    if x_internal_api_key is None or not hmac.compare_digest(
        x_internal_api_key, expected
    ):
        raise ServiceError(
            code="UNAUTHORIZED",
            message="A valid internal API key is required.",
            status_code=401,
        )
