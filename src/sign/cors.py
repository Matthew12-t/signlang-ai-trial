"""Browser-local CORS policy for the edge-local Sign Service."""

from __future__ import annotations

from collections.abc import Iterable

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware


class BrowserLocalCORSMiddleware:
    """Apply an explicit origin policy and conditional PNA support."""

    def __init__(
        self,
        app,
        *,
        allowed_origins: Iterable[str],
        allow_private_network: bool,
    ) -> None:
        self.allowed_origins = frozenset(allowed_origins)
        self.allow_private_network = allow_private_network
        self.cors = CORSMiddleware(
            app,
            allow_origins=list(self.allowed_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID"],
            allow_credentials=False,
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.cors(scope, receive, send)
            return

        request_headers = Headers(scope=scope)
        origin = request_headers.get("origin")
        allow_pna = (
            scope.get("method") == "OPTIONS"
            and self.allow_private_network
            and origin in self.allowed_origins
            and request_headers.get("access-control-request-private-network", "").lower() == "true"
        )

        async def send_with_pna(message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if allow_pna and message["status"] < 400:
                    headers["Access-Control-Allow-Private-Network"] = "true"
                else:
                    message["headers"] = [
                        (key, value)
                        for key, value in message.get("headers", [])
                        if key.lower() != b"access-control-allow-private-network"
                    ]
            await send(message)

        await self.cors(scope, receive, send_with_pna)
