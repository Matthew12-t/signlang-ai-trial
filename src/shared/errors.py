"""Normalized service errors and FastAPI exception handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}


# Backwards-compatible domain name used by the Gloss module.
AppError = ServiceError


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _payload(error: ServiceError, request: Request) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
            "requestId": _request_id(request),
            "details": error.details,
        }
    }


def error_response(
    request: Request,
    error: ServiceError,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    response_headers = {"X-Request-ID": _request_id(request)}
    if headers:
        response_headers.update(headers)
    return JSONResponse(
        status_code=error.status_code,
        content=_payload(error, request),
        headers=response_headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request, error: ServiceError
    ) -> JSONResponse:
        logger.warning(
            "Service error code=%s status=%s requestId=%s path=%s",
            error.code,
            error.status_code,
            _request_id(request),
            request.url.path,
        )
        return error_response(request, error)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        unsupported_language = any(
            item.get("loc") == ("body", "language")
            and item.get("type") == "literal_error"
            for item in error.errors()
        )
        normalized = ServiceError(
            code="UNSUPPORTED_LANGUAGE" if unsupported_language else "INVALID_REQUEST",
            message=(
                "Only English is supported."
                if unsupported_language
                else "The request is invalid."
            ),
            status_code=422 if unsupported_language else 400,
        )
        return error_response(request, normalized)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        if error.status_code == 404:
            normalized = ServiceError("NOT_FOUND", "Resource not found.", 404)
        elif error.status_code == 405:
            normalized = ServiceError("METHOD_NOT_ALLOWED", "Method not allowed.", 405)
        else:
            normalized = ServiceError(
                "HTTP_ERROR", "The request could not be completed.", error.status_code
            )
        return error_response(request, normalized, error.headers)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, error: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled service error", exc_info=error)
        normalized = ServiceError(
            code="INTERNAL_ERROR",
            message="An internal error occurred.",
            status_code=500,
        )
        return error_response(request, normalized)
