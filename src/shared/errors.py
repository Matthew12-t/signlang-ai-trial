"""Normalized service errors and FastAPI exception handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(
        self,
        *,
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


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request, error: ServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=_payload(error, request),
            headers={"X-Request-ID": _request_id(request)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        normalized = ServiceError(
            code="INVALID_REQUEST",
            message="The request payload or parameters are invalid.",
            status_code=400,
            details={"errors": jsonable_encoder(error.errors())},
        )
        return JSONResponse(
            status_code=normalized.status_code,
            content=_payload(normalized, request),
            headers={"X-Request-ID": _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, error: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled service error", exc_info=error)
        normalized = ServiceError(
            code="INTERNAL_ERROR",
            message="The service could not complete the request.",
            status_code=500,
        )
        return JSONResponse(
            status_code=normalized.status_code,
            content=_payload(normalized, request),
            headers={"X-Request-ID": _request_id(request)},
        )
