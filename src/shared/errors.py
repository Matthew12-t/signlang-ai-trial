"""Shared application errors and sanitized HTTP handlers."""

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
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


def _request_id(request: Request) -> str:
    return request.state.request_id


def _response(request: Request, error: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
                "requestId": _request_id(request),
                "details": error.details,
            }
        },
    )


async def app_error_handler(request: Request, error: AppError) -> JSONResponse:
    return _response(request, error)


async def request_validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    unsupported_language = any(
        validation_error.get("loc") == ("body", "language")
        and validation_error.get("type") == "literal_error"
        for validation_error in error.errors()
    )
    if unsupported_language:
        app_error = AppError(
            "UNSUPPORTED_LANGUAGE",
            "Only English is supported.",
            422,
        )
    else:
        app_error = AppError("INVALID_REQUEST", "The request is invalid.", 422)
    return _response(request, app_error)

