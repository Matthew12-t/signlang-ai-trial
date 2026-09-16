"""Shared application errors and sanitized HTTP handlers."""

from collections.abc import Mapping
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


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


def error_response(
    request: Request,
    error: AppError,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        headers=headers,
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
    return error_response(request, error)


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
    return error_response(request, app_error)


async def http_exception_handler(
    request: Request, error: StarletteHTTPException
) -> JSONResponse:
    if error.status_code == 404:
        app_error = AppError("NOT_FOUND", "Resource not found.", 404)
    elif error.status_code == 405:
        app_error = AppError("METHOD_NOT_ALLOWED", "Method not allowed.", 405)
    else:
        app_error = AppError(
            "HTTP_ERROR",
            "The request could not be completed.",
            error.status_code,
        )
    return error_response(request, app_error, headers=error.headers)

