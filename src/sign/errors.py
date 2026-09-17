"""Errors and the public error envelope for Sign Language Service."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class SignServiceError(Exception):
    """An expected service error that can be returned safely to callers."""

    code: str
    message: str
    status_code: int
    retryable: bool
    request_id: UUID | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def envelope(self, request_id: UUID | None = None) -> dict[str, Any]:
        resolved_request_id = request_id or self.request_id
        if resolved_request_id is None:
            raise ValueError("An error response requires a request ID")
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "requestId": str(resolved_request_id),
                "details": self.details,
            }
        }


def invalid_request(message: str, *, details: dict[str, Any] | None = None) -> SignServiceError:
    return SignServiceError("INVALID_REQUEST", message, 400, False, details=details or {})


def unauthorized(message: str = "Internal API key is missing or invalid.") -> SignServiceError:
    return SignServiceError("UNAUTHORIZED", message, 401, False)


def payload_too_large(message: str, *, details: dict[str, Any] | None = None) -> SignServiceError:
    return SignServiceError("PAYLOAD_TOO_LARGE", message, 413, False, details=details or {})


def unsupported_media_type(message: str) -> SignServiceError:
    return SignServiceError("UNSUPPORTED_MEDIA_TYPE", message, 415, False)


def vocabulary_mismatch(message: str, *, details: dict[str, Any] | None = None) -> SignServiceError:
    return SignServiceError(
        "VOCABULARY_VERSION_MISMATCH", message, 422, False, details=details or {}
    )


def inference_busy(message: str = "GPU inference queue is full.") -> SignServiceError:
    return SignServiceError("INFERENCE_BUSY", message, 429, True)


def inference_failed(message: str, *, details: dict[str, Any] | None = None) -> SignServiceError:
    return SignServiceError("INFERENCE_FAILED", message, 500, False, details=details or {})


def model_not_ready(
    message: str,
    *,
    request_id: UUID | None = None,
    details: dict[str, Any] | None = None,
) -> SignServiceError:
    return SignServiceError(
        "MODEL_NOT_READY", message, 503, True, request_id=request_id, details=details or {}
    )


def inference_timeout(message: str = "Inference exceeded the configured deadline.") -> SignServiceError:
    return SignServiceError("INFERENCE_TIMEOUT", message, 504, True)


class VideoDecodeError(Exception):
    """Raised when an accepted video cannot be decoded into frames."""
