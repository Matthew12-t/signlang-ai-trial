"""Gloss normalization HTTP API routes."""

import time
from typing import Annotated

from fastapi import APIRouter, Header, Request, Security
from fastapi.security import APIKeyHeader

from src.gloss.service import GlossValidationError
from src.shared.errors import AppError
from src.shared.models import ErrorEnvelope, GlossNormalizeRequest, GlossNormalizeResponse
from src.shared.observability import get_logger


router = APIRouter(prefix="/v1/gloss", tags=["gloss"])
logger = get_logger(__name__)

_internal_api_key_header = APIKeyHeader(
    name="X-Internal-API-Key",
    scheme_name="InternalApiKey",
    description="Required only when INTERNAL_API_KEY is configured.",
    auto_error=False,
)
_request_id_response_header = {
    "description": "Caller-supplied request ID or a generated UUID.",
    "schema": {"type": "string"},
}
_responses = {
    200: {"headers": {"X-Request-ID": _request_id_response_header}},
    400: {
        "model": ErrorEnvelope,
        "description": "Invalid request body or parameters.",
        "headers": {"X-Request-ID": _request_id_response_header},
    },
    401: {
        "model": ErrorEnvelope,
        "description": "Missing or invalid configured internal API key.",
        "headers": {"X-Request-ID": _request_id_response_header},
    },
    413: {
        "model": ErrorEnvelope,
        "description": "The request exceeds the configured token limit.",
        "headers": {"X-Request-ID": _request_id_response_header},
    },
    422: {
        "model": ErrorEnvelope,
        "description": "The requested language is unsupported.",
        "headers": {"X-Request-ID": _request_id_response_header},
    },
}

_VALIDATION_ERRORS = {
    "UNSUPPORTED_LANGUAGE": (422, "Only English is supported."),
    "PAYLOAD_TOO_LARGE": (413, "The request contains too many tokens."),
}


@router.post(
    "/normalize",
    response_model=GlossNormalizeResponse,
    responses=_responses,
    openapi_extra={"security": [{}]},
)
async def normalize_gloss(
    body: GlossNormalizeRequest,
    request: Request,
    _request_id: Annotated[
        str | None,
        Header(
            alias="X-Request-ID",
            description="Optional request ID; invalid values are replaced with a UUID.",
        ),
    ] = None,
    _internal_api_key: Annotated[str | None, Security(_internal_api_key_header)] = None,
) -> GlossNormalizeResponse:
    settings = request.app.state.settings
    started_at = time.perf_counter()
    try:
        result = await request.app.state.gloss_service.normalize(body)
    except GlossValidationError as error:
        status_code, message = _VALIDATION_ERRORS.get(
            error.code, (422, "The request is invalid.")
        )
        latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        logger.info(
            "gloss_normalize request_id=%s mode=%s status=%d latency_ms=%d token_count=%d",
            request.state.request_id,
            settings.gloss_mode,
            status_code,
            latency_ms,
            len(body.tokens),
        )
        raise AppError(error.code, message, status_code) from None

    latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    fallback = any(warning.startswith("LLM_FALLBACK_") for warning in result.warnings)
    warnings = ",".join(result.warnings) or "none"
    configured_provider = settings.hf_provider if settings.gloss_mode == "qwen" else "none"
    configured_model = settings.llm_model if settings.gloss_mode == "qwen" else "none"
    logger.info(
        "gloss_normalize request_id=%s mode=%s method=%s fallback=%s warnings=%s "
        "configured_provider=%s configured_model=%s status=200 latency_ms=%d token_count=%d",
        request.state.request_id,
        settings.gloss_mode,
        result.method,
        str(fallback).lower(),
        warnings,
        configured_provider,
        configured_model,
        latency_ms,
        len(body.tokens),
    )
    return GlossNormalizeResponse(
        requestId=request.state.request_id,
        utteranceId=body.utterance_id,
        text=result.text,
        method=result.method,
        sourceTokenIds=result.source_token_ids,
        warnings=result.warnings,
        latencyMs=latency_ms,
    )

