"""Gloss normalization HTTP API routes."""

import hmac
import time

from fastapi import APIRouter, Depends, Header, Request

from src.gloss.service import GlossValidationError
from src.shared.errors import AppError
from src.shared.models import GlossNormalizeRequest, GlossNormalizeResponse
from src.shared.observability import get_logger


router = APIRouter(prefix="/v1/gloss", tags=["gloss"])
logger = get_logger(__name__)

_VALIDATION_ERRORS = {
    "UNSUPPORTED_LANGUAGE": (422, "Only English is supported."),
    "PAYLOAD_TOO_LARGE": (413, "The request contains too many tokens."),
}


async def enforce_internal_key(
    request: Request,
    internal_api_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
) -> None:
    settings = request.app.state.settings
    configured_key = settings.internal_api_key
    if configured_key is not None and not hmac.compare_digest(
        configured_key.get_secret_value(), internal_api_key or ""
    ):
        logger.warning(
            "gloss_normalize request_id=%s mode=%s status=401",
            request.state.request_id,
            settings.gloss_mode,
        )
        raise AppError("UNAUTHORIZED", "Authentication is required.", 401)


@router.post("/normalize", response_model=GlossNormalizeResponse)
async def normalize_gloss(
    body: GlossNormalizeRequest,
    request: Request,
    _: None = Depends(enforce_internal_key),
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
    logger.info(
        "gloss_normalize request_id=%s mode=%s status=200 latency_ms=%d token_count=%d",
        request.state.request_id,
        settings.gloss_mode,
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

