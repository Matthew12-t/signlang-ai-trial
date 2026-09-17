"""Conversation Recall HTTP API routes."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Header, Request, Security
from fastapi.security import APIKeyHeader

from src.shared.models import (
    ErrorEnvelope,
    ModelDescriptor,
    RecallQueryRequest,
    RecallQueryResponse,
)
from src.shared.observability import get_logger


router = APIRouter(prefix="/v1/recall", tags=["recall"])
logger = get_logger(__name__)

_internal_api_key_header = APIKeyHeader(
    name="X-Internal-API-Key",
    scheme_name="InternalApiKey",
    description="Required only when INTERNAL_API_KEY is configured.",
    auto_error=False,
)
_request_id_header = {
    "description": "Caller-supplied request ID or a generated UUID.",
    "schema": {"type": "string"},
}
_responses = {
    200: {"headers": {"X-Request-ID": _request_id_header}},
    **{
        status: {
            "model": ErrorEnvelope,
            "headers": {"X-Request-ID": _request_id_header},
        }
        for status in (400, 401, 413, 422, 429, 502, 503, 504)
    },
}


@router.post(
    "/query",
    response_model=RecallQueryResponse,
    responses=_responses,
    openapi_extra={"security": [{}]},
)
async def query_recall(
    body: RecallQueryRequest,
    request: Request,
    _request_id: Annotated[
        str | None,
        Header(alias="X-Request-ID", description="Optional request ID."),
    ] = None,
    _internal_api_key: Annotated[str | None, Security(_internal_api_key_header)] = None,
) -> RecallQueryResponse:
    settings = request.app.state.settings
    started = time.perf_counter()
    result = await request.app.state.recall_service.query(body)
    latency_ms = max(0, int((time.perf_counter() - started) * 1000))
    logger.info(
        "recall_query request_id=%s provider=%s model=%s status=200 latency_ms=%d "
        "context_entries=%d context_chars=%d grounded=%s",
        request.state.request_id,
        settings.hf_provider,
        settings.recall_model,
        latency_ms,
        len(body.context_entries),
        sum(len(entry.text) for entry in body.context_entries),
        str(result.grounded).lower(),
    )
    return RecallQueryResponse(
        requestId=request.state.request_id,
        answer=result.answer,
        grounded=result.grounded,
        evidence=result.evidence,
        notFoundReason=result.not_found_reason,
        model=ModelDescriptor(provider="huggingface", id=settings.recall_model),
        latencyMs=latency_ms,
    )

