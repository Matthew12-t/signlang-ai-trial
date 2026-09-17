"""Text-to-Speech HTTP API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel

from src.shared.errors import ServiceError
from src.shared.security import require_internal_api_key
from src.tts.service import TTSService

router = APIRouter(
    prefix="/v1/tts",
    tags=["Text-to-Speech"],
    dependencies=[Depends(require_internal_api_key)],
)


class SynthesisRequest(BaseModel):
    text: str
    language: str = "en"
    voice: str | None = None
    format: str = "wav"


def _service(request: Request) -> TTSService:
    service = getattr(request.app.state, "tts_service", None)
    if service is None:
        raise ServiceError(
            code="MODEL_UNAVAILABLE",
            message="The Text-to-Speech service is disabled.",
            status_code=503,
            retryable=True,
        )
    return service


@router.post(
    "/synthesize",
    responses={200: {"content": {"audio/wav": {}}}},
)
async def synthesize_speech(
    payload: SynthesisRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
) -> Response:
    artifact = await _service(request).synthesize_speech(
        payload.text,
        language=payload.language.lower().strip(),
        voice=payload.voice,
        output_format=payload.format,
        idempotency_key=idempotency_key,
    )
    return Response(
        content=artifact.content,
        media_type=artifact.media_type,
        headers={
            "X-Request-ID": request.state.request_id,
            "X-Model-ID": artifact.model_id,
            "X-Latency-Ms": str(artifact.latency_ms),
            "X-Audio-Sample-Rate": str(artifact.sample_rate),
        },
    )
