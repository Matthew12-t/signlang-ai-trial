"""Speech-to-Text HTTP API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from src.shared.errors import ServiceError
from src.shared.models import TranscriptionResponse
from src.shared.security import require_internal_api_key
from src.stt.service import STTService

router = APIRouter(
    prefix="/v1/stt",
    tags=["Speech-to-Text"],
    dependencies=[Depends(require_internal_api_key)],
)


def _service(request: Request) -> STTService:
    service = getattr(request.app.state, "stt_service", None)
    if service is None:
        raise ServiceError(
            code="MODEL_UNAVAILABLE",
            message="The Speech-to-Text service is disabled.",
            status_code=503,
            retryable=True,
        )
    return service


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse,
    response_model_by_alias=True,
)
async def create_transcription(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    encoding: str = Form(default="wav"),
    timestamps: bool = Form(default=True),
) -> TranscriptionResponse:
    normalized_encoding = encoding.lower().strip()
    allowed_content_types = {
        "wav": {"audio/wav", "audio/x-wav", "application/octet-stream"},
        "webm": {"audio/webm", "video/webm", "application/octet-stream"},
        "flac": {"audio/flac", "audio/x-flac", "application/octet-stream"},
    }
    if normalized_encoding not in allowed_content_types:
        raise ServiceError(
            code="UNSUPPORTED_MEDIA_TYPE",
            message=f"Audio encoding '{encoding}' is not supported.",
            status_code=415,
        )
    if audio.content_type and audio.content_type not in allowed_content_types[
        normalized_encoding
    ]:
        raise ServiceError(
            code="UNSUPPORTED_MEDIA_TYPE",
            message="The uploaded content type does not match the audio encoding.",
            status_code=415,
            details={"contentType": audio.content_type, "encoding": normalized_encoding},
        )

    service = _service(request)
    content = await audio.read(service.settings.stt_max_audio_bytes + 1)
    await audio.close()
    return await service.transcribe_audio(
        content,
        language=language.lower().strip(),
        encoding=normalized_encoding,
        include_timestamps=timestamps,
        request_id=request.state.request_id,
    )
