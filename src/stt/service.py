"""Speech-to-Text application service."""

from __future__ import annotations

import time
from typing import Protocol

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.models import (
    ModelDescriptor,
    ProviderTranscript,
    TranscriptionResponse,
    TranscriptionSegment,
)


class SpeechToTextProvider(Protocol):
    model_id: str
    provider_name: str

    @property
    def ready(self) -> bool: ...

    @property
    def load_error(self) -> str | None: ...

    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str,
        include_timestamps: bool,
    ) -> ProviderTranscript: ...


class STTService:
    def __init__(self, provider: SpeechToTextProvider, settings: Settings) -> None:
        self.provider = provider
        self.settings = settings

    @property
    def ready(self) -> bool:
        return self.provider.ready

    async def transcribe_audio(
        self,
        audio: bytes,
        *,
        language: str,
        encoding: str,
        include_timestamps: bool,
        request_id: str,
    ) -> TranscriptionResponse:
        if not audio:
            raise ServiceError(
                code="INVALID_REQUEST",
                message="The audio file is empty.",
                status_code=400,
            )
        if len(audio) > self.settings.stt_max_audio_bytes:
            raise ServiceError(
                code="PAYLOAD_TOO_LARGE",
                message="The audio file exceeds the configured size limit.",
                status_code=413,
                details={"maxBytes": self.settings.stt_max_audio_bytes},
            )
        if encoding not in {"wav", "webm", "flac"}:
            raise ServiceError(
                code="UNSUPPORTED_MEDIA_TYPE",
                message=f"Audio encoding '{encoding}' is not supported.",
                status_code=415,
            )
        if language not in self.settings.stt_allowed_languages:
            raise ServiceError(
                code="UNSUPPORTED_LANGUAGE",
                message=f"Speech recognition language '{language}' is not enabled.",
                status_code=422,
                details={"allowedLanguages": self.settings.stt_allowed_languages},
            )

        started = time.perf_counter()
        result = await self.provider.transcribe(
            audio,
            language=language,
            include_timestamps=include_timestamps,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        return TranscriptionResponse(
            request_id=request_id,
            text=result.text,
            language=result.language,
            segments=[
                TranscriptionSegment(
                    text=segment.text,
                    start_ms=round(segment.start_seconds * 1000),
                    end_ms=round(segment.end_seconds * 1000),
                )
                for segment in result.segments
            ],
            model=ModelDescriptor(
                provider=self.provider.provider_name,
                id=self.provider.model_id,
            ),
            latency_ms=latency_ms,
        )
