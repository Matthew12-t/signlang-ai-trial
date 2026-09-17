"""Local CTranslate2 runtime for Systran/faster-whisper-large-v3."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.models import ProviderTranscript, ProviderTranscriptSegment

logger = logging.getLogger(__name__)


class FasterWhisperProvider:
    provider_name = "local-ctranslate2"

    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.stt_model
        self._settings = settings
        self._model: Any | None = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.stt_max_concurrency)

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def load(self) -> None:
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                self._model = await asyncio.to_thread(self._load_sync)
                self._load_error = None
                logger.info("Loaded STT model %s", self.model_id)
            except Exception as error:
                self._load_error = str(error)
                logger.exception("Unable to load STT model %s", self.model_id)
                raise

    def _load_sync(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ServiceError(
                code="MODEL_DEPENDENCY_MISSING",
                message="The faster-whisper runtime is not installed.",
                status_code=503,
            ) from error

        kwargs: dict[str, Any] = {
            "device": self._settings.stt_device,
            "compute_type": self._settings.stt_compute_type,
        }
        if self._settings.model_cache_dir:
            kwargs["download_root"] = self._settings.model_cache_dir
        return WhisperModel(self.model_id, **kwargs)

    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str,
        include_timestamps: bool,
    ) -> ProviderTranscript:
        if self._model is None:
            try:
                await self.load()
            except ServiceError:
                raise
            except Exception as error:
                raise ServiceError(
                    code="MODEL_UNAVAILABLE",
                    message="The Speech-to-Text model could not be loaded.",
                    status_code=503,
                    retryable=True,
                ) from error

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._settings.model_queue_timeout_seconds,
            )
        except TimeoutError as error:
            raise ServiceError(
                code="MODEL_BUSY",
                message="The Speech-to-Text model is processing another request.",
                status_code=503,
                retryable=True,
            ) from error

        try:
            return await asyncio.to_thread(
                self._transcribe_sync, audio, language, include_timestamps
            )
        except ServiceError:
            raise
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise ServiceError(
                    code="MODEL_OUT_OF_MEMORY",
                    message="The Speech-to-Text model ran out of GPU memory.",
                    status_code=503,
                ) from error
            raise ServiceError(
                code="INFERENCE_FAILED",
                message="Speech-to-Text inference failed.",
                status_code=502,
                retryable=True,
            ) from error
        except Exception as error:
            raise ServiceError(
                code="UPSTREAM_BAD_RESPONSE",
                message="The Speech-to-Text model returned an unusable result.",
                status_code=502,
                retryable=True,
            ) from error
        finally:
            self._semaphore.release()

    def _transcribe_sync(
        self, audio: bytes, language: str, include_timestamps: bool
    ) -> ProviderTranscript:
        segments, info = self._model.transcribe(
            io.BytesIO(audio),
            language=language,
            task="transcribe",
            beam_size=self._settings.stt_beam_size,
            vad_filter=self._settings.stt_vad_filter,
            vad_parameters={
                "min_silence_duration_ms": self._settings.stt_min_silence_ms
            },
            condition_on_previous_text=(
                self._settings.stt_condition_on_previous_text
            ),
        )
        materialized = tuple(
            ProviderTranscriptSegment(
                text=segment.text.strip(),
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
            )
            for segment in segments
            if segment.text.strip()
        )
        text = " ".join(segment.text for segment in materialized).strip()
        if not include_timestamps:
            materialized = ()
        return ProviderTranscript(
            text=text,
            language=getattr(info, "language", None) or language,
            segments=materialized,
        )
