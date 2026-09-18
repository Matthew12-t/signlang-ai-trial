"""Hugging Face Inference Providers adapters for Speech-to-Text and Text-to-Speech.

These providers satisfy the same protocols as the local runtimes but call hosted
inference over HTTP, so no model weights are downloaded or executed here. Select
them with ``STT_PROVIDER=hf-api`` and ``TTS_PROVIDER=hf-api``.

Hosted inference serves different repositories than the local runtimes:
``Systran/faster-whisper-large-v3`` is a CTranslate2 conversion and VoxCPM2 is not
hosted at all, so these adapters read their own model settings rather than
``HF_STT_MODEL``/``HF_TTS_MODEL``.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.models import (
    GeneratedAudio,
    ProviderTranscript,
    ProviderTranscriptSegment,
)

logger = logging.getLogger(__name__)


def _status_code_of(error: BaseException) -> int | None:
    """Dig an HTTP status out of a hub error without depending on its class."""

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _translate_error(error: BaseException, *, label: str) -> ServiceError:
    status = _status_code_of(error)
    if status in {401, 403}:
        return ServiceError(
            code="MODEL_UNAVAILABLE",
            message=f"Hugging Face rejected the token for {label}.",
            status_code=503,
        )
    if status == 402:
        return ServiceError(
            code="INFERENCE_QUOTA_EXHAUSTED",
            message=(
                f"The Hugging Face inference credits for {label} are used up."
            ),
            status_code=503,
        )
    if status == 404:
        return ServiceError(
            code="MODEL_UNAVAILABLE",
            message=f"The {label} model is not served by Hugging Face.",
            status_code=503,
        )
    if status == 429:
        return ServiceError(
            code="MODEL_BUSY",
            message=f"Hugging Face rate limited the {label} request.",
            status_code=503,
            retryable=True,
        )
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return ServiceError(
            code="INFERENCE_TIMEOUT",
            message=f"Hugging Face did not answer the {label} request in time.",
            status_code=504,
            retryable=True,
        )
    return ServiceError(
        code="UPSTREAM_BAD_RESPONSE",
        message=f"Hugging Face returned an unusable {label} response.",
        status_code=502,
        retryable=True,
    )


class _HfApiProvider:
    """Shared client lifecycle, concurrency limit, and error normalization."""

    provider_name = "hf-inference-api"

    def __init__(
        self,
        settings: Settings,
        *,
        model_id: str,
        timeout_seconds: float,
        max_concurrency: int,
        label: str,
    ) -> None:
        self.model_id = model_id
        self._settings = settings
        self._label = label
        self._timeout_seconds = timeout_seconds
        self._client: Any | None = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def ready(self) -> bool:
        return self._client is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def load(self) -> None:
        async with self._load_lock:
            if self._client is not None:
                return
            try:
                self._client = self._build_client()
                self._load_error = None
                logger.info("Using hosted %s model %s", self._label, self.model_id)
            except Exception as error:
                self._load_error = str(error)
                logger.exception(
                    "Unable to prepare hosted %s model %s",
                    self._label,
                    self.model_id,
                )
                raise

    def _build_client(self) -> Any:
        try:
            from huggingface_hub import AsyncInferenceClient
        except ImportError as error:
            raise ServiceError(
                code="MODEL_DEPENDENCY_MISSING",
                message="The huggingface_hub runtime is not installed.",
                status_code=503,
            ) from error

        if self._settings.hf_token is None:
            raise ServiceError(
                code="MODEL_UNAVAILABLE",
                message="HF_TOKEN must be set to call hosted inference.",
                status_code=503,
            )

        return AsyncInferenceClient(
            provider=self._settings.hf_api_provider,
            token=self._settings.hf_token,
            timeout=self._timeout_seconds,
        )

    async def _call(
        self, operation: Callable[[Any], Coroutine[Any, Any, Any]]
    ) -> Any:
        if self._client is None:
            try:
                await self.load()
            except ServiceError:
                raise
            except Exception as error:
                raise ServiceError(
                    code="MODEL_UNAVAILABLE",
                    message=f"The hosted {self._label} client is unavailable.",
                    status_code=503,
                    retryable=True,
                ) from error

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._settings.model_queue_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise ServiceError(
                code="MODEL_BUSY",
                message=f"Too many in-flight hosted {self._label} requests.",
                status_code=503,
                retryable=True,
            ) from error

        try:
            return await operation(self._client)
        except ServiceError:
            raise
        except Exception as error:
            raise _translate_error(error, label=self._label) from error
        finally:
            self._semaphore.release()


class HfApiSTTProvider(_HfApiProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            model_id=settings.hf_api_stt_model,
            timeout_seconds=settings.stt_timeout_seconds,
            max_concurrency=settings.stt_max_concurrency,
            label="Speech-to-Text",
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str,
        include_timestamps: bool,
    ) -> ProviderTranscript:
        extra_body = {"return_timestamps": True} if include_timestamps else None
        result = await self._call(
            lambda client: client.automatic_speech_recognition(
                audio, model=self.model_id, extra_body=extra_body
            )
        )
        return _to_transcript(result, language, include_timestamps)


class HfApiTTSProvider(_HfApiProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            model_id=settings.hf_api_tts_model,
            timeout_seconds=settings.tts_timeout_seconds,
            max_concurrency=settings.tts_max_concurrency,
            label="Text-to-Speech",
        )

    async def synthesize(self, text: str) -> GeneratedAudio:
        encoded = await self._call(
            lambda client: client.text_to_speech(text, model=self.model_id)
        )
        return await asyncio.to_thread(_decode_audio, encoded)


def _to_transcript(
    result: Any, language: str, include_timestamps: bool
) -> ProviderTranscript:
    """Normalize an AutomaticSpeechRecognitionOutput into the shared shape.

    Hosted Whisper reports no detected language, so the requested one is echoed
    back. Timestamps arrive only when the selected provider honours
    ``return_timestamps``; an empty tuple is still a valid transcript.
    """

    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise ServiceError(
            code="UPSTREAM_BAD_RESPONSE",
            message="Hugging Face returned an empty transcript.",
            status_code=502,
            retryable=True,
        )

    segments: list[ProviderTranscriptSegment] = []
    if include_timestamps:
        for chunk in getattr(result, "chunks", None) or ():
            chunk_text = (getattr(chunk, "text", None) or "").strip()
            stamp = getattr(chunk, "timestamp", None) or ()
            if not chunk_text or len(stamp) != 2:
                continue
            start, end = stamp
            if start is None or end is None:
                continue
            segments.append(
                ProviderTranscriptSegment(
                    text=chunk_text,
                    start_seconds=float(start),
                    end_seconds=float(end),
                )
            )

    return ProviderTranscript(text=text, language=language, segments=tuple(segments))


def _decode_audio(encoded: bytes) -> GeneratedAudio:
    """Decode hosted audio bytes into the float32 mono samples the pipeline wants.

    Hosted text-to-speech answers with an encoded file (FLAC, WAV, or MP3
    depending on the provider) while ``concatenate_audio`` and ``encode_wav``
    operate on raw samples.
    """

    try:
        import numpy as np
        import soundfile
    except ImportError as error:
        raise ServiceError(
            code="MODEL_DEPENDENCY_MISSING",
            message="The soundfile runtime is not installed.",
            status_code=503,
        ) from error

    if not encoded:
        raise ServiceError(
            code="UPSTREAM_BAD_RESPONSE",
            message="Hugging Face returned empty audio.",
            status_code=502,
            retryable=True,
        )

    try:
        samples, sample_rate = soundfile.read(
            io.BytesIO(encoded), dtype="float32", always_2d=True
        )
    except Exception as error:
        raise ServiceError(
            code="UPSTREAM_BAD_RESPONSE",
            message="The hosted audio could not be decoded.",
            status_code=502,
            retryable=True,
        ) from error

    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    if mono.size == 0:
        raise ServiceError(
            code="UPSTREAM_BAD_RESPONSE",
            message="Hugging Face returned empty audio.",
            status_code=502,
            retryable=True,
        )
    return GeneratedAudio(
        samples=np.ascontiguousarray(mono, dtype=np.float32),
        sample_rate=int(sample_rate),
    )
