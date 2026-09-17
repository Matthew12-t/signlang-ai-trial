"""Text-to-Speech application service."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from typing import Protocol

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.models import AudioArtifact, GeneratedAudio
from src.tts.audio import concatenate_audio, encode_wav


class TextToSpeechProvider(Protocol):
    model_id: str

    @property
    def ready(self) -> bool: ...

    @property
    def load_error(self) -> str | None: ...

    async def synthesize(self, text: str) -> GeneratedAudio: ...


def split_text(text: str, max_chars: int) -> list[str]:
    """Split on sentence/word boundaries without dropping input text."""

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        words = sentence.split()
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word
        if len(current) > max_chars:
            chunks.extend(
                current[index : index + max_chars]
                for index in range(0, len(current), max_chars)
            )
            current = ""
    if current:
        chunks.append(current)
    return chunks


class TTSService:
    def __init__(
        self,
        provider: TextToSpeechProvider,
        settings: Settings,
        *,
        cache_size: int = 64,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[str, AudioArtifact]] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.provider.ready

    async def synthesize_speech(
        self,
        text: str,
        *,
        language: str,
        voice: str | None,
        output_format: str,
        idempotency_key: str,
    ) -> AudioArtifact:
        normalized_text = text.strip()
        if not normalized_text:
            raise ServiceError(
                code="INVALID_REQUEST",
                message="Text must not be empty.",
                status_code=400,
            )
        if len(normalized_text) > self.settings.tts_max_text_chars:
            raise ServiceError(
                code="PAYLOAD_TOO_LARGE",
                message="Text exceeds the configured character limit.",
                status_code=413,
                details={"maxCharacters": self.settings.tts_max_text_chars},
            )
        if language not in self.settings.tts_allowed_languages:
            raise ServiceError(
                code="UNSUPPORTED_LANGUAGE",
                message=f"Speech synthesis language '{language}' is not enabled.",
                status_code=422,
                details={"allowedLanguages": self.settings.tts_allowed_languages},
            )
        if voice not in {None, "default"}:
            raise ServiceError(
                code="INVALID_REQUEST",
                message="Only the default voice is available in the MVP.",
                status_code=400,
            )
        if output_format != "wav":
            raise ServiceError(
                code="UNSUPPORTED_MEDIA_TYPE",
                message="Only WAV output is available in the MVP.",
                status_code=415,
            )

        payload_hash = hashlib.sha256(
            f"{normalized_text}\0{language}\0{voice}\0{output_format}".encode()
        ).hexdigest()
        cached = await self._cached(idempotency_key, payload_hash)
        if cached is not None:
            return cached

        started = time.perf_counter()
        text_chunks = split_text(normalized_text, self.settings.tts_chunk_chars)
        generated = [
            await self.provider.synthesize(text_chunk)
            for text_chunk in text_chunks
        ]
        combined = concatenate_audio(generated, pause_ms=self.settings.tts_pause_ms)
        artifact = AudioArtifact(
            content=encode_wav(combined),
            media_type="audio/wav",
            sample_rate=combined.sample_rate,
            model_id=self.provider.model_id,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        await self._store(idempotency_key, payload_hash, artifact)
        return artifact

    async def _cached(
        self, idempotency_key: str, payload_hash: str
    ) -> AudioArtifact | None:
        async with self._cache_lock:
            cached = self._cache.get(idempotency_key)
            if cached is None:
                return None
            cached_hash, artifact = cached
            if cached_hash != payload_hash:
                raise ServiceError(
                    code="IDEMPOTENCY_CONFLICT",
                    message="The idempotency key was already used for another payload.",
                    status_code=409,
                )
            self._cache.move_to_end(idempotency_key)
            return artifact

    async def _store(
        self, idempotency_key: str, payload_hash: str, artifact: AudioArtifact
    ) -> None:
        async with self._cache_lock:
            self._cache[idempotency_key] = (payload_hash, artifact)
            self._cache.move_to_end(idempotency_key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
