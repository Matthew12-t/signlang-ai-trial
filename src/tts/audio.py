"""Audio assembly and WAV encoding helpers."""

from __future__ import annotations

import io
import wave
from collections.abc import Sequence

import numpy as np

from src.shared.errors import ServiceError
from src.shared.models import GeneratedAudio


def concatenate_audio(
    chunks: Sequence[GeneratedAudio], *, pause_ms: int
) -> GeneratedAudio:
    if not chunks:
        raise ServiceError(
            code="UPSTREAM_BAD_RESPONSE",
            message="The Text-to-Speech model did not produce audio.",
            status_code=502,
            retryable=True,
        )

    sample_rate = chunks[0].sample_rate
    if any(chunk.sample_rate != sample_rate for chunk in chunks):
        raise ServiceError(
            code="AUDIO_ENCODING_FAILED",
            message="Generated audio chunks use different sample rates.",
            status_code=502,
        )

    pause = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        if index:
            parts.append(pause)
        parts.append(np.asarray(chunk.samples, dtype=np.float32).reshape(-1))
    return GeneratedAudio(samples=np.concatenate(parts), sample_rate=sample_rate)


def encode_wav(audio: GeneratedAudio) -> bytes:
    samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    buffer = io.BytesIO()
    try:
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(audio.sample_rate)
            output.writeframes(pcm.tobytes())
    except (OSError, wave.Error) as error:
        raise ServiceError(
            code="AUDIO_ENCODING_FAILED",
            message="The generated waveform could not be encoded as WAV.",
            status_code=502,
        ) from error
    return buffer.getvalue()
