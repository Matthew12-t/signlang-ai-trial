"""Opt-in tests that download and execute the real model weights."""

from __future__ import annotations

import asyncio
import io
import os
import wave

import numpy as np
import pytest

from src.shared.config import Settings
from src.shared.providers.faster_whisper import FasterWhisperProvider
from src.shared.providers.voxcpm import VoxCPMProvider

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_SMOKE_TESTS") != "1",
        reason="Set RUN_MODEL_SMOKE_TESTS=1 to load the real AI models.",
    ),
]


def _silent_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
    return buffer.getvalue()


def test_faster_whisper_model_loads_and_transcribes() -> None:
    async def exercise() -> None:
        provider = FasterWhisperProvider(Settings.from_env())
        await provider.load()
        result = await provider.transcribe(
            _silent_wav(), language="en", include_timestamps=True
        )
        assert result.language == "en"

    asyncio.run(exercise())


def test_voxcpm_model_loads_and_synthesizes() -> None:
    async def exercise() -> None:
        provider = VoxCPMProvider(Settings.from_env())
        await provider.load()
        result = await provider.synthesize("Hello from Isyara.")
        assert result.sample_rate > 0
        assert len(result.samples) > 0

    asyncio.run(exercise())
