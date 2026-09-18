import asyncio
from dataclasses import replace

import numpy as np
import pytest

from src.shared.config import Settings
from src.shared.models import GeneratedAudio, ProviderTranscript
from src.shared.providers.faster_whisper import FasterWhisperProvider
from src.shared.providers.voxcpm import VoxCPMProvider


@pytest.mark.anyio
async def test_faster_whisper_lazy_loads_once_for_concurrent_requests(
    monkeypatch,
) -> None:
    settings = replace(Settings.from_env(), stt_max_concurrency=2)
    provider = FasterWhisperProvider(settings)
    load_count = 0

    def load_sync():
        nonlocal load_count
        load_count += 1
        return object()

    def transcribe_sync(
        audio: bytes, language: str, include_timestamps: bool
    ) -> ProviderTranscript:
        return ProviderTranscript("hello", language, ())

    monkeypatch.setattr(provider, "_load_sync", load_sync)
    monkeypatch.setattr(provider, "_transcribe_sync", transcribe_sync)

    await asyncio.gather(
        provider.transcribe(b"a", language="en", include_timestamps=False),
        provider.transcribe(b"b", language="en", include_timestamps=False),
    )

    assert provider.ready is True
    assert load_count == 1


@pytest.mark.anyio
async def test_voxcpm_lazy_loads_once_for_concurrent_requests(monkeypatch) -> None:
    settings = replace(Settings.from_env(), tts_max_concurrency=2)
    provider = VoxCPMProvider(settings)
    load_count = 0

    def load_sync():
        nonlocal load_count
        load_count += 1
        return object()

    def synthesize_sync(text: str) -> GeneratedAudio:
        return GeneratedAudio(np.zeros(10, dtype=np.float32), 16_000)

    monkeypatch.setattr(provider, "_load_sync", load_sync)
    monkeypatch.setattr(provider, "_synthesize_sync", synthesize_sync)

    await asyncio.gather(provider.synthesize("a"), provider.synthesize("b"))

    assert provider.ready is True
    assert load_count == 1
