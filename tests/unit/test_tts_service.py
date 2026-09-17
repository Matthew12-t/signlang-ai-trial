import asyncio
from dataclasses import replace

import numpy as np
import pytest

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.models import GeneratedAudio
from src.tts.service import TTSService, split_text


class FakeTTSProvider:
    model_id = "openbmb/VoxCPM2"
    ready = True
    load_error = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> GeneratedAudio:
        self.calls.append(text)
        return GeneratedAudio(
            samples=np.zeros(160, dtype=np.float32),
            sample_rate=16_000,
        )


def test_split_text_preserves_all_words() -> None:
    text = "This is the first sentence. This is the second sentence."
    chunks = split_text(text, 30)
    assert " ".join(chunks) == text
    assert all(len(chunk) <= 30 for chunk in chunks)


@pytest.mark.anyio
async def test_tts_idempotency_returns_cached_audio() -> None:
    settings = replace(
        Settings.from_env(),
        tts_chunk_chars=500,
        tts_allowed_languages=("en",),
    )
    provider = FakeTTSProvider()
    service = TTSService(provider, settings)

    first = await service.synthesize_speech(
        "Hello", language="en", voice=None, output_format="wav", idempotency_key="key-1"
    )
    second = await service.synthesize_speech(
        "Hello", language="en", voice=None, output_format="wav", idempotency_key="key-1"
    )

    assert first.content == second.content
    assert first.content.startswith(b"RIFF")
    assert provider.calls == ["Hello"]


@pytest.mark.anyio
async def test_tts_rejects_reused_key_with_different_payload() -> None:
    settings = replace(Settings.from_env(), tts_allowed_languages=("en",))
    service = TTSService(FakeTTSProvider(), settings)
    await service.synthesize_speech(
        "Hello", language="en", voice=None, output_format="wav", idempotency_key="same"
    )

    with pytest.raises(ServiceError) as caught:
        await service.synthesize_speech(
            "Different", language="en", voice=None, output_format="wav", idempotency_key="same"
        )
    assert caught.value.code == "IDEMPOTENCY_CONFLICT"


@pytest.mark.anyio
async def test_tts_coalesces_concurrent_requests_with_the_same_key() -> None:
    settings = replace(
        Settings.from_env(),
        tts_allowed_languages=("en",),
        tts_chunk_chars=500,
    )
    provider = FakeTTSProvider()
    original_synthesize = provider.synthesize

    async def delayed_synthesize(text: str) -> GeneratedAudio:
        await asyncio.sleep(0.01)
        return await original_synthesize(text)

    provider.synthesize = delayed_synthesize
    service = TTSService(provider, settings)

    first, second = await asyncio.gather(
        service.synthesize_speech(
            "Hello",
            language="en",
            voice=None,
            output_format="wav",
            idempotency_key="shared-key",
        ),
        service.synthesize_speech(
            "Hello",
            language="en",
            voice=None,
            output_format="wav",
            idempotency_key="shared-key",
        ),
    )

    assert first.content == second.content
    assert provider.calls == ["Hello"]


@pytest.mark.anyio
async def test_tts_inference_timeout_is_normalized() -> None:
    settings = replace(
        Settings.from_env(),
        tts_allowed_languages=("en",),
        tts_timeout_seconds=0.001,
    )
    provider = FakeTTSProvider()

    async def slow_synthesize(text: str) -> GeneratedAudio:
        await asyncio.sleep(0.03)
        return GeneratedAudio(np.zeros(10, dtype=np.float32), 16_000)

    provider.synthesize = slow_synthesize
    service = TTSService(provider, settings)

    with pytest.raises(ServiceError) as caught:
        await service.synthesize_speech(
            "Hello",
            language="en",
            voice=None,
            output_format="wav",
            idempotency_key="timeout-key",
        )

    assert caught.value.code == "INFERENCE_TIMEOUT"
    assert caught.value.status_code == 504
    assert caught.value.retryable is True
    await asyncio.sleep(0.04)
