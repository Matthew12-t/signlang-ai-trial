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
