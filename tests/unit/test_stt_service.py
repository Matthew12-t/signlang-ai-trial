import asyncio
from dataclasses import replace

import pytest

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.models import ProviderTranscript
from src.stt.service import STTService


class SlowSTTProvider:
    model_id = "test-stt"
    provider_name = "test"
    ready = True
    load_error = None

    async def transcribe(
        self, audio: bytes, *, language: str, include_timestamps: bool
    ) -> ProviderTranscript:
        await asyncio.sleep(0.03)
        return ProviderTranscript("hello", language, ())


@pytest.mark.anyio
async def test_stt_inference_timeout_is_normalized() -> None:
    settings = replace(
        Settings.from_env(),
        stt_allowed_languages=("en",),
        stt_timeout_seconds=0.001,
    )
    service = STTService(SlowSTTProvider(), settings)

    with pytest.raises(ServiceError) as caught:
        await service.transcribe_audio(
            b"audio",
            language="en",
            encoding="wav",
            include_timestamps=True,
            request_id="request-id",
        )

    assert caught.value.code == "INFERENCE_TIMEOUT"
    assert caught.value.status_code == 504
    assert caught.value.retryable is True
    await asyncio.sleep(0.04)
