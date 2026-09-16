"""Opt-in hosted-provider smoke test; it never runs in the normal suite."""

import os

import pytest

from src.gloss.service import GlossService
from src.shared.config import Settings
from src.shared.models import GlossNormalizeRequest
from src.shared.providers.huggingface import HuggingFaceChatProvider


pytestmark = pytest.mark.smoke


@pytest.mark.asyncio
async def test_huggingface_normalizes_confirmed_thank_you_token() -> None:
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not (os.getenv("RUN_HF_SMOKE") == "1" and hf_token):
        pytest.skip("requires RUN_HF_SMOKE=1 and HF_TOKEN")

    settings = Settings(gloss_mode="qwen")
    provider = HuggingFaceChatProvider.from_settings(settings)
    service = GlossService(settings, provider)
    request = GlossNormalizeRequest(
        utteranceId="smoke-utterance-1",
        language="en",
        tokens=[
            {
                "id": "tok_smoke_1",
                "label": "THANK_YOU",
                "confirmedAt": "2026-09-17T10:00:00Z",
            }
        ],
    )

    result = await service.normalize(request)

    assert result.text
    assert result.source_token_ids == ["tok_smoke_1"]
    assert result.method == "qwen"
    assert result.warnings == []
