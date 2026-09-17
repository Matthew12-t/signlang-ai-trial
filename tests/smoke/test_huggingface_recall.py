"""Opt-in hosted Recall smoke test; excluded from the normal suite."""

import os

import pytest

from src.recall.retrieval import ProvidedContextRetriever
from src.recall.service import RecallService
from src.shared.config import Settings
from src.shared.models import RecallQueryRequest
from src.shared.providers.huggingface import HuggingFaceChatProvider


pytestmark = pytest.mark.smoke


@pytest.mark.asyncio
async def test_huggingface_answers_from_supplied_context() -> None:
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not (os.getenv("RUN_HF_RECALL_SMOKE") == "1" and hf_token):
        pytest.skip("requires RUN_HF_RECALL_SMOKE=1 and HF_TOKEN")

    settings = Settings.from_env()
    service = RecallService(
        provider=HuggingFaceChatProvider.from_settings(
            settings, model=settings.recall_model
        ),
        retriever=ProvidedContextRetriever(),
        model=settings.recall_model,
        max_context_chars=settings.recall_max_context_chars,
        max_answer_tokens=settings.recall_max_answer_tokens,
        temperature=settings.recall_temperature,
    )
    request = RecallQueryRequest.model_validate(
        {
            "sessionId": "smoke-session-1",
            "query": "When is the submission deadline?",
            "language": "en",
            "contextEntries": [
                {
                    "id": "tr_smoke_1",
                    "sequence": 1,
                    "source": "speech",
                    "speaker": "Participant",
                    "text": "The submission deadline is Friday at five PM.",
                    "startedAt": "2026-09-17T10:00:00Z",
                    "endedAt": "2026-09-17T10:00:04Z",
                }
            ],
        }
    )

    result = await service.query(request)

    assert result.grounded is True
    assert result.evidence
    assert {item.entry_id for item in result.evidence} == {"tr_smoke_1"}

