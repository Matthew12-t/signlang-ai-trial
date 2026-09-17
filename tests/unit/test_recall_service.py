import json
from datetime import UTC, datetime, timedelta

import pytest

from src.recall.prompt import build_recall_messages
from src.recall.retrieval import ProvidedContextRetriever
from src.recall.service import RecallService
from src.shared.errors import AppError
from src.shared.models import RecallQueryRequest, TranscriptEntry
from src.shared.providers import (
    ProviderBadResponse,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


class FakeProvider:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def complete_json(self, messages, response_schema, **options):
        self.calls.append(
            {"messages": messages, "response_schema": response_schema, **options}
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def transcript(
    entry_id: str = "tr_17",
    sequence: int = 17,
    text: str = "The submission deadline is Friday at five PM.",
) -> TranscriptEntry:
    started = datetime(2026, 9, 18, 10, 18, 11, tzinfo=UTC) + timedelta(seconds=sequence)
    return TranscriptEntry(
        id=entry_id,
        sequence=sequence,
        source="speech",
        speaker="Participant",
        text=text,
        startedAt=started,
        endedAt=started + timedelta(seconds=3),
    )


def recall_request(entries: list[TranscriptEntry] | None = None) -> RecallQueryRequest:
    return RecallQueryRequest(
        sessionId="ses_1",
        query="When is the deadline?",
        language="en",
        contextEntries=[transcript()] if entries is None else entries,
        options={"maxAnswerTokens": 160, "requireEvidence": True},
    )


def service(provider: FakeProvider) -> RecallService:
    return RecallService(
        provider=provider,
        retriever=ProvidedContextRetriever(),
        model="Qwen/Qwen3-8B",
        max_context_chars=24_000,
        max_answer_tokens=160,
        temperature=0.0,
    )


def test_prompt_contains_only_structured_request_context_and_schema() -> None:
    request = recall_request()
    messages = build_recall_messages(request, request.context_entries)

    assert "only from the supplied transcript" in messages[0].content
    payload = json.loads(messages[1].content)
    assert payload == {
        "question": "When is the deadline?",
        "language": "en",
        "options": {"maxAnswerTokens": 160, "requireEvidence": True},
        "transcriptEntries": [request.context_entries[0].model_dump(by_alias=True, mode="json")],
    }
    assert "response JSON schema" in messages[0].content


@pytest.mark.asyncio
async def test_empty_context_returns_not_found_without_calling_provider() -> None:
    provider = FakeProvider(AssertionError("provider must not be called"))

    result = await service(provider).query(recall_request([]))

    assert result.grounded is False
    assert result.not_found_reason == "EMPTY_CONTEXT"
    assert result.evidence == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_valid_grounded_answer_returns_locally_derived_evidence_timestamp() -> None:
    provider = FakeProvider(
        {
            "answer": "The deadline is Friday at 5 PM.",
            "grounded": True,
            "evidence": [
                {
                    "entryId": "tr_17",
                    "quote": "submission deadline is Friday at five PM",
                }
            ],
            "notFoundReason": None,
        }
    )

    result = await service(provider).query(recall_request())

    assert result.grounded is True
    assert result.not_found_reason is None
    assert result.evidence[0].entry_id == "tr_17"
    assert result.evidence[0].started_at == transcript().started_at
    assert provider.calls[0]["temperature"] == 0.0
    assert provider.calls[0]["max_tokens"] == 160


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {
            "answer": "Friday.",
            "grounded": True,
            "evidence": [{"entryId": "missing", "quote": "Friday"}],
            "notFoundReason": None,
        },
        {
            "answer": "Saturday.",
            "grounded": True,
            "evidence": [{"entryId": "tr_17", "quote": "Saturday"}],
            "notFoundReason": None,
        },
        {
            "answer": "Friday.",
            "grounded": True,
            "evidence": [],
            "notFoundReason": None,
        },
    ],
)
async def test_invalid_evidence_is_replaced_with_safe_not_found(response: dict) -> None:
    result = await service(FakeProvider(response)).query(recall_request())

    assert result.grounded is False
    assert result.evidence == []
    assert result.not_found_reason == "EVIDENCE_VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_unsupported_answer_is_returned_as_not_in_context() -> None:
    provider = FakeProvider(
        {
            "answer": "I could not find that information in the transcript.",
            "grounded": False,
            "evidence": [],
            "notFoundReason": "NOT_IN_CONTEXT",
        }
    )

    result = await service(provider).query(recall_request())

    assert result.grounded is False
    assert result.not_found_reason == "NOT_IN_CONTEXT"


@pytest.mark.asyncio
async def test_malformed_provider_output_maps_to_bad_gateway() -> None:
    with pytest.raises(AppError) as raised:
        await service(FakeProvider({"unexpected": True})).query(recall_request())

    assert (raised.value.status_code, raised.value.code) == (502, "UPSTREAM_BAD_RESPONSE")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (ProviderRateLimited(), 429, "RATE_LIMITED"),
        (ProviderBadResponse(), 502, "UPSTREAM_BAD_RESPONSE"),
        (ProviderUnavailable(), 503, "MODEL_UNAVAILABLE"),
        (ProviderTimeout(), 504, "HF_TIMEOUT"),
    ],
)
async def test_provider_errors_use_stable_http_mapping(error, status: int, code: str) -> None:
    with pytest.raises(AppError) as raised:
        await service(FakeProvider(error)).query(recall_request())

    assert (raised.value.status_code, raised.value.code) == (status, code)
