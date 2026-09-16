import json

import pytest

from src.gloss.service import GlossService, GlossValidationError
from src.shared.config import Settings
from src.shared.models import GlossNormalizeRequest, LLMGlossResult
from src.shared.providers import (
    ProviderBadResponse,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


class FakeAsyncProvider:
    def __init__(self, result: dict[str, object] | Exception) -> None:
        self.result = result
        self.calls = 0
        self.request: dict[str, object] | None = None

    async def complete_json(
        self,
        messages: list[object],
        response_schema: dict[str, object],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        self.calls += 1
        self.request = {
            "messages": messages,
            "response_schema": response_schema,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def request(*labels: str) -> GlossNormalizeRequest:
    return GlossNormalizeRequest(
        utteranceId="utterance-1",
        language="en",
        tokens=[
            {"id": f"tok_{index}", "label": label, "confirmedAt": "2026-09-17T10:00:00Z"}
            for index, label in enumerate(labels, start=1)
        ],
    )


@pytest.mark.asyncio
async def test_template_mode_uses_rule_result_without_calling_provider() -> None:
    provider = FakeAsyncProvider({"text": "wrong", "sourceTokenIds": ["tok_1"]})

    result = await GlossService(Settings(gloss_mode="template"), provider).normalize(request("THANK_YOU"))

    assert result.text == "Thank you."
    assert result.method == "template"
    assert result.source_token_ids == ["tok_1"]
    assert result.warnings == []
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_qwen_mode_returns_validated_provider_result() -> None:
    provider = FakeAsyncProvider({"text": "Thank you.", "sourceTokenIds": ["tok_1"]})

    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request("THANK_YOU"))

    assert result.text == "Thank you."
    assert result.method == "qwen"
    assert result.source_token_ids == ["tok_1"]
    assert result.warnings == []


@pytest.mark.asyncio
async def test_timeout_falls_back_to_template_with_stable_warning() -> None:
    result = await GlossService(
        Settings(gloss_mode="qwen"), FakeAsyncProvider(ProviderTimeout())
    ).normalize(request("THANK_YOU"))

    assert result.method == "template"
    assert result.warnings == ["LLM_FALLBACK_PROVIDER_TIMEOUT"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_ids",
    [
        [],
        ["tok_2"],
        ["tok_1", "tok_1"],
        ["tok_2", "tok_1"],
        ["tok_1", "tok_2", "tok_3"],
    ],
)
async def test_invalid_source_token_provenance_falls_back(source_ids: list[str]) -> None:
    provider = FakeAsyncProvider({"text": "Please repeat that.", "sourceTokenIds": source_ids})

    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(
        request("REPEAT", "PLEASE")
    )

    assert result.method == "template"
    assert result.text == "Please repeat that."
    assert result.source_token_ids == ["tok_1", "tok_2"]
    assert result.warnings == ["LLM_FALLBACK_INVALID_PROVENANCE"]


@pytest.mark.asyncio
async def test_qwen_mode_without_provider_falls_back() -> None:
    result = await GlossService(Settings(gloss_mode="qwen")).normalize(request("THANK_YOU"))

    assert result.method == "template"
    assert result.warnings == ["LLM_FALLBACK_PROVIDER_UNAVAILABLE"]


@pytest.mark.asyncio
async def test_pydantic_invalid_provider_output_falls_back() -> None:
    provider = FakeAsyncProvider({"text": "", "sourceTokenIds": ["tok_1"]})

    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request("THANK_YOU"))

    assert result.method == "template"
    assert result.warnings == ["LLM_FALLBACK_INVALID_RESPONSE"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Terima kasih.",
        "Café.",
        "Thank\u200b you.",
        "Thank\nyou.",
        "12345.",
        "<script>Thank you.</script>",
    ],
)
async def test_unsafe_or_non_english_provider_text_falls_back(unsafe_text: str) -> None:
    provider = FakeAsyncProvider({"text": unsafe_text, "sourceTokenIds": ["tok_1"]})

    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(
        request("THANK_YOU")
    )

    assert result.text == "Thank you."
    assert result.method == "template"
    assert result.warnings == ["LLM_FALLBACK_INVALID_RESPONSE"]


@pytest.mark.asyncio
async def test_rate_limit_falls_back_with_stable_warning() -> None:
    result = await GlossService(
        Settings(gloss_mode="qwen"), FakeAsyncProvider(ProviderRateLimited())
    ).normalize(request("THANK_YOU"))

    assert result.warnings == ["LLM_FALLBACK_RATE_LIMITED"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ProviderUnavailable(), ProviderBadResponse(), ProviderError()])
async def test_unavailable_provider_errors_fall_back(error: ProviderError) -> None:
    result = await GlossService(Settings(gloss_mode="qwen"), FakeAsyncProvider(error)).normalize(
        request("THANK_YOU")
    )

    assert result.warnings == ["LLM_FALLBACK_PROVIDER_UNAVAILABLE"]


@pytest.mark.asyncio
async def test_token_limit_is_rejected_before_normalization() -> None:
    service = GlossService(Settings(gloss_max_tokens=1))

    with pytest.raises(GlossValidationError, match="PAYLOAD_TOO_LARGE") as raised:
        await service.normalize(request("HELLO", "PLEASE"))

    assert raised.value.code == "PAYLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_non_english_input_is_defensively_rejected() -> None:
    unsafe_request = GlossNormalizeRequest.model_construct(
        utterance_id="utterance-1", language="id", tokens=request("HELLO").tokens
    )

    with pytest.raises(GlossValidationError, match="UNSUPPORTED_LANGUAGE") as raised:
        await GlossService(Settings()).normalize(unsafe_request)

    assert raised.value.code == "UNSUPPORTED_LANGUAGE"


@pytest.mark.asyncio
async def test_provider_receives_lossless_labels_without_confirmation_times() -> None:
    provider = FakeAsyncProvider({"text": "C++ café.", "sourceTokenIds": ["tok_1", "tok_2"]})

    await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request(" C++ ", "Café"))

    assert provider.request is not None
    assert json.loads(provider.request["messages"][1].content) == {
        "tokens": [
            {"id": "tok_1", "label": "C++"},
            {"id": "tok_2", "label": "Café"},
        ]
    }
    assert "confirmedAt" not in provider.request["messages"][1].content


@pytest.mark.asyncio
async def test_provider_uses_deterministic_settings_and_llm_schema() -> None:
    provider = FakeAsyncProvider({"text": "Thank you.", "sourceTokenIds": ["tok_1"]})
    settings = Settings(gloss_mode="qwen", gloss_llm_max_output_tokens=73)

    await GlossService(settings, provider).normalize(request("THANK_YOU"))

    assert provider.request is not None
    assert provider.request["temperature"] == 0
    assert provider.request["max_tokens"] == 73
    assert provider.request["response_schema"] == LLMGlossResult.model_json_schema(by_alias=True)


@pytest.mark.asyncio
async def test_default_provider_prompt_contains_exact_schema_and_valid_object_example() -> None:
    provider = FakeAsyncProvider({"text": "Thank you.", "sourceTokenIds": ["tok_1"]})
    expected_schema = LLMGlossResult.model_json_schema(by_alias=True)

    await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request("THANK_YOU"))

    assert provider.request is not None
    system_message = provider.request["messages"][0].content
    assert f"Response JSON Schema: {json.dumps(expected_schema, separators=(',', ':'))}" in system_message
    assert 'Valid response example: {"text":"Thank you.","sourceTokenIds":["tok_1"]}' in system_message
