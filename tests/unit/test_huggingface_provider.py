import httpx
import pytest

from src.shared.config import Settings
from src.shared.models import ChatMessage
from src.shared.providers import (
    ProviderBadResponse,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from src.shared.providers.huggingface import HuggingFaceChatProvider


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeClient:
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="Return JSON."),
        ChatMessage(role="user", content="Hello"),
    ]


@pytest.mark.asyncio
async def test_complete_json_parses_object_and_forwards_request_values() -> None:
    client = FakeClient(FakeResponse('{"text": "Hello"}'))
    provider = HuggingFaceChatProvider(client=client, model="test-model")

    result = await provider.complete_json(
        messages(), {"type": "object"}, max_tokens=42, temperature=0.2
    )

    assert result == {"text": "Hello"}
    assert client.calls == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 42,
            "temperature": 0.2,
        }
    ]


@pytest.mark.asyncio
async def test_complete_json_maps_malformed_json_to_bad_response() -> None:
    provider = HuggingFaceChatProvider(
        client=FakeClient(FakeResponse("not JSON")), model="test-model"
    )

    with pytest.raises(ProviderBadResponse) as raised:
        await provider.complete_json(messages(), {}, max_tokens=42, temperature=0.2)

    assert_sanitized_error(raised.value, "Provider returned invalid JSON")


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["[]", '"text"', "42", "null"])
async def test_complete_json_rejects_non_object_json(content: str) -> None:
    provider = HuggingFaceChatProvider(
        client=FakeClient(FakeResponse(content)), model="test-model"
    )

    with pytest.raises(ProviderBadResponse):
        await provider.complete_json(messages(), {}, max_tokens=42, temperature=0.2)


class ResponseError(Exception):
    def __init__(self, status_code: int, message: str = "provider secret") -> None:
        self.response = type("Response", (), {"status_code": status_code})()
        super().__init__(message)


def assert_sanitized_error(error: Exception, message: str) -> None:
    assert str(error) == message
    assert error.__suppress_context__ is True
    assert error.__cause__ is None


@pytest.mark.asyncio
async def test_complete_json_maps_429_to_rate_limited() -> None:
    provider = HuggingFaceChatProvider(
        client=FakeClient(ResponseError(429)), model="test-model"
    )

    with pytest.raises(ProviderRateLimited) as raised:
        await provider.complete_json(messages(), {}, max_tokens=42, temperature=0.2)

    assert_sanitized_error(raised.value, "Provider rate limit exceeded")


@pytest.mark.asyncio
async def test_complete_json_maps_5xx_to_unavailable() -> None:
    provider = HuggingFaceChatProvider(
        client=FakeClient(ResponseError(503)), model="test-model"
    )

    with pytest.raises(ProviderUnavailable) as raised:
        await provider.complete_json(messages(), {}, max_tokens=42, temperature=0.2)

    assert_sanitized_error(raised.value, "Provider is unavailable")


@pytest.mark.asyncio
async def test_complete_json_maps_timeout_to_provider_timeout() -> None:
    provider = HuggingFaceChatProvider(
        client=FakeClient(TimeoutError("provider secret")), model="test-model"
    )

    with pytest.raises(ProviderTimeout) as raised:
        await provider.complete_json(messages(), {}, max_tokens=42, temperature=0.2)

    assert_sanitized_error(raised.value, "Provider request timed out")


@pytest.mark.asyncio
async def test_complete_json_maps_httpx_timeout_to_provider_timeout() -> None:
    provider = HuggingFaceChatProvider(
        client=FakeClient(httpx.ReadTimeout("https://token:secret@example.invalid")),
        model="test-model",
    )

    with pytest.raises(ProviderTimeout) as raised:
        await provider.complete_json(messages(), {}, max_tokens=42, temperature=0.2)

    assert_sanitized_error(raised.value, "Provider request timed out")


@pytest.mark.asyncio
async def test_complete_json_hides_unexpected_provider_error_text() -> None:
    provider = HuggingFaceChatProvider(
        client=FakeClient(RuntimeError("https://token:secret@example.invalid")),
        model="test-model",
    )

    with pytest.raises(ProviderBadResponse) as raised:
        await provider.complete_json(messages(), {}, max_tokens=42, temperature=0.2)

    assert_sanitized_error(raised.value, "Provider returned an invalid response")


@pytest.mark.asyncio
async def test_complete_json_omits_response_format_without_structured_output() -> None:
    client = FakeClient(FakeResponse("{}"))
    provider = HuggingFaceChatProvider(client=client, model="test-model")

    await provider.complete_json(messages(), {"type": "object"}, max_tokens=42, temperature=0.2)

    assert "response_format" not in client.calls[0]


@pytest.mark.asyncio
async def test_complete_json_uses_strict_schema_when_structured_output_is_enabled() -> None:
    client = FakeClient(FakeResponse("{}"))
    provider = HuggingFaceChatProvider(
        client=client, model="test-model", use_structured_output=True
    )

    await provider.complete_json(
        messages(), {"type": "object", "properties": {"text": {"type": "string"}}},
        max_tokens=42,
        temperature=0.2,
    )

    assert client.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "isyara_response",
            "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            "strict": True,
        },
    }


def test_from_settings_rejects_missing_token_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**kwargs: object) -> object:
        raise AssertionError("InferenceClient must not be constructed")

    monkeypatch.setattr("src.shared.providers.huggingface.InferenceClient", fail_if_called)

    with pytest.raises(ProviderUnavailable, match="HF token is not configured"):
        HuggingFaceChatProvider.from_settings(Settings(hf_token=None))


@pytest.mark.asyncio
async def test_from_settings_builds_configured_client_and_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(FakeResponse("{}"))
    captured: dict[str, object] = {}

    def build_client(**kwargs: object) -> FakeClient:
        captured.update(kwargs)
        return client

    monkeypatch.setattr("src.shared.providers.huggingface.InferenceClient", build_client)
    provider = HuggingFaceChatProvider.from_settings(
        Settings(
            hf_token="test-token",
            hf_provider="together",
            request_timeout_seconds=12.5,
            llm_model="configured-model",
            hf_use_structured_output=True,
        )
    )

    await provider.complete_json(messages(), {"type": "object"}, max_tokens=42, temperature=0.2)

    assert captured == {
        "provider": "together",
        "api_key": "test-token",
        "timeout": 12.5,
    }
    assert client.calls[0]["model"] == "configured-model"
    assert client.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "isyara_response",
            "schema": {"type": "object"},
            "strict": True,
        },
    }
