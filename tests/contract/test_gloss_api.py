from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.shared.config import Settings


client = TestClient(create_app(Settings(gloss_mode="template")))


def payload(language: str = "en") -> dict[str, object]:
    return {
        "utteranceId": "utt_1",
        "language": language,
        "tokens": [
            {"id": "tok_1", "label": "I", "confirmedAt": "2026-09-17T10:00:00Z"},
            {
                "id": "tok_2",
                "label": "NOT_UNDERSTAND",
                "confirmedAt": "2026-09-17T10:00:01Z",
            },
        ],
    }


def assert_request_id(response: object) -> str:
    request_id = response.headers["X-Request-ID"]
    assert response.json()["requestId"] == request_id
    return request_id


def assert_error(
    response: object,
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
) -> None:
    assert response.status_code == status_code
    response_request_id = response.headers["X-Request-ID"]
    if request_id is not None:
        assert response_request_id == request_id
    assert response.json() == {
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "requestId": response_request_id,
            "details": {},
        }
    }


def test_normalize_returns_template_result_and_request_id() -> None:
    response = client.post("/v1/gloss/normalize", json=payload())

    assert response.status_code == 200
    body = response.json()
    assert body["utteranceId"] == "utt_1"
    assert body["text"] == "I do not understand."
    assert body["method"] == "template"
    assert body["sourceTokenIds"] == ["tok_1", "tok_2"]
    assert body["warnings"] == []
    assert isinstance(body["latencyMs"], int)
    assert body["latencyMs"] >= 0
    UUID(assert_request_id(response))


def test_supplied_request_id_is_echoed() -> None:
    response = client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Request-ID": "req-test-1"},
    )

    assert response.status_code == 200
    assert assert_request_id(response) == "req-test-1"


def test_missing_request_id_generates_uuid() -> None:
    response = client.post("/v1/gloss/normalize", json=payload())

    UUID(assert_request_id(response))


@pytest.mark.parametrize("request_id", ["", "   ", "x" * 129])
def test_invalid_request_id_is_replaced(request_id: str) -> None:
    response = client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Request-ID": request_id},
    )

    generated = assert_request_id(response)
    UUID(generated)
    assert generated != request_id


def test_non_english_language_uses_shared_error_envelope() -> None:
    response = client.post(
        "/v1/gloss/normalize",
        json=payload("id"),
        headers={"X-Request-ID": "req-language"},
    )

    assert_error(
        response,
        status_code=422,
        code="UNSUPPORTED_LANGUAGE",
        message="Only English is supported.",
        request_id="req-language",
    )


def test_empty_tokens_use_sanitized_invalid_request_error() -> None:
    request_payload = payload()
    request_payload["tokens"] = []
    request_payload["private"] = "must-not-be-echoed"

    response = client.post(
        "/v1/gloss/normalize",
        json=request_payload,
        headers={"X-Request-ID": "req-invalid"},
    )

    assert_error(
        response,
        status_code=422,
        code="INVALID_REQUEST",
        message="The request is invalid.",
        request_id="req-invalid",
    )
    assert "must-not-be-echoed" not in response.text


def test_more_than_64_tokens_is_payload_too_large() -> None:
    request_payload = payload()
    request_payload["tokens"] = [
        {
            "id": f"tok_{index}",
            "label": "HELLO",
            "confirmedAt": "2026-09-17T10:00:00Z",
        }
        for index in range(65)
    ]

    response = client.post(
        "/v1/gloss/normalize",
        json=request_payload,
        headers={"X-Request-ID": "req-large"},
    )

    assert_error(
        response,
        status_code=413,
        code="PAYLOAD_TOO_LARGE",
        message="The request contains too many tokens.",
        request_id="req-large",
    )


def test_route_is_open_when_internal_key_is_not_configured() -> None:
    response = client.post("/v1/gloss/normalize", json=payload())

    assert response.status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-Internal-API-Key": "wrong"}])
def test_configured_internal_key_rejects_missing_or_wrong_key(
    headers: dict[str, str],
) -> None:
    protected_client = TestClient(
        create_app(Settings(gloss_mode="template", internal_api_key="secret"))
    )

    response = protected_client.post("/v1/gloss/normalize", json=payload(), headers=headers)

    assert_error(
        response,
        status_code=401,
        code="UNAUTHORIZED",
        message="Authentication is required.",
    )
    assert "secret" not in response.text


def test_configured_internal_key_accepts_correct_key() -> None:
    protected_client = TestClient(
        create_app(Settings(gloss_mode="template", internal_api_key="secret"))
    )

    response = protected_client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Internal-API-Key": "secret"},
    )

    assert response.status_code == 200


def test_configured_internal_key_is_checked_before_body_validation() -> None:
    protected_client = TestClient(
        create_app(Settings(gloss_mode="template", internal_api_key="secret"))
    )

    response = protected_client.post(
        "/v1/gloss/normalize",
        json={"private": "must-not-be-echoed"},
        headers={"X-Request-ID": "req-auth-first"},
    )

    assert_error(
        response,
        status_code=401,
        code="UNAUTHORIZED",
        message="Authentication is required.",
        request_id="req-auth-first",
    )
    assert "must-not-be-echoed" not in response.text


def test_qwen_mode_without_token_installs_service_without_provider() -> None:
    application = create_app(Settings(gloss_mode="qwen", hf_token=None))

    assert application.state.gloss_service._provider is None
    response = TestClient(application).post("/v1/gloss/normalize", json=payload())
    assert response.status_code == 200
    assert response.json()["method"] == "template"
    assert response.json()["warnings"] == ["LLM_FALLBACK_PROVIDER_UNAVAILABLE"]


def test_qwen_mode_with_token_injects_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_provider = object()
    captured: dict[str, Settings] = {}

    def fake_from_settings(settings: Settings) -> object:
        captured["settings"] = settings
        return fake_provider

    monkeypatch.setattr(
        "src.main.HuggingFaceChatProvider.from_settings",
        fake_from_settings,
    )
    settings = Settings(gloss_mode="qwen", hf_token="hf-test-token")

    application = create_app(settings)

    assert captured == {"settings": settings}
    assert application.state.gloss_service._provider is fake_provider


def test_operation_log_excludes_token_labels_and_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="src.gloss.api")
    protected_client = TestClient(
        create_app(Settings(gloss_mode="template", internal_api_key="secret"))
    )

    response = protected_client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Internal-API-Key": "secret", "X-Request-ID": "req-log"},
    )

    assert response.status_code == 200
    assert "req-log" in caplog.text
    assert "NOT_UNDERSTAND" not in caplog.text
    assert "secret" not in caplog.text
