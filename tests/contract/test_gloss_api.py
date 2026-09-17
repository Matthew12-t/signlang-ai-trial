from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.shared.config import Settings


def offline_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "internal_api_key": None,
        "log_level": "INFO",
        "request_timeout_seconds": 15,
        "llm_backend": "huggingface",
        "llm_model": "Qwen/Qwen3-4B",
        "hf_token": None,
        "hf_provider": "auto",
        "hf_use_structured_output": False,
        "gloss_mode": "template",
        "gloss_max_tokens": 64,
        "gloss_llm_max_output_tokens": 96,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


client = TestClient(create_app(offline_settings()))


def test_openapi_contains_gloss_and_health_paths() -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/gloss/normalize" in schema["paths"]
    assert "/health/live" in schema["paths"]
    assert "/health/ready" in schema["paths"]
    response_schema = schema["paths"]["/v1/gloss/normalize"]["post"]["responses"]["200"]
    assert response_schema["content"]["application/json"]["schema"]


def test_openapi_documents_auth_request_ids_and_shared_error_envelopes() -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/v1/gloss/normalize"]["post"]

    assert schema["components"]["securitySchemes"]["InternalApiKey"] == {
        "type": "apiKey",
        "description": "Required only when INTERNAL_API_KEY is configured.",
        "in": "header",
        "name": "X-Internal-API-Key",
    }
    assert operation["security"] == [{"InternalApiKey": []}, {}]
    assert any(parameter["name"] == "X-Request-ID" for parameter in operation["parameters"])
    for status in ("200", "400", "401", "413", "422"):
        assert "X-Request-ID" in operation["responses"][status]["headers"]
    for status in ("400", "401", "413", "422"):
        response_schema = operation["responses"][status]["content"]["application/json"]["schema"]
        assert response_schema == {"$ref": "#/components/schemas/ErrorEnvelope"}

    for path in ("/health/live", "/health/ready"):
        health_operation = schema["paths"][path]["get"]
        assert any(
            parameter["name"] == "X-Request-ID"
            for parameter in health_operation["parameters"]
        )
        assert "X-Request-ID" in health_operation["responses"]["200"]["headers"]


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
        status_code=400,
        code="INVALID_REQUEST",
        message="The request is invalid.",
        request_id="req-invalid",
    )
    assert "must-not-be-echoed" not in response.text


@pytest.mark.parametrize("label", ["___", "???"])
def test_label_without_usable_content_is_invalid_request(label: str) -> None:
    request_payload = payload()
    request_payload["tokens"][0]["label"] = label

    response = client.post("/v1/gloss/normalize", json=request_payload)

    assert_error(
        response,
        status_code=400,
        code="INVALID_REQUEST",
        message="The request is invalid.",
    )


def test_unknown_labels_preserve_content_and_source_ids() -> None:
    request_payload = payload()
    request_payload["tokens"] = [
        {
            "id": "tok_cpp",
            "label": "C++",
            "confirmedAt": "2026-09-17T10:00:00Z",
        },
        {
            "id": "tok_cafe",
            "label": "Café",
            "confirmedAt": "2026-09-17T10:00:01Z",
        },
    ]

    response = client.post("/v1/gloss/normalize", json=request_payload)

    assert response.status_code == 200
    assert response.json()["text"] == "C++ café."
    assert response.json()["sourceTokenIds"] == ["tok_cpp", "tok_cafe"]
    assert response.json()["warnings"] == ["UNMATCHED_TEMPLATE"]


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
        create_app(offline_settings(internal_api_key="secret"))
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
        create_app(offline_settings(internal_api_key="secret"))
    )

    response = protected_client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Internal-API-Key": "secret"},
    )

    assert response.status_code == 200


def test_configured_internal_key_is_checked_before_body_validation() -> None:
    protected_client = TestClient(
        create_app(offline_settings(internal_api_key="secret"))
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


@pytest.mark.parametrize("provided_key", [None, "wrong"])
def test_configured_internal_key_is_checked_before_malformed_json(
    provided_key: str | None,
) -> None:
    protected_client = TestClient(
        create_app(offline_settings(internal_api_key="secret"))
    )
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": "req-malformed-auth",
    }
    if provided_key is not None:
        headers["X-Internal-API-Key"] = provided_key

    response = protected_client.post(
        "/v1/gloss/normalize",
        content=b'{"utteranceId":',
        headers=headers,
    )

    assert_error(
        response,
        status_code=401,
        code="UNAUTHORIZED",
        message="Authentication is required.",
        request_id="req-malformed-auth",
    )


@pytest.mark.parametrize("provided_key", [None, "wrong"])
def test_root_path_gloss_route_rejects_missing_or_wrong_key(
    provided_key: str | None,
) -> None:
    protected_client = TestClient(
        create_app(offline_settings(internal_api_key="secret")),
        root_path="/prefix",
    )
    headers = {"X-Request-ID": "req-root-auth"}
    if provided_key is not None:
        headers["X-Internal-API-Key"] = provided_key

    response = protected_client.post(
        "/prefix/v1/gloss/normalize",
        json=payload(),
        headers=headers,
    )

    assert_error(
        response,
        status_code=401,
        code="UNAUTHORIZED",
        message="Authentication is required.",
        request_id="req-root-auth",
    )


def test_root_path_gloss_route_accepts_correct_key() -> None:
    protected_client = TestClient(
        create_app(offline_settings(internal_api_key="secret")),
        root_path="/prefix",
    )

    response = protected_client.post(
        "/prefix/v1/gloss/normalize",
        json=payload(),
        headers={"X-Internal-API-Key": "secret"},
    )

    assert response.status_code == 200


def test_root_path_gloss_auth_precedes_malformed_json_parsing() -> None:
    protected_client = TestClient(
        create_app(offline_settings(internal_api_key="secret")),
        root_path="/prefix",
    )

    response = protected_client.post(
        "/prefix/v1/gloss/normalize",
        content=b'{"utteranceId":',
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": "req-root-malformed",
        },
    )

    assert_error(
        response,
        status_code=401,
        code="UNAUTHORIZED",
        message="Authentication is required.",
        request_id="req-root-malformed",
    )


def test_non_ascii_internal_key_is_safely_rejected() -> None:
    protected_client = TestClient(
        create_app(offline_settings(internal_api_key="secret")),
        raise_server_exceptions=False,
    )

    response = protected_client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers=[
            (b"x-internal-api-key", b"\xff"),
            (b"x-request-id", b"req-nonascii-key"),
        ],
    )

    assert_error(
        response,
        status_code=401,
        code="UNAUTHORIZED",
        message="Authentication is required.",
        request_id="req-nonascii-key",
    )


def test_unexpected_service_error_is_sanitized_with_request_id() -> None:
    class ExplodingGlossService:
        async def normalize(self, request: object) -> object:
            raise RuntimeError("provider URL and secret must never escape")

    application = create_app(offline_settings())
    application.state.gloss_service = ExplodingGlossService()
    failing_client = TestClient(application, raise_server_exceptions=False)

    response = failing_client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Request-ID": "req-internal"},
    )

    assert_error(
        response,
        status_code=500,
        code="INTERNAL_ERROR",
        message="An internal error occurred.",
        request_id="req-internal",
    )
    assert "provider URL" not in response.text
    assert "secret" not in response.text


def test_method_not_allowed_uses_shared_error_and_preserves_allow_header() -> None:
    response = client.get(
        "/v1/gloss/normalize",
        headers={"X-Request-ID": "req-method"},
    )

    assert_error(
        response,
        status_code=405,
        code="METHOD_NOT_ALLOWED",
        message="Method not allowed.",
        request_id="req-method",
    )
    assert response.headers["Allow"] == "POST"


def test_qwen_mode_without_token_installs_service_without_provider() -> None:
    application = create_app(offline_settings(gloss_mode="qwen"))

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
    settings = offline_settings(gloss_mode="qwen", hf_token="hf-test-token")

    application = create_app(settings)

    assert captured == {"settings": settings}
    assert application.state.gloss_service._provider is fake_provider


def test_operation_log_excludes_token_labels_and_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="src.gloss.api")
    protected_client = TestClient(
        create_app(offline_settings(internal_api_key="secret"))
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
    assert "method=template" in caplog.text
    assert "fallback=false" in caplog.text
    assert "configured_provider=none" in caplog.text
    assert "configured_model=none" in caplog.text


def test_qwen_fallback_log_includes_safe_result_and_configuration_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="src.gloss.api")
    qwen_client = TestClient(
        create_app(
            offline_settings(
                gloss_mode="qwen",
                hf_token=None,
                hf_provider="auto",
                llm_model="Qwen/Qwen3-4B",
            )
        )
    )

    response = qwen_client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Request-ID": "req-qwen-log"},
    )

    assert response.status_code == 200
    assert "method=template" in caplog.text
    assert "fallback=true" in caplog.text
    assert "warnings=LLM_FALLBACK_PROVIDER_UNAVAILABLE" in caplog.text
    assert "configured_provider=auto" in caplog.text
    assert "configured_model=Qwen/Qwen3-4B" in caplog.text
    assert "NOT_UNDERSTAND" not in caplog.text


def test_explicit_offline_settings_ignore_host_qwen_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLOSS_MODE", "qwen")
    monkeypatch.setenv("HF_TOKEN", "host-secret")

    application = create_app(offline_settings())

    response = TestClient(application).post("/v1/gloss/normalize", json=payload())

    assert response.status_code == 200
    assert response.json()["method"] == "template"
