from fastapi.testclient import TestClient

from src.main import create_app
from src.recall.retrieval import ProvidedContextRetriever
from src.recall.service import RecallService
from src.shared.config import Settings
from src.shared.providers import ProviderTimeout


class FakeProvider:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    async def complete_json(self, messages, response_schema, **options):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def settings(**overrides: object) -> Settings:
    values = {
        "enabled_services": ("recall",),
        "preload_models": False,
        "hf_token": None,
        "recall_model": "Qwen/Qwen3-8B",
        "recall_max_context_chars": 24_000,
        "recall_max_answer_tokens": 160,
        "recall_temperature": 0.0,
    }
    values.update(overrides)
    return Settings(**values)


def payload(*, language: str = "en", entries: list[dict] | None = None) -> dict:
    return {
        "sessionId": "ses_1",
        "query": "When is the deadline?",
        "language": language,
        "contextEntries": [
            {
                "id": "tr_17",
                "sequence": 17,
                "source": "speech",
                "speaker": "Participant",
                "text": "The submission deadline is Friday at five PM.",
                "startedAt": "2026-09-18T10:18:11Z",
                "endedAt": "2026-09-18T10:18:14Z",
            }
        ]
        if entries is None
        else entries,
        "options": {"maxAnswerTokens": 160, "requireEvidence": True},
    }


def client_with_provider(provider: FakeProvider, **overrides: object) -> TestClient:
    configured = settings(**overrides)
    app = create_app(configured)
    app.state.recall_service = RecallService(
        provider=provider,
        retriever=ProvidedContextRetriever(),
        model=configured.recall_model,
        max_context_chars=configured.recall_max_context_chars,
        max_answer_tokens=configured.recall_max_answer_tokens,
        temperature=configured.recall_temperature,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_recall_returns_grounded_answer_with_request_id_and_model() -> None:
    provider = FakeProvider(
        {
            "answer": "The deadline is Friday at 5 PM.",
            "grounded": True,
            "evidence": [{"entryId": "tr_17", "quote": "deadline is Friday"}],
            "notFoundReason": None,
        }
    )

    response = client_with_provider(provider).post(
        "/v1/recall/query",
        json=payload(),
        headers={"X-Request-ID": "req-recall-1"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-recall-1"
    assert response.json() == {
        "requestId": "req-recall-1",
        "answer": "The deadline is Friday at 5 PM.",
        "grounded": True,
        "evidence": [
            {
                "entryId": "tr_17",
                "quote": "deadline is Friday",
                "startedAt": "2026-09-18T10:18:11Z",
            }
        ],
        "notFoundReason": None,
        "model": {"provider": "huggingface", "id": "Qwen/Qwen3-8B"},
        "latencyMs": response.json()["latencyMs"],
    }


def test_empty_context_short_circuits_without_provider() -> None:
    provider = FakeProvider(AssertionError("must not be called"))

    response = client_with_provider(provider).post(
        "/v1/recall/query", json=payload(entries=[])
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["notFoundReason"] == "EMPTY_CONTEXT"
    assert provider.calls == 0


def test_recall_rejects_unsupported_language() -> None:
    response = client_with_provider(FakeProvider({})).post(
        "/v1/recall/query", json=payload(language="id")
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_recall_authentication_happens_before_malformed_body_parsing() -> None:
    response = client_with_provider(
        FakeProvider({}), internal_api_key="secret"
    ).post(
        "/v1/recall/query",
        content=b"{broken",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_recall_maps_provider_timeout_to_shared_error() -> None:
    response = client_with_provider(FakeProvider(ProviderTimeout())).post(
        "/v1/recall/query", json=payload()
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "HF_TIMEOUT"


def test_recall_readiness_is_degraded_without_hf_token() -> None:
    response = TestClient(create_app(settings())).get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["recall"] == "llm_unavailable"


def test_openapi_documents_recall_contract() -> None:
    schema = TestClient(create_app(settings())).get("/openapi.json").json()
    operation = schema["paths"]["/v1/recall/query"]["post"]

    assert operation["security"] == [{"InternalApiKey": []}, {}]
    for status in ("200", "400", "401", "413", "422", "429", "502", "503", "504"):
        assert "X-Request-ID" in operation["responses"][status]["headers"]
