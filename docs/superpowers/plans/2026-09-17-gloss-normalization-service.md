# Gloss Normalization Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the English Gloss Normalization HTTP service with deterministic rules by default, optional Qwen normalization through a provider-neutral interface, and safe fallback behavior.

**Architecture:** A modular FastAPI application exposes health and Gloss endpoints. `GlossService` owns normalization policy, `normalize_with_rules` owns deterministic transformations, and `ChatProvider` isolates hosted LLM APIs; the first adapter uses Hugging Face Inference Providers. Tests inject fake providers so ordinary test runs never spend inference credits.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic 2, pydantic-settings, huggingface_hub, Uvicorn, pytest, pytest-asyncio, HTTPX

**Spec:** `docs/superpowers/specs/2026-09-17-gloss-recall-services-design.md`

## Global Constraints

- Work on branch `feat/gloss-normalization-service`.
- Support only `language="en"` in the MVP.
- `GLOSS_MODE=template` is the default; `GLOSS_MODE=qwen` is opt-in.
- Gloss must remain usable without network access, an HF token, or remaining HF credits.
- LLM access must occur only through `ChatProvider`; domain code must not import `huggingface_hub`.
- Never log `HF_TOKEN`, the internal API key, full prompts, or full token content.
- Every response must expose or echo `X-Request-ID`.
- Unit and contract tests must not call a real hosted provider.
- Use snake_case internally and the camelCase aliases defined by the HTTP contract externally.

---

## File Map

### Created

- `pyproject.toml` — runtime and development dependencies plus pytest configuration.
- `src/__init__.py` — application package marker.
- `src/main.py` — FastAPI app factory, middleware, exception handlers, and health routes.
- `src/gloss/__init__.py` — Gloss package marker.
- `src/gloss/rules.py` — deterministic templates and generic fallback.
- `src/shared/__init__.py` — shared package marker.
- `src/shared/providers/__init__.py` — provider package exports.
- `src/shared/providers/base.py` — provider-neutral `ChatProvider` protocol and provider exceptions.
- `tests/contract/test_health.py` — liveness and readiness contract tests.
- `tests/unit/test_gloss_rules.py` — deterministic-rule tests.
- `tests/unit/test_gloss_service.py` — template/Qwen/fallback policy tests.
- `tests/unit/test_huggingface_provider.py` — adapter parsing and error-mapping tests with a fake SDK client.
- `tests/contract/test_gloss_api.py` — HTTP contract, authentication, and request-ID tests.
- `tests/smoke/test_huggingface_gloss.py` — opt-in live provider smoke test.
- `contracts/openapi.json` — generated machine-readable OpenAPI contract.

### Modified

- `src/shared/config.py` — typed environment configuration.
- `src/shared/models.py` — shared HTTP, provider, and Gloss models.
- `src/shared/errors.py` — stable application errors and HTTP envelope.
- `src/shared/observability.py` — safe structured logging setup.
- `src/shared/providers/huggingface.py` — Hugging Face implementation of `ChatProvider`.
- `src/gloss/service.py` — Gloss application service.
- `src/gloss/api.py` — `/v1/gloss/normalize` router.
- `contracts/openapi.yaml` — generated/verified machine-readable API contract.
- `.env.example` — Gloss and generic LLM variables.
- `README.md` — installation, run, test, and configuration instructions.

---

### Task 1: Bootstrap the FastAPI Application and Typed Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/shared/__init__.py`
- Create: `src/gloss/__init__.py`
- Create: `src/main.py`
- Modify: `src/shared/config.py`
- Test: `tests/contract/test_health.py`

**Interfaces:**
- Produces: `Settings`, `get_settings()`, `create_app(settings: Settings | None = None) -> FastAPI`, and module-level `app`.
- Consumes: no application interfaces.

- [ ] **Step 1: Add packaging and test configuration**

Create `pyproject.toml` with these exact dependency groups:

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "isyara-ai-services"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115,<1",
  "huggingface-hub>=0.34,<2",
  "pydantic>=2.10,<3",
  "pydantic-settings>=2.7,<3",
  "uvicorn[standard]>=0.34,<1",
]

[project.optional-dependencies]
dev = [
  "httpx>=0.28,<1",
  "pytest>=8.3,<9",
  "pytest-asyncio>=0.25,<1",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
addopts = "-q"
asyncio_mode = "auto"
testpaths = ["tests"]
markers = ["smoke: calls an external provider and consumes hosted inference credit"]
```

- [ ] **Step 2: Install the editable development environment**

Run: `python -m pip install -e ".[dev]"`

Expected: installation completes and `python -c "import fastapi, huggingface_hub, pydantic_settings"` exits with code 0.

- [ ] **Step 3: Write the failing health/config contract test**

Create `tests/contract/test_health.py`:

```python
from fastapi.testclient import TestClient

from src.main import create_app
from src.shared.config import Settings


def test_liveness_does_not_require_hugging_face() -> None:
    app = create_app(Settings(hf_token=None, gloss_mode="template"))
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "isyara-ai-services"}


def test_template_readiness_is_ready_without_hf_token() -> None:
    app = create_app(Settings(hf_token=None, gloss_mode="template"))
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "isyara-ai-services",
        "checks": {"configuration": "ok", "gloss": "ready"},
    }


def test_qwen_readiness_is_degraded_without_hf_token() -> None:
    app = create_app(Settings(hf_token=None, gloss_mode="qwen"))
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["gloss"] == "llm_unavailable"
```

- [ ] **Step 4: Run the test and verify the bootstrap is missing**

Run: `python -m pytest tests/contract/test_health.py -v`

Expected: FAIL because `src.main.create_app` and the concrete `Settings` class do not exist.

- [ ] **Step 5: Implement typed settings**

Replace `src/shared/config.py` with:

```python
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    internal_api_key: SecretStr | None = None
    log_level: str = "INFO"
    request_timeout_seconds: float = 15.0

    llm_backend: Literal["huggingface"] = "huggingface"
    llm_model: str = "Qwen/Qwen3-4B"
    hf_token: SecretStr | None = None
    hf_provider: str = "auto"
    hf_use_structured_output: bool = False

    gloss_mode: Literal["template", "qwen"] = "template"
    gloss_max_tokens: int = 64
    gloss_llm_max_output_tokens: int = 96


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 6: Implement the app factory and health endpoints**

Create package marker files and `src/main.py`:

```python
from fastapi import FastAPI

from src.shared.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    application = FastAPI(title="Isyara AI Services", version="0.1.0")
    application.state.settings = resolved

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok", "service": "isyara-ai-services"}

    @application.get("/health/ready")
    async def ready() -> dict[str, object]:
        llm_ready = resolved.hf_token is not None
        gloss_status = "ready"
        status = "ready"
        if resolved.gloss_mode == "qwen" and not llm_ready:
            gloss_status = "llm_unavailable"
            status = "degraded"
        return {
            "status": status,
            "service": "isyara-ai-services",
            "checks": {"configuration": "ok", "gloss": gloss_status},
        }

    return application


app = create_app()
```

- [ ] **Step 7: Run the health tests**

Run: `python -m pytest tests/contract/test_health.py -v`

Expected: 3 tests PASS.

- [ ] **Step 8: Commit the bootstrap**

```bash
git add pyproject.toml src tests/contract/test_health.py
git commit -m "feat: bootstrap FastAPI service"
```

---

### Task 2: Define Gloss Models and Deterministic Rules

**Files:**
- Modify: `src/shared/models.py`
- Create: `src/gloss/rules.py`
- Test: `tests/unit/test_gloss_rules.py`

**Interfaces:**
- Produces: `ConfirmedSignToken`, `GlossNormalizeRequest`, `GlossNormalizeResponse`, `RuleNormalization`, and `normalize_with_rules(tokens)`.
- Consumes: `Settings.gloss_max_tokens` at the API/service boundary in later tasks.

- [ ] **Step 1: Write failing deterministic-rule tests**

Create `tests/unit/test_gloss_rules.py`:

```python
import pytest

from src.gloss.rules import normalize_with_rules
from src.shared.models import ConfirmedSignToken


def token(token_id: str, label: str) -> ConfirmedSignToken:
    return ConfirmedSignToken(id=token_id, label=label, confirmedAt="2026-09-17T10:00:00Z")


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["I", "NOT_UNDERSTAND"], "I do not understand."),
        (["YOU", "REPEAT", "PLEASE"], "Could you repeat that, please?"),
        (["REPEAT", "PLEASE"], "Please repeat that."),
        (["THANK_YOU"], "Thank you."),
        (["HELLO"], "Hello."),
        (["YES"], "Yes."),
        (["NO"], "No."),
    ],
)
def test_exact_templates(labels: list[str], expected: str) -> None:
    result = normalize_with_rules([token(str(index), label) for index, label in enumerate(labels)])
    assert result.text == expected
    assert result.warnings == []


def test_labels_are_canonicalized_before_matching() -> None:
    result = normalize_with_rules([token("1", "i"), token("2", "not-understand")])
    assert result.text == "I do not understand."


def test_unmatched_input_is_preserved_and_warned() -> None:
    result = normalize_with_rules([token("1", "NEED_HELP"), token("2", "NOW")])
    assert result.text == "Need help now."
    assert result.warnings == ["UNMATCHED_TEMPLATE"]
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/unit/test_gloss_rules.py -v`

Expected: FAIL because the Gloss models and rule function do not exist.

- [ ] **Step 3: Implement the HTTP/domain models**

Replace `src/shared/models.py` with models using `ConfigDict(populate_by_name=True)` and aliases:

```python
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConfirmedSignToken(APIModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    confirmed_at: datetime = Field(alias="confirmedAt")

    @field_validator("label")
    @classmethod
    def reject_blank_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label must not be blank")
        return value


class GlossNormalizeRequest(APIModel):
    utterance_id: str = Field(alias="utteranceId", min_length=1, max_length=128)
    language: str
    tokens: list[ConfirmedSignToken] = Field(min_length=1)


class GlossNormalizeResponse(APIModel):
    request_id: str = Field(alias="requestId")
    utterance_id: str = Field(alias="utteranceId")
    text: str
    method: Literal["template", "qwen"]
    source_token_ids: list[str] = Field(alias="sourceTokenIds")
    warnings: list[str]
    latency_ms: int = Field(alias="latencyMs", ge=0)


class LLMGlossResult(APIModel):
    text: str = Field(min_length=1, max_length=500)
    source_token_ids: list[str] = Field(alias="sourceTokenIds")


class RuleNormalization(APIModel):
    text: str
    source_token_ids: list[str]
    warnings: list[str]
```

- [ ] **Step 4: Implement exact templates and safe generic fallback**

Create `src/gloss/rules.py`:

```python
import re

from src.shared.models import ConfirmedSignToken, RuleNormalization


EXACT_TEMPLATES: dict[tuple[str, ...], str] = {
    ("I", "NOT_UNDERSTAND"): "I do not understand.",
    ("YOU", "REPEAT", "PLEASE"): "Could you repeat that, please?",
    ("REPEAT", "PLEASE"): "Please repeat that.",
    ("THANK_YOU",): "Thank you.",
    ("HELLO",): "Hello.",
    ("YES",): "Yes.",
    ("NO",): "No.",
}


def canonicalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.strip().upper()).strip("_")


def normalize_with_rules(tokens: list[ConfirmedSignToken]) -> RuleNormalization:
    labels = tuple(canonicalize_label(token.label) for token in tokens)
    source_ids = [token.id for token in tokens]
    exact = EXACT_TEMPLATES.get(labels)
    if exact is not None:
        return RuleNormalization(text=exact, sourceTokenIds=source_ids, warnings=[])

    words = " ".join(label.replace("_", " ").lower() for label in labels).strip()
    text = f"{words[:1].upper()}{words[1:]}."
    return RuleNormalization(
        text=text,
        sourceTokenIds=source_ids,
        warnings=["UNMATCHED_TEMPLATE"],
    )
```

- [ ] **Step 5: Run rule tests**

Run: `python -m pytest tests/unit/test_gloss_rules.py -v`

Expected: all parametrized and fallback tests PASS.

- [ ] **Step 6: Commit the deterministic domain**

```bash
git add src/shared/models.py src/gloss/rules.py tests/unit/test_gloss_rules.py
git commit -m "feat: add deterministic gloss rules"
```

---

### Task 3: Add the Provider-Neutral Chat Contract and Hugging Face Adapter

**Files:**
- Create: `src/shared/providers/__init__.py`
- Create: `src/shared/providers/base.py`
- Modify: `src/shared/providers/huggingface.py`
- Test: `tests/unit/test_huggingface_provider.py`

**Interfaces:**
- Produces: `ChatProvider.complete_json`, `ProviderError` subclasses, and `HuggingFaceChatProvider`.
- Consumes: `ChatMessage` and `Settings`.

- [ ] **Step 1: Write failing provider adapter tests**

Create a fake client whose `chat_completion` returns an object with `choices[0].message.content`. Cover these exact cases in `tests/unit/test_huggingface_provider.py`:

```python
import pytest

from src.shared.models import ChatMessage
from src.shared.providers.base import ProviderBadResponse, ProviderRateLimited
from src.shared.providers.huggingface import HuggingFaceChatProvider


class Message:
    content = '{"text":"Thank you.","sourceTokenIds":["tok_1"]}'


class Choice:
    message = Message()


class Response:
    choices = [Choice()]


class FakeClient:
    def chat_completion(self, **kwargs: object) -> Response:
        self.kwargs = kwargs
        return Response()


@pytest.mark.asyncio
async def test_complete_json_parses_content_and_forwards_generation_options() -> None:
    client = FakeClient()
    provider = HuggingFaceChatProvider(client=client, model="Qwen/Qwen3-4B")
    result = await provider.complete_json(
        [ChatMessage(role="user", content="normalize")],
        {"type": "object"},
        max_tokens=96,
        temperature=0,
    )
    assert result["text"] == "Thank you."
    assert client.kwargs["model"] == "Qwen/Qwen3-4B"
    assert client.kwargs["max_tokens"] == 96


@pytest.mark.asyncio
async def test_malformed_json_maps_to_bad_response() -> None:
    class BrokenMessage:
        content = "not-json"

    class BrokenClient(FakeClient):
        def chat_completion(self, **kwargs: object) -> Response:
            response = Response()
            response.choices[0].message = BrokenMessage()
            return response

    provider = HuggingFaceChatProvider(client=BrokenClient(), model="model")
    with pytest.raises(ProviderBadResponse):
        await provider.complete_json([], {}, max_tokens=10, temperature=0)


@pytest.mark.asyncio
async def test_429_maps_to_rate_limited() -> None:
    class Response429:
        status_code = 429

    class RateLimitError(Exception):
        response = Response429()

    class LimitedClient(FakeClient):
        def chat_completion(self, **kwargs: object) -> Response:
            raise RateLimitError("secret upstream details")

    provider = HuggingFaceChatProvider(client=LimitedClient(), model="model")
    with pytest.raises(ProviderRateLimited):
        await provider.complete_json([], {}, max_tokens=10, temperature=0)
```

- [ ] **Step 2: Run adapter tests and verify failure**

Run: `python -m pytest tests/unit/test_huggingface_provider.py -v`

Expected: FAIL because provider contracts are absent.

- [ ] **Step 3: Define the provider contract and stable exceptions**

Create `src/shared/providers/base.py`:

```python
from typing import Protocol

from src.shared.models import ChatMessage


class ProviderError(Exception):
    pass


class ProviderTimeout(ProviderError):
    pass


class ProviderRateLimited(ProviderError):
    pass


class ProviderUnavailable(ProviderError):
    pass


class ProviderBadResponse(ProviderError):
    pass


class ChatProvider(Protocol):
    async def complete_json(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, object],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        raise NotImplementedError
```

Export these names from `src/shared/providers/__init__.py`.

- [ ] **Step 4: Implement the Hugging Face adapter**

Replace `src/shared/providers/huggingface.py` with:

```python
import asyncio
import json
from typing import Any

from huggingface_hub import InferenceClient

from src.shared.config import Settings
from src.shared.models import ChatMessage
from src.shared.providers.base import (
    ProviderBadResponse,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


class HuggingFaceChatProvider:
    def __init__(
        self,
        *,
        client: object,
        model: str,
        use_structured_output: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._use_structured_output = use_structured_output

    @classmethod
    def from_settings(cls, settings: Settings) -> "HuggingFaceChatProvider":
        if settings.hf_token is None:
            raise ProviderUnavailable("HF token is not configured")
        client = InferenceClient(
            provider=settings.hf_provider,
            api_key=settings.hf_token.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )
        return cls(
            client=client,
            model=settings.llm_model,
            use_structured_output=settings.hf_use_structured_output,
        )

    async def complete_json(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, object],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._use_structured_output:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "isyara_response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        try:
            response = await asyncio.to_thread(
                getattr(self._client, "chat_completion"),
                **kwargs,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise ProviderTimeout("Provider timed out") from exc
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                raise ProviderRateLimited("Provider rate limited") from exc
            if isinstance(status, int) and status >= 500:
                raise ProviderUnavailable("Provider unavailable") from exc
            raise ProviderBadResponse("Provider rejected the request") from exc

        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderBadResponse("Provider returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ProviderBadResponse("Provider JSON must be an object")
        return parsed
```

- [ ] **Step 5: Run adapter tests**

Run: `python -m pytest tests/unit/test_huggingface_provider.py -v`

Expected: JSON parsing, option forwarding, malformed output, and 429 mapping tests PASS.

- [ ] **Step 6: Commit the provider boundary**

```bash
git add src/shared/providers tests/unit/test_huggingface_provider.py
git commit -m "feat: add provider-neutral chat adapter"
```

---

### Task 4: Implement Gloss Policy, Qwen Validation, and Safe Fallback

**Files:**
- Modify: `src/gloss/service.py`
- Test: `tests/unit/test_gloss_service.py`

**Interfaces:**
- Produces: `GlossService.normalize(request) -> GlossServiceResult`.
- Consumes: `ChatProvider`, `Settings`, `normalize_with_rules`, `LLMGlossResult`, and `ChatMessage`.

- [ ] **Step 1: Write failing service tests with an injected fake provider**

Create `tests/unit/test_gloss_service.py` with these behaviors:

```python
import pytest

from src.gloss.service import GlossService, GlossValidationError
from src.shared.config import Settings
from src.shared.models import ConfirmedSignToken, GlossNormalizeRequest
from src.shared.providers.base import ProviderTimeout


def request() -> GlossNormalizeRequest:
    return GlossNormalizeRequest(
        utteranceId="utt_1",
        language="en",
        tokens=[ConfirmedSignToken(id="tok_1", label="THANK_YOU", confirmedAt="2026-09-17T10:00:00Z")],
    )


class FakeProvider:
    def __init__(self, result: dict[str, object] | Exception) -> None:
        self.result = result
        self.calls = 0

    async def complete_json(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.asyncio
async def test_template_mode_does_not_call_provider() -> None:
    provider = FakeProvider({})
    result = await GlossService(Settings(gloss_mode="template"), provider).normalize(request())
    assert result.text == "Thank you."
    assert result.method == "template"
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_qwen_mode_accepts_valid_provenance() -> None:
    provider = FakeProvider({"text": "Thank you.", "sourceTokenIds": ["tok_1"]})
    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request())
    assert result.method == "qwen"
    assert result.source_token_ids == ["tok_1"]


@pytest.mark.asyncio
async def test_qwen_timeout_falls_back_to_template() -> None:
    provider = FakeProvider(ProviderTimeout())
    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request())
    assert result.method == "template"
    assert "LLM_FALLBACK_PROVIDER_TIMEOUT" in result.warnings


@pytest.mark.asyncio
async def test_qwen_wrong_source_ids_fall_back() -> None:
    provider = FakeProvider({"text": "Invented text", "sourceTokenIds": ["other"]})
    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request())
    assert result.method == "template"
    assert "LLM_FALLBACK_INVALID_PROVENANCE" in result.warnings


@pytest.mark.asyncio
async def test_qwen_mode_without_provider_falls_back() -> None:
    result = await GlossService(Settings(gloss_mode="qwen"), None).normalize(request())
    assert result.method == "template"
    assert "LLM_FALLBACK_PROVIDER_UNAVAILABLE" in result.warnings


@pytest.mark.asyncio
async def test_qwen_invalid_schema_falls_back() -> None:
    provider = FakeProvider({"text": 42, "sourceTokenIds": ["tok_1"]})
    result = await GlossService(Settings(gloss_mode="qwen"), provider).normalize(request())
    assert result.method == "template"
    assert "LLM_FALLBACK_INVALID_RESPONSE" in result.warnings


@pytest.mark.asyncio
async def test_token_limit_is_enforced() -> None:
    oversized = request().model_copy(
        update={"tokens": request().tokens * 2},
    )
    with pytest.raises(GlossValidationError, match="PAYLOAD_TOO_LARGE"):
        await GlossService(Settings(gloss_mode="template", gloss_max_tokens=1)).normalize(oversized)


@pytest.mark.asyncio
async def test_only_english_is_supported() -> None:
    indonesian = request().model_copy(update={"language": "id"})
    with pytest.raises(GlossValidationError, match="UNSUPPORTED_LANGUAGE"):
        await GlossService(Settings(gloss_mode="template")).normalize(indonesian)
```

- [ ] **Step 2: Run service tests and verify failure**

Run: `python -m pytest tests/unit/test_gloss_service.py -v`

Expected: FAIL because `GlossService` is not implemented.

- [ ] **Step 3: Implement `GlossService`**

Implement these exact public/result types in `src/gloss/service.py`:

```python
from dataclasses import dataclass

from src.shared.models import GlossNormalizeRequest


@dataclass(frozen=True, slots=True)
class GlossServiceResult:
    text: str
    method: str
    source_token_ids: list[str]
    warnings: list[str]


class GlossValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class GlossService:
    async def normalize(self, request: GlossNormalizeRequest) -> GlossServiceResult:
        raise NotImplementedError
```

The implementation must:

1. Reject `language != "en"` with `GlossValidationError("UNSUPPORTED_LANGUAGE")`.
2. Reject more than `settings.gloss_max_tokens` with `GlossValidationError("PAYLOAD_TOO_LARGE")`.
3. Compute the deterministic result before considering Qwen.
4. Return it immediately in template mode.
5. In Qwen mode, serialize only `id` and canonicalized `label` into the user message.
6. Parse the provider object with `LLMGlossResult.model_validate`.
7. Require source IDs to match the request exactly and in order.
8. Fall back to the deterministic result for missing provider, any `ProviderError`, schema failure, empty text, or invalid provenance, adding one stable warning code.

Replace the skeleton above with this complete implementation, using this system instruction verbatim:

```text
Convert only the supplied confirmed sign-token labels into one concise English sentence. Do not add facts, people, objects, times, or intent absent from the labels. Return only data matching the requested schema and repeat every sourceTokenId exactly once in the original order.
```

```python
import json
from dataclasses import dataclass

from pydantic import ValidationError

from src.gloss.rules import canonicalize_label, normalize_with_rules
from src.shared.config import Settings
from src.shared.models import ChatMessage, GlossNormalizeRequest, LLMGlossResult
from src.shared.providers.base import (
    ChatProvider,
    ProviderBadResponse,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


SYSTEM_PROMPT = (
    "Convert only the supplied confirmed sign-token labels into one concise English sentence. "
    "Do not add facts, people, objects, times, or intent absent from the labels. "
    "Return only data matching the requested schema and repeat every sourceTokenId exactly once "
    "in the original order."
)


@dataclass(frozen=True, slots=True)
class GlossServiceResult:
    text: str
    method: str
    source_token_ids: list[str]
    warnings: list[str]


class GlossValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class GlossService:
    def __init__(self, settings: Settings, provider: ChatProvider | None = None) -> None:
        self._settings = settings
        self._provider = provider

    async def normalize(self, request: GlossNormalizeRequest) -> GlossServiceResult:
        if request.language != "en":
            raise GlossValidationError("UNSUPPORTED_LANGUAGE")
        if len(request.tokens) > self._settings.gloss_max_tokens:
            raise GlossValidationError("PAYLOAD_TOO_LARGE")

        template = normalize_with_rules(request.tokens)
        template_result = GlossServiceResult(
            text=template.text,
            method="template",
            source_token_ids=template.source_token_ids,
            warnings=list(template.warnings),
        )
        if self._settings.gloss_mode == "template":
            return template_result
        if self._provider is None:
            return self._fallback(template_result, "LLM_FALLBACK_PROVIDER_UNAVAILABLE")

        payload = {
            "tokens": [
                {"id": token.id, "label": canonicalize_label(token.label)}
                for token in request.tokens
            ]
        }
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
        ]
        try:
            raw = await self._provider.complete_json(
                messages,
                LLMGlossResult.model_json_schema(by_alias=True),
                max_tokens=self._settings.gloss_llm_max_output_tokens,
                temperature=0,
            )
            llm_result = LLMGlossResult.model_validate(raw)
        except ProviderTimeout:
            return self._fallback(template_result, "LLM_FALLBACK_PROVIDER_TIMEOUT")
        except ProviderRateLimited:
            return self._fallback(template_result, "LLM_FALLBACK_RATE_LIMITED")
        except (ProviderUnavailable, ProviderBadResponse, ProviderError):
            return self._fallback(template_result, "LLM_FALLBACK_PROVIDER_UNAVAILABLE")
        except ValidationError:
            return self._fallback(template_result, "LLM_FALLBACK_INVALID_RESPONSE")

        expected_ids = [token.id for token in request.tokens]
        if llm_result.source_token_ids != expected_ids:
            return self._fallback(template_result, "LLM_FALLBACK_INVALID_PROVENANCE")
        return GlossServiceResult(
            text=llm_result.text,
            method="qwen",
            source_token_ids=llm_result.source_token_ids,
            warnings=[],
        )

    @staticmethod
    def _fallback(result: GlossServiceResult, warning: str) -> GlossServiceResult:
        return GlossServiceResult(
            text=result.text,
            method="template",
            source_token_ids=result.source_token_ids,
            warnings=[*result.warnings, warning],
        )
```

- [ ] **Step 4: Run service tests**

Run: `python -m pytest tests/unit/test_gloss_service.py -v`

Expected: template, valid Qwen, timeout fallback, provenance fallback, size, and language tests PASS.

- [ ] **Step 5: Commit the application service**

```bash
git add src/gloss/service.py tests/unit/test_gloss_service.py
git commit -m "feat: add gloss normalization policy"
```

---

### Task 5: Expose the Gloss HTTP Contract, Authentication, Request IDs, and Errors

**Files:**
- Modify: `src/shared/errors.py`
- Modify: `src/shared/observability.py`
- Modify: `src/gloss/api.py`
- Modify: `src/main.py`
- Test: `tests/contract/test_gloss_api.py`

**Interfaces:**
- Produces: `POST /v1/gloss/normalize`, shared error envelope, request-ID middleware, and optional internal-key enforcement.
- Consumes: `GlossService`, `GlossNormalizeRequest`, `GlossNormalizeResponse`, and `Settings`.

- [ ] **Step 1: Write failing HTTP contract tests**

Create `tests/contract/test_gloss_api.py` with a `TestClient(create_app(Settings(gloss_mode="template")))` and cover:

```python
def payload(language: str = "en") -> dict[str, object]:
    return {
        "utteranceId": "utt_1",
        "language": language,
        "tokens": [
            {"id": "tok_1", "label": "I", "confirmedAt": "2026-09-17T10:00:00Z"},
            {"id": "tok_2", "label": "NOT_UNDERSTAND", "confirmedAt": "2026-09-17T10:00:01Z"},
        ],
    }


def test_normalize_returns_contract_and_echoes_request_id() -> None:
    response = client.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Request-ID": "req-test-1"},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test-1"
    assert response.json() == {
        "requestId": "req-test-1",
        "utteranceId": "utt_1",
        "text": "I do not understand.",
        "method": "template",
        "sourceTokenIds": ["tok_1", "tok_2"],
        "warnings": [],
        "latencyMs": response.json()["latencyMs"],
    }


def test_unsupported_language_uses_error_envelope() -> None:
    response = client.post("/v1/gloss/normalize", json=payload("id"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_generated_request_id_is_returned_in_header_and_body() -> None:
    response = client.post("/v1/gloss/normalize", json=payload())
    assert response.headers["X-Request-ID"] == response.json()["requestId"]


def test_configured_internal_key_is_required() -> None:
    protected = TestClient(create_app(Settings(internal_api_key="secret", gloss_mode="template")))
    denied = protected.post("/v1/gloss/normalize", json=payload())
    allowed = protected.post(
        "/v1/gloss/normalize",
        json=payload(),
        headers={"X-Internal-API-Key": "secret"},
    )
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "UNAUTHORIZED"
    assert allowed.status_code == 200


def test_empty_tokens_use_shared_validation_envelope() -> None:
    invalid = payload()
    invalid["tokens"] = []
    response = client.post("/v1/gloss/normalize", json=invalid)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_more_than_configured_token_limit_returns_413() -> None:
    oversized = payload()
    oversized["tokens"] = [
        {"id": f"tok_{index}", "label": "HELLO", "confirmedAt": "2026-09-17T10:00:00Z"}
        for index in range(65)
    ]
    response = client.post("/v1/gloss/normalize", json=oversized)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
```

- [ ] **Step 2: Run contract tests and verify failure**

Run: `python -m pytest tests/contract/test_gloss_api.py -v`

Expected: FAIL because the router, middleware, and shared error handlers are absent.

- [ ] **Step 3: Implement stable errors and request IDs**

In `src/shared/errors.py`, define `AppError(code, message, status_code, retryable=False, details=None)` and a FastAPI handler returning:

```json
{
  "error": {
    "code": "UNSUPPORTED_LANGUAGE",
    "message": "Only English is supported.",
    "retryable": false,
    "requestId": "req-test-1",
    "details": {}
  }
}
```

Add handlers for `AppError` and `RequestValidationError`. Validation errors must use `INVALID_REQUEST` without returning raw request bodies.

In `src/main.py`, add HTTP middleware that validates an incoming `X-Request-ID` as a non-empty string of at most 128 characters, otherwise generates `str(uuid4())`; store it on `request.state.request_id` and include it in the response header.

- [ ] **Step 4: Implement authentication and the Gloss router**

Replace `src/gloss/api.py` with an `APIRouter(prefix="/v1/gloss", tags=["gloss"])`. Its normalize route must:

1. Use `hmac.compare_digest` when `internal_api_key` is configured.
2. Retrieve `GlossService` from `request.app.state.gloss_service`.
3. Measure latency with `time.perf_counter`.
4. Map `GlossValidationError` codes to 413 or 422 `AppError` responses.
5. Return `GlossNormalizeResponse` using camelCase aliases.

Update `create_app` to construct `HuggingFaceChatProvider` only when Qwen mode and an HF token are configured, construct `GlossService`, register exception handlers, and include the Gloss router.

- [ ] **Step 5: Add safe logging**

Implement `configure_logging(level: str)` and `get_logger(name: str)` in `src/shared/observability.py` using the standard library. Log operation, request ID, mode, status, latency, and token count; never include token labels, prompt content, or secrets.

- [ ] **Step 6: Run contract and full unit tests**

Run: `python -m pytest tests/contract/test_gloss_api.py tests/unit -v`

Expected: all tests PASS and no external network call occurs.

- [ ] **Step 7: Commit the HTTP service**

```bash
git add src/main.py src/gloss/api.py src/shared/errors.py src/shared/observability.py tests/contract/test_gloss_api.py
git commit -m "feat: expose gloss normalization API"
```

---

### Task 6: Publish the Contract, Documentation, and Opt-In Smoke Test

**Files:**
- Modify: `contracts/openapi.yaml`
- Modify: `.env.example`
- Modify: `README.md`
- Create: `tests/smoke/test_huggingface_gloss.py`

**Interfaces:**
- Produces: reproducible setup/run commands, complete OpenAPI, and an explicit live-provider verification path.
- Consumes: the finished application and `HuggingFaceChatProvider`.

- [ ] **Step 1: Add an OpenAPI contract regression test**

Extend `tests/contract/test_gloss_api.py`:

```python
def test_openapi_contains_gloss_and_health_paths() -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/gloss/normalize" in schema["paths"]
    assert "/health/live" in schema["paths"]
    assert "/health/ready" in schema["paths"]
    response_schema = schema["paths"]["/v1/gloss/normalize"]["post"]["responses"]["200"]
    assert response_schema["content"]["application/json"]["schema"]
```

- [ ] **Step 2: Run the OpenAPI regression test**

Run: `python -m pytest tests/contract/test_gloss_api.py::test_openapi_contains_gloss_and_health_paths -v`

Expected: PASS against the runtime schema.

- [ ] **Step 3: Export and verify the machine-readable contract**

Run this formatting command after implementation:

```powershell
python -c "import json; from src.main import app; print(json.dumps(app.openapi(), indent=2))" | Set-Content -Encoding utf8 contracts/openapi.json
```

Then replace the placeholder `contracts/openapi.yaml` with a short pointer explaining that `contracts/openapi.json` is generated from the FastAPI schema and is the machine-readable source for the implemented service. Add `contracts/openapi.json` to the task commit.

- [ ] **Step 4: Document environment and local operation**

Update `.env.example` to include exactly:

```dotenv
LLM_BACKEND=huggingface
LLM_MODEL=Qwen/Qwen3-4B
HF_TOKEN=
HF_PROVIDER=auto
HF_USE_STRUCTURED_OUTPUT=false
GLOSS_MODE=template
GLOSS_MAX_TOKENS=64
GLOSS_LLM_MAX_OUTPUT_TOKENS=96
```

Update `README.md` with:

```text
python -m pip install -e ".[dev]"
python -m uvicorn src.main:app --reload
python -m pytest
```

Explain template and Qwen modes, limited HF free-tier behavior, required inference-token permissions, request examples, fallback warnings, and how to add another `ChatProvider`.

- [ ] **Step 5: Add an opt-in Hugging Face smoke test**

Create `tests/smoke/test_huggingface_gloss.py` that is skipped unless both `RUN_HF_SMOKE=1` and `HF_TOKEN` are set. It must instantiate `Settings(gloss_mode="qwen")`, call `GlossService` with `THANK_YOU`, and assert non-empty text with unchanged `source_token_ids`. Mark it `@pytest.mark.smoke`.

- [ ] **Step 6: Run offline verification**

Run: `python -m pytest -m "not smoke" -v`

Expected: all unit and contract tests PASS without network access.

Run: `python -m compileall -q src tests`

Expected: exit code 0.

- [ ] **Step 7: Run live smoke verification only when explicitly enabled**

Run: `$env:RUN_HF_SMOKE='1'; python -m pytest tests/smoke/test_huggingface_gloss.py -v`

Expected with a valid funded/free-tier HF token: PASS. If no token is available, do not run this command; the test remains skipped in the normal suite.

- [ ] **Step 8: Commit documentation and contract artifacts**

```bash
git add .env.example README.md contracts tests/smoke/test_huggingface_gloss.py tests/contract/test_gloss_api.py
git commit -m "docs: publish gloss service contract"
```

---

### Task 7: Final Gloss Branch Verification

**Files:**
- Verify only; modify files only when a failing check identifies a concrete defect.

**Interfaces:**
- Consumes all Gloss branch deliverables.
- Produces a branch ready for review and merge before Recall work starts.

- [ ] **Step 1: Run the complete offline test suite**

Run: `python -m pytest -m "not smoke" -v`

Expected: all tests PASS, zero deselected tests other than the single smoke test, and no network calls.

- [ ] **Step 2: Verify source compilation**

Run: `python -m compileall -q src tests`

Expected: exit code 0.

- [ ] **Step 3: Verify the generated contract is current**

Regenerate OpenAPI into a temporary file and compare it with `contracts/openapi.json`. Normalize JSON key order before comparison so formatting differences do not fail the check.

Expected: no semantic difference.

- [ ] **Step 4: Verify secrets are absent**

Run: `git grep -n -E "hf_[A-Za-z0-9]{20,}|INTERNAL_API_KEY=.+|HF_TOKEN=.+" -- ':!docs/superpowers/**'`

Expected: no committed credential values.

- [ ] **Step 5: Inspect the final diff**

Run: `git diff main...HEAD --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: clean working tree.

- [ ] **Step 6: Record verification evidence in the handoff**

Report the exact test commands and pass counts, whether the optional live smoke test ran, and any provider limitation observed. Do not claim live Hugging Face compatibility unless the opt-in smoke test actually ran successfully.
