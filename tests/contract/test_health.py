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


def test_liveness_does_not_require_hugging_face() -> None:
    app = create_app(offline_settings())
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "isyara-ai-services"}


def test_template_readiness_is_ready_without_hf_token() -> None:
    app = create_app(offline_settings())
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "isyara-ai-services",
        "checks": {"configuration": "ok", "gloss": "ready"},
    }


def test_qwen_readiness_is_degraded_without_hf_token() -> None:
    app = create_app(offline_settings(gloss_mode="qwen"))
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["gloss"] == "llm_unavailable"


def test_qwen_readiness_is_degraded_with_whitespace_hf_token() -> None:
    app = create_app(offline_settings(hf_token="   ", gloss_mode="qwen"))

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["gloss"] == "llm_unavailable"
