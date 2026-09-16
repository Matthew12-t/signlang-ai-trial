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
