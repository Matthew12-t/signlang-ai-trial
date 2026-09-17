from pathlib import Path

from fastapi.testclient import TestClient

from src.sign.api import create_app
from src.sign.config import Settings


def test_mock_service_health_smoke(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocabulary.json"
    vocabulary.write_text(
        '{"version":"mvp-en-v1","modelVersion":"signbart-mvp-v1",'
        '"labels":[{"index":0,"label":"REPEAT"}]}',
        encoding="utf-8",
    )
    settings = Settings(
        model_backend="mock",
        vocabulary_path=vocabulary,
        confident_threshold=0.8,
        unknown_threshold=0.4,
        min_top1_top2_margin=0.2,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").json() == {
            "status": "alive",
            "service": "sign-language",
        }
        ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
