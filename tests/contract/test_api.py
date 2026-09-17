from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np

from src.sign.api import create_app
from src.sign.config import Settings
from src.sign.model import MockSignRecognitionModel
from src.sign.schemas import DecodedVideo, Frame, ModelInputConfig, Vocabulary
from src.sign.service import SignPredictionService


class StubDecoder:
    def decode(self, video: bytes, content_type: str) -> DecodedVideo:
        return DecodedVideo(
            frames=tuple(
                Frame(data=np.zeros((8, 8, 3), dtype=np.uint8), timestamp_ms=index * 100)
                for index in range(4)
            ),
            duration_ms=400,
        )


def make_client(
    tmp_path: Path,
    *,
    internal_api_key: str = "test-key",
    allowed_origins: tuple[str, ...] = (),
    allow_private_network: bool = False,
) -> TestClient:
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(
        '{"version":"mvp-en-v1","modelVersion":"signbart-mvp-v1",'
        '"labels":[{"index":0,"label":"REPEAT","displayText":"Repeat"},'
        '{"index":1,"label":"AGAIN","displayText":"Again"}]}',
        encoding="utf-8",
    )
    settings = Settings(
        model_backend="mock",
        vocabulary_path=vocabulary_path,
        vocabulary_version="mvp-en-v1",
        confident_threshold=0.80,
        unknown_threshold=0.40,
        min_top1_top2_margin=0.20,
        internal_api_key=internal_api_key,
        allowed_origins=allowed_origins,
        allow_private_network=allow_private_network,
    )
    vocabulary = Vocabulary.from_file(vocabulary_path)
    model = MockSignRecognitionModel(
        vocabulary,
        ModelInputConfig(settings.required_frames, settings.input_width, settings.input_height, settings.input_layout),
    )
    service = SignPredictionService(settings, decoder=StubDecoder(), model=model)
    return TestClient(create_app(settings, service))


def test_predict_contract_returns_request_id_and_confirmation_fields(tmp_path: Path) -> None:
    request_id = uuid4()
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sign/predict",
            headers={"X-Request-ID": str(request_id), "X-Internal-API-Key": "test-key"},
            files={"video": ("clip.webm", b"video-bytes", "video/webm")},
            data={"topK": "2", "vocabularyVersion": "mvp-en-v1"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["requestId"] == str(request_id)
    assert body["predictionId"].startswith("pred_")
    assert body["status"] == "CONFIDENT"
    assert body["prediction"] == "REPEAT"
    assert body["requiresConfirmation"] is False
    assert len(body["candidates"]) == 2
    assert body["modelVersion"] == "mock-sign-v0"


def test_missing_api_key_uses_error_envelope(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sign/predict",
            headers={"X-Request-ID": str(uuid4())},
            files={"video": ("clip.webm", b"video-bytes", "video/webm")},
            data={"topK": "2", "vocabularyVersion": "mvp-en-v1"},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_browser_local_prediction_does_not_require_api_key(tmp_path: Path) -> None:
    request_id = uuid4()
    with make_client(
        tmp_path,
        internal_api_key="",
        allowed_origins=("http://localhost:5173",),
    ) as client:
        response = client.post(
            "/v1/sign/predict",
            headers={
                "Origin": "http://localhost:5173",
                "X-Request-ID": str(request_id),
            },
            files={"video": ("clip.webm", b"video-bytes", "video/webm")},
            data={"topK": "2", "vocabularyVersion": "mvp-en-v1"},
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.json()["requestId"] == str(request_id)


def test_private_deployment_rejects_missing_and_wrong_api_key(tmp_path: Path) -> None:
    with make_client(tmp_path, internal_api_key="test-key") as client:
        missing = client.post(
            "/v1/sign/predict",
            headers={"X-Request-ID": str(uuid4())},
            files={"video": ("clip.webm", b"video-bytes", "video/webm")},
            data={"topK": "2", "vocabularyVersion": "mvp-en-v1"},
        )
        wrong = client.post(
            "/v1/sign/predict",
            headers={"X-Request-ID": str(uuid4()), "X-Internal-API-Key": "wrong-key"},
            files={"video": ("clip.webm", b"video-bytes", "video/webm")},
            data={"topK": "2", "vocabularyVersion": "mvp-en-v1"},
        )
    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_allowed_origin_preflight_returns_cors_headers(tmp_path: Path) -> None:
    with make_client(
        tmp_path,
        allowed_origins=("http://localhost:5173",),
        allow_private_network=True,
    ) as client:
        response = client.options(
            "/v1/sign/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-request-id",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    assert "x-request-id" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-private-network" not in response.headers


def test_private_network_preflight_requires_flag_and_allowed_origin(tmp_path: Path) -> None:
    with make_client(
        tmp_path,
        allowed_origins=("http://localhost:5173",),
        allow_private_network=True,
    ) as client:
        allowed = client.options(
            "/v1/sign/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-request-id",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        denied = client.options(
            "/v1/sign/predict",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-request-id",
                "Access-Control-Request-Private-Network": "true",
            },
        )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-private-network"] == "true"
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers
    assert "access-control-allow-private-network" not in denied.headers


def test_unlisted_origin_does_not_receive_cors_permission(tmp_path: Path) -> None:
    with make_client(tmp_path, allowed_origins=("http://localhost:5173",)) as client:
        response = client.options(
            "/v1/sign/predict",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-request-id",
            },
        )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_prediction_does_not_write_video_to_disk(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
        response = client.post(
            "/v1/sign/predict",
            headers={"X-Request-ID": str(uuid4()), "X-Internal-API-Key": "test-key"},
            files={"video": ("clip.webm", b"video-bytes", "video/webm")},
            data={"topK": "2", "vocabularyVersion": "mvp-en-v1"},
        )
        after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert response.status_code == 200
    assert after == before


def test_empty_clip_returns_model_level_invalid_input(tmp_path: Path) -> None:
    request_id = uuid4()
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sign/predict",
            headers={"X-Request-ID": str(request_id), "X-Internal-API-Key": "test-key"},
            files={"video": ("clip.webm", b"", "video/webm")},
            data={"topK": "2", "vocabularyVersion": "mvp-en-v1"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["requestId"] == str(request_id)
    assert body["status"] == "INVALID_INPUT"
    assert body["prediction"] is None
    assert body["candidates"] == []


def test_unready_model_is_explicit(tmp_path: Path) -> None:
    settings = Settings(vocabulary_path=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"
