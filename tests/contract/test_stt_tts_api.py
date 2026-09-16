from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient

from src.main import create_app
from src.shared.config import Settings
from src.shared.models import (
    GeneratedAudio,
    ProviderTranscript,
    ProviderTranscriptSegment,
)
from src.stt.service import STTService
from src.tts.service import TTSService


class FakeSTTProvider:
    model_id = "Systran/faster-whisper-large-v3"
    provider_name = "local-ctranslate2"
    ready = True
    load_error = None

    async def transcribe(
        self, audio: bytes, *, language: str, include_timestamps: bool
    ) -> ProviderTranscript:
        segments = (
            ProviderTranscriptSegment("Hello world.", 0.0, 1.25),
        ) if include_timestamps else ()
        return ProviderTranscript("Hello world.", language, segments)


class FakeTTSProvider:
    model_id = "openbmb/VoxCPM2"
    ready = True
    load_error = None

    async def synthesize(self, text: str) -> GeneratedAudio:
        return GeneratedAudio(
            samples=np.zeros(480, dtype=np.float32),
            sample_rate=48_000,
        )


def _client() -> TestClient:
    settings = replace(
        Settings.from_env(),
        enabled_services=("stt", "tts"),
        preload_models=False,
        internal_api_key="test-secret",
    )
    return TestClient(
        create_app(
            settings=settings,
            stt_service=STTService(FakeSTTProvider(), settings),
            tts_service=TTSService(FakeTTSProvider(), settings),
        )
    )


def test_stt_contract() -> None:
    with _client() as client:
        response = client.post(
            "/v1/stt/transcriptions",
            headers={"X-Internal-API-Key": "test-secret"},
            files={"audio": ("sample.wav", b"RIFF-test", "audio/wav")},
            data={"language": "en", "encoding": "wav", "timestamps": "true"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Hello world."
    assert body["segments"][0]["endMs"] == 1250
    assert body["model"]["id"] == "Systran/faster-whisper-large-v3"
    assert response.headers["X-Request-ID"] == body["requestId"]


def test_tts_contract() -> None:
    with _client() as client:
        response = client.post(
            "/v1/tts/synthesize",
            headers={
                "X-Internal-API-Key": "test-secret",
                "Idempotency-Key": "tts-contract-test",
            },
            json={"text": "Hello", "language": "en", "voice": None, "format": "wav"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["X-Model-ID"] == "openbmb/VoxCPM2"
    assert response.headers["X-Audio-Sample-Rate"] == "48000"
    assert response.content.startswith(b"RIFF")


def test_internal_api_key_is_enforced() -> None:
    with _client() as client:
        response = client.post(
            "/v1/stt/transcriptions",
            files={"audio": ("sample.wav", b"RIFF-test", "audio/wav")},
            data={"language": "en", "encoding": "wav"},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_readiness_reports_injected_models() -> None:
    with _client() as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "ai-services",
        "checks": {"stt": "ready", "tts": "ready"},
    }
