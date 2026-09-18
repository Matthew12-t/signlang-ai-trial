import io
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.providers.huggingface import (
    HfApiSTTProvider,
    HfApiTTSProvider,
    _decode_audio,
    _to_transcript,
)


def _settings(**overrides) -> Settings:
    base = replace(Settings.from_env(), hf_token="hf_test_token")
    return replace(base, **overrides) if overrides else base


def _encoded(samples: np.ndarray, sample_rate: int, fmt: str = "WAV") -> bytes:
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, sample_rate, format=fmt)
    return buffer.getvalue()


class _FakeClient:
    def __init__(self, *, asr=None, tts=None, error=None) -> None:
        self._asr = asr
        self._tts = tts
        self._error = error
        self.calls: list[dict] = []

    async def automatic_speech_recognition(self, audio, *, model, extra_body=None):
        self.calls.append({"audio": audio, "model": model, "extra_body": extra_body})
        if self._error is not None:
            raise self._error
        return self._asr

    async def text_to_speech(self, text, *, model):
        self.calls.append({"text": text, "model": model})
        if self._error is not None:
            raise self._error
        return self._tts


def _with_client(provider, client):
    provider._build_client = lambda: client
    return provider


def _http_error(status: int) -> Exception:
    error = RuntimeError(f"HTTP {status}")
    error.response = SimpleNamespace(status_code=status)
    return error


# --- transcript normalization -------------------------------------------------


def test_transcript_keeps_chunks_as_segments() -> None:
    result = SimpleNamespace(
        text="  hello world  ",
        chunks=[
            SimpleNamespace(text=" hello ", timestamp=(0.0, 0.5)),
            SimpleNamespace(text="world", timestamp=(0.5, 1.25)),
        ],
    )

    transcript = _to_transcript(result, "en", True)

    assert transcript.text == "hello world"
    assert transcript.language == "en"
    assert [segment.text for segment in transcript.segments] == ["hello", "world"]
    assert transcript.segments[1].start_seconds == 0.5
    assert transcript.segments[1].end_seconds == 1.25


def test_transcript_drops_chunks_without_usable_timestamps() -> None:
    result = SimpleNamespace(
        text="hello",
        chunks=[
            SimpleNamespace(text="hello", timestamp=(0.0, None)),
            SimpleNamespace(text="", timestamp=(0.0, 1.0)),
            SimpleNamespace(text="kept", timestamp=(1.0, 2.0)),
        ],
    )

    transcript = _to_transcript(result, "en", True)

    assert [segment.text for segment in transcript.segments] == ["kept"]


def test_transcript_omits_segments_when_timestamps_not_requested() -> None:
    result = SimpleNamespace(
        text="hello", chunks=[SimpleNamespace(text="hello", timestamp=(0.0, 1.0))]
    )

    assert _to_transcript(result, "en", False).segments == ()


def test_transcript_without_chunks_is_still_valid() -> None:
    transcript = _to_transcript(SimpleNamespace(text="hello", chunks=None), "en", True)

    assert transcript.text == "hello"
    assert transcript.segments == ()


def test_empty_transcript_is_an_upstream_error() -> None:
    with pytest.raises(ServiceError) as caught:
        _to_transcript(SimpleNamespace(text="   ", chunks=None), "en", True)

    assert caught.value.code == "UPSTREAM_BAD_RESPONSE"
    assert caught.value.status_code == 502


# --- audio decoding -----------------------------------------------------------


def test_decode_audio_round_trips_mono_wav() -> None:
    samples = np.linspace(-0.5, 0.5, 400, dtype=np.float32)

    decoded = _decode_audio(_encoded(samples, 24_000))

    assert decoded.sample_rate == 24_000
    assert decoded.samples.dtype == np.float32
    assert decoded.samples.shape == (400,)
    assert np.allclose(decoded.samples, samples, atol=1e-4)


def test_decode_audio_downmixes_stereo_to_mono() -> None:
    left = np.full(100, 0.5, dtype=np.float32)
    right = np.full(100, -0.1, dtype=np.float32)

    decoded = _decode_audio(_encoded(np.stack([left, right], axis=1), 16_000))

    assert decoded.samples.shape == (100,)
    assert np.allclose(decoded.samples, 0.2, atol=1e-4)


def test_decode_audio_reads_flac() -> None:
    samples = np.zeros(128, dtype=np.float32)

    decoded = _decode_audio(_encoded(samples, 22_050, fmt="FLAC"))

    assert decoded.sample_rate == 22_050
    assert decoded.samples.shape == (128,)


def test_decode_audio_rejects_empty_payload() -> None:
    with pytest.raises(ServiceError) as caught:
        _decode_audio(b"")

    assert caught.value.code == "UPSTREAM_BAD_RESPONSE"


def test_decode_audio_rejects_undecodable_payload() -> None:
    with pytest.raises(ServiceError) as caught:
        _decode_audio(b"not audio at all")

    assert caught.value.code == "UPSTREAM_BAD_RESPONSE"
    assert caught.value.status_code == 502


# --- provider behaviour -------------------------------------------------------


@pytest.mark.anyio
async def test_stt_provider_requests_timestamps_and_normalizes() -> None:
    client = _FakeClient(
        asr=SimpleNamespace(
            text="hello", chunks=[SimpleNamespace(text="hello", timestamp=(0.0, 1.0))]
        )
    )
    provider = _with_client(HfApiSTTProvider(_settings()), client)

    transcript = await provider.transcribe(
        b"audio", language="en", include_timestamps=True
    )

    assert transcript.text == "hello"
    assert client.calls[0]["model"] == "openai/whisper-large-v3"
    assert client.calls[0]["extra_body"] == {"return_timestamps": True}
    assert provider.ready is True
    assert provider.provider_name == "hf-inference-api"


@pytest.mark.anyio
async def test_stt_provider_skips_timestamp_request_when_not_wanted() -> None:
    client = _FakeClient(asr=SimpleNamespace(text="hello", chunks=None))
    provider = _with_client(HfApiSTTProvider(_settings()), client)

    await provider.transcribe(b"audio", language="en", include_timestamps=False)

    assert client.calls[0]["extra_body"] is None


@pytest.mark.anyio
async def test_tts_provider_decodes_hosted_audio() -> None:
    client = _FakeClient(tts=_encoded(np.zeros(320, dtype=np.float32), 24_000))
    provider = _with_client(HfApiTTSProvider(_settings()), client)

    audio = await provider.synthesize("hello")

    assert audio.sample_rate == 24_000
    assert audio.samples.shape == (320,)
    assert client.calls[0]["model"] == "hexgrad/Kokoro-82M"


@pytest.mark.anyio
async def test_missing_token_reports_model_unavailable() -> None:
    provider = HfApiSTTProvider(replace(_settings(), hf_token=None))

    with pytest.raises(ServiceError) as caught:
        await provider.transcribe(b"audio", language="en", include_timestamps=False)

    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert caught.value.status_code == 503
    assert provider.ready is False
    assert "HF_TOKEN" in (provider.load_error or "")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, "MODEL_UNAVAILABLE", False),
        (403, "MODEL_UNAVAILABLE", False),
        (404, "MODEL_UNAVAILABLE", False),
        (429, "MODEL_BUSY", True),
        (500, "UPSTREAM_BAD_RESPONSE", True),
    ],
)
async def test_http_errors_map_to_service_errors(status, code, retryable) -> None:
    client = _FakeClient(error=_http_error(status))
    provider = _with_client(HfApiSTTProvider(_settings()), client)

    with pytest.raises(ServiceError) as caught:
        await provider.transcribe(b"audio", language="en", include_timestamps=False)

    assert caught.value.code == code
    assert caught.value.retryable is retryable


@pytest.mark.anyio
async def test_client_timeout_maps_to_inference_timeout() -> None:
    client = _FakeClient(error=TimeoutError("too slow"))
    provider = _with_client(HfApiTTSProvider(_settings()), client)

    with pytest.raises(ServiceError) as caught:
        await provider.synthesize("hello")

    assert caught.value.code == "INFERENCE_TIMEOUT"
    assert caught.value.status_code == 504


@pytest.mark.anyio
async def test_client_is_built_once_for_concurrent_calls() -> None:
    import asyncio

    client = _FakeClient(asr=SimpleNamespace(text="hello", chunks=None))
    provider = HfApiSTTProvider(_settings(stt_max_concurrency=2))
    builds = 0

    def build():
        nonlocal builds
        builds += 1
        return client

    provider._build_client = build

    await asyncio.gather(
        provider.transcribe(b"a", language="en", include_timestamps=False),
        provider.transcribe(b"b", language="en", include_timestamps=False),
    )

    assert builds == 1
