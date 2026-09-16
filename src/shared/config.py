"""Environment-backed configuration for the AI service process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _floating_point(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _csv(name: str, default: str) -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings.

    Model inference is local. ``HF_TOKEN`` is only used by the Hugging Face Hub
    clients while downloading model artifacts.
    """

    internal_api_key: str | None
    log_level: str
    request_timeout_seconds: float
    model_queue_timeout_seconds: float
    enabled_services: tuple[str, ...]
    preload_models: bool

    hf_token: str | None
    model_cache_dir: str | None

    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_beam_size: int
    stt_allowed_languages: tuple[str, ...]
    stt_vad_filter: bool
    stt_min_silence_ms: int
    stt_condition_on_previous_text: bool
    stt_max_audio_bytes: int
    stt_max_concurrency: int

    tts_model: str
    tts_device: str
    tts_optimize: bool
    tts_load_denoiser: bool
    tts_normalize: bool
    tts_cfg_value: float
    tts_inference_timesteps: int
    tts_seed: int
    tts_allowed_languages: tuple[str, ...]
    tts_max_text_chars: int
    tts_chunk_chars: int
    tts_pause_ms: int
    tts_max_concurrency: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            internal_api_key=os.getenv("INTERNAL_API_KEY") or None,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            request_timeout_seconds=_floating_point("REQUEST_TIMEOUT_SECONDS", 15.0),
            model_queue_timeout_seconds=_floating_point(
                "MODEL_QUEUE_TIMEOUT_SECONDS", 2.0
            ),
            enabled_services=_csv("ENABLED_SERVICES", "stt,tts"),
            preload_models=_boolean("PRELOAD_MODELS", True),
            hf_token=os.getenv("HF_TOKEN") or None,
            model_cache_dir=os.getenv("MODEL_CACHE_DIR") or None,
            stt_model=os.getenv(
                "HF_STT_MODEL", "Systran/faster-whisper-large-v3"
            ),
            stt_device=os.getenv("STT_DEVICE", "cuda"),
            stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "float16"),
            stt_beam_size=_integer("STT_BEAM_SIZE", 5),
            stt_allowed_languages=_csv("STT_ALLOWED_LANGUAGES", "en"),
            stt_vad_filter=_boolean("STT_VAD_FILTER", True),
            stt_min_silence_ms=_integer("STT_MIN_SILENCE_MS", 500),
            stt_condition_on_previous_text=_boolean(
                "STT_CONDITION_ON_PREVIOUS_TEXT", False
            ),
            stt_max_audio_bytes=_integer("STT_MAX_AUDIO_BYTES", 10_000_000),
            stt_max_concurrency=max(1, _integer("STT_MAX_CONCURRENCY", 1)),
            tts_model=os.getenv("HF_TTS_MODEL", "openbmb/VoxCPM2"),
            tts_device=os.getenv("TTS_DEVICE", "cuda"),
            tts_optimize=_boolean("TTS_OPTIMIZE", True),
            tts_load_denoiser=_boolean("TTS_LOAD_DENOISER", False),
            tts_normalize=_boolean("TTS_NORMALIZE", True),
            tts_cfg_value=_floating_point("TTS_CFG_VALUE", 2.0),
            tts_inference_timesteps=_integer("TTS_INFERENCE_TIMESTEPS", 10),
            tts_seed=_integer("TTS_SEED", 42),
            tts_allowed_languages=_csv("TTS_ALLOWED_LANGUAGES", "en,id"),
            tts_max_text_chars=_integer("TTS_MAX_TEXT_CHARS", 500),
            tts_chunk_chars=_integer("TTS_CHUNK_CHARS", 200),
            tts_pause_ms=_integer("TTS_PAUSE_MS", 120),
            tts_max_concurrency=max(1, _integer("TTS_MAX_CONCURRENCY", 1)),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
