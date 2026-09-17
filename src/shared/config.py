"""Environment-backed configuration for the AI service process."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal


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

    internal_api_key: str | None = None
    log_level: str = "INFO"
    request_timeout_seconds: float = 15.0
    model_queue_timeout_seconds: float = 2.0
    enabled_services: tuple[str, ...] = ("gloss",)
    preload_models: bool = True

    hf_token: str | None = None
    model_cache_dir: str | None = None

    stt_model: str = "Systran/faster-whisper-large-v3"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_beam_size: int = 5
    stt_allowed_languages: tuple[str, ...] = ("en",)
    stt_vad_filter: bool = True
    stt_min_silence_ms: int = 500
    stt_condition_on_previous_text: bool = False
    stt_max_audio_bytes: int = 10_000_000
    stt_max_concurrency: int = 1

    tts_model: str = "openbmb/VoxCPM2"
    tts_device: str = "cuda"
    tts_optimize: bool = True
    tts_load_denoiser: bool = False
    tts_normalize: bool = True
    tts_cfg_value: float = 2.0
    tts_inference_timesteps: int = 10
    tts_seed: int = 42
    tts_allowed_languages: tuple[str, ...] = ("en", "id")
    tts_max_text_chars: int = 500
    tts_chunk_chars: int = 200
    tts_pause_ms: int = 120
    tts_max_concurrency: int = 1

    llm_backend: Literal["huggingface"] = "huggingface"
    llm_model: str = "Qwen/Qwen3-4B"
    hf_provider: str = "auto"
    hf_use_structured_output: bool = False
    gloss_mode: Literal["template", "qwen"] = "template"
    gloss_max_tokens: int = 64
    gloss_llm_max_output_tokens: int = 96
    recall_model: str = "Qwen/Qwen3-8B"
    recall_max_context_chars: int = 24_000
    recall_max_answer_tokens: int = 160
    recall_temperature: float = 0.0

    # Compatibility with the former BaseSettings constructor used by offline tests.
    _env_file: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.hf_token is not None and not self.hf_token.strip():
            object.__setattr__(self, "hf_token", None)
        if self.internal_api_key is not None and not self.internal_api_key.strip():
            object.__setattr__(self, "internal_api_key", None)

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
            llm_backend="huggingface",
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen3-4B"),
            hf_provider=os.getenv("HF_PROVIDER", "auto"),
            hf_use_structured_output=_boolean("HF_USE_STRUCTURED_OUTPUT", False),
            gloss_mode=os.getenv("GLOSS_MODE", "template"),
            gloss_max_tokens=_integer("GLOSS_MAX_TOKENS", 64),
            gloss_llm_max_output_tokens=_integer(
                "GLOSS_LLM_MAX_OUTPUT_TOKENS", 96
            ),
            recall_model=os.getenv("HF_RECALL_MODEL", "Qwen/Qwen3-8B"),
            recall_max_context_chars=_integer("RECALL_MAX_CONTEXT_CHARS", 24_000),
            recall_max_answer_tokens=_integer("RECALL_MAX_ANSWER_TOKENS", 160),
            recall_temperature=_floating_point("RECALL_TEMPERATURE", 0.0),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
