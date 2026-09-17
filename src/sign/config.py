"""Environment-backed configuration for Sign Language Service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return None if not value else float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _origins_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(dict.fromkeys(origin.strip() for origin in raw.split(",") if origin.strip()))


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    internal_api_key: str = ""
    allowed_origins: tuple[str, ...] = ()
    allow_private_network: bool = False
    model_backend: str = "mock"
    model_path: Path | None = None
    model_version: str = "signbart-mvp-v1"
    vocabulary_path: Path = Path("config/mvp-en-v1.json")
    vocabulary_version: str = "mvp-en-v1"
    signbart_loader: str = ""
    signbart_repo: Path | None = None
    signbart_config: Path | None = None
    device: str = "cuda:0"
    use_fp16: bool = False
    required_frames: int = 64
    input_width: int = 224
    input_height: int = 224
    input_layout: str = "NCTHW"
    max_clip_bytes: int = 10_000_000
    max_clip_duration_ms: int = 3_000
    max_concurrency: int = 1
    queue_size: int = 2
    inference_timeout_ms: int = 2_000
    confident_threshold: float | None = None
    unknown_threshold: float | None = None
    min_top1_top2_margin: float | None = None
    store_debug_media: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        raw_model_path = _env("SIGN_MODEL_PATH", "")
        return cls(
            host=_env("SIGN_HOST", "127.0.0.1"),
            port=int(_env("SIGN_PORT", "8765")),
            internal_api_key=_env("INTERNAL_API_KEY", ""),
            allowed_origins=_origins_env("SIGN_ALLOWED_ORIGINS"),
            allow_private_network=_bool_env("SIGN_ALLOW_PRIVATE_NETWORK", False),
            model_backend=_env("SIGN_MODEL_BACKEND", "mock").lower(),
            model_path=Path(raw_model_path) if raw_model_path else None,
            model_version=_env("SIGN_MODEL_VERSION", "signbart-mvp-v1"),
            vocabulary_path=Path(_env("SIGN_VOCABULARY_PATH", "config/mvp-en-v1.json")),
            vocabulary_version=_env("SIGN_VOCABULARY_VERSION", "mvp-en-v1"),
            signbart_loader=_env("SIGN_SIGNBART_LOADER", ""),
            signbart_repo=Path(_env("SIGN_SIGNBART_REPO", "")) if _env("SIGN_SIGNBART_REPO", "") else None,
            signbart_config=Path(_env("SIGN_SIGNBART_CONFIG", "")) if _env("SIGN_SIGNBART_CONFIG", "") else None,
            device=_env("SIGN_DEVICE", "cuda:0"),
            use_fp16=_bool_env("SIGN_USE_FP16", False),
            required_frames=int(_env("SIGN_REQUIRED_FRAMES", "64")),
            input_width=int(_env("SIGN_INPUT_WIDTH", "224")),
            input_height=int(_env("SIGN_INPUT_HEIGHT", "224")),
            input_layout=_env("SIGN_INPUT_LAYOUT", "NCTHW").upper(),
            max_clip_bytes=int(_env("SIGN_MAX_CLIP_BYTES", "10000000")),
            max_clip_duration_ms=int(_env("SIGN_MAX_CLIP_DURATION_MS", "3000")),
            max_concurrency=int(_env("SIGN_MAX_CONCURRENCY", "1")),
            queue_size=int(_env("SIGN_QUEUE_SIZE", "2")),
            inference_timeout_ms=int(_env("SIGN_INFERENCE_TIMEOUT_MS", "2000")),
            confident_threshold=_optional_float("SIGN_CONFIDENT_THRESHOLD"),
            unknown_threshold=_optional_float("SIGN_UNKNOWN_THRESHOLD"),
            min_top1_top2_margin=_optional_float("SIGN_MIN_TOP1_TOP2_MARGIN"),
            store_debug_media=_bool_env("SIGN_STORE_DEBUG_MEDIA", False),
        )

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.model_backend not in {"mock", "signbart"}:
            errors.append("SIGN_MODEL_BACKEND must be mock or signbart")
        if "*" in self.allowed_origins:
            errors.append("SIGN_ALLOWED_ORIGINS must not contain '*'")
        if self.allow_private_network and not self.allowed_origins:
            errors.append("SIGN_ALLOW_PRIVATE_NETWORK requires SIGN_ALLOWED_ORIGINS")
        if not self.model_version:
            errors.append("SIGN_MODEL_VERSION must not be empty")
        if not self.vocabulary_version:
            errors.append("SIGN_VOCABULARY_VERSION must not be empty")
        if self.required_frames < 1:
            errors.append("SIGN_REQUIRED_FRAMES must be positive")
        if self.input_width < 1 or self.input_height < 1:
            errors.append("SIGN_INPUT_WIDTH and SIGN_INPUT_HEIGHT must be positive")
        if self.input_layout not in {"NCTHW", "NTCHW"}:
            errors.append("SIGN_INPUT_LAYOUT must be NCTHW or NTCHW")
        if self.max_clip_bytes < 1 or self.max_clip_duration_ms < 1:
            errors.append("video limits must be positive")
        if self.max_concurrency < 1 or self.queue_size < 0:
            errors.append("concurrency settings are invalid")
        if self.inference_timeout_ms < 1:
            errors.append("SIGN_INFERENCE_TIMEOUT_MS must be positive")
        thresholds = {
            "SIGN_CONFIDENT_THRESHOLD": self.confident_threshold,
            "SIGN_UNKNOWN_THRESHOLD": self.unknown_threshold,
            "SIGN_MIN_TOP1_TOP2_MARGIN": self.min_top1_top2_margin,
        }
        for name, value in thresholds.items():
            if value is None:
                errors.append(f"{name} must be configured")
            elif not 0 <= value <= 1:
                errors.append(f"{name} must be between 0 and 1")
        if (
            self.confident_threshold is not None
            and self.unknown_threshold is not None
            and self.confident_threshold < self.unknown_threshold
        ):
            errors.append("SIGN_CONFIDENT_THRESHOLD must be at least SIGN_UNKNOWN_THRESHOLD")
        return errors
