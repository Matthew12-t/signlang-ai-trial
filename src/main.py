"""FastAPI entrypoint for Isyara AI Services."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.shared.config import Settings, get_settings
from src.shared.errors import register_error_handlers
from src.shared.observability import configure_logging, install_request_middleware
from src.shared.providers.faster_whisper import FasterWhisperProvider
from src.shared.providers.voxcpm import VoxCPMProvider
from src.stt.api import router as stt_router
from src.stt.service import STTService
from src.tts.api import router as tts_router
from src.tts.service import TTSService

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    stt_service: STTService | None = None,
    tts_service: TTSService | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if application.state.stt_service is None and "stt" in runtime_settings.enabled_services:
            stt_provider = FasterWhisperProvider(runtime_settings)
            application.state.stt_service = STTService(stt_provider, runtime_settings)
            if runtime_settings.preload_models:
                try:
                    await stt_provider.load()
                except Exception:
                    logger.exception("STT started in degraded mode")

        if application.state.tts_service is None and "tts" in runtime_settings.enabled_services:
            tts_provider = VoxCPMProvider(runtime_settings)
            application.state.tts_service = TTSService(tts_provider, runtime_settings)
            if runtime_settings.preload_models:
                try:
                    await tts_provider.load()
                except Exception:
                    logger.exception("TTS started in degraded mode")
        yield

    application = FastAPI(
        title="Isyara AI Services",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.stt_service = stt_service
    application.state.tts_service = tts_service
    register_error_handlers(application)
    install_request_middleware(application)
    application.include_router(stt_router)
    application.include_router(tts_router)

    @application.get("/health/live", tags=["Health"])
    async def live() -> dict[str, str]:
        return {"status": "live", "service": "ai-services"}

    @application.get("/health/ready", tags=["Health"])
    async def ready() -> dict[str, object]:
        checks: dict[str, str] = {}
        for service_name in ("stt", "tts"):
            if service_name not in runtime_settings.enabled_services:
                checks[service_name] = "disabled"
                continue
            service = getattr(application.state, f"{service_name}_service", None)
            checks[service_name] = (
                "ready" if service is not None and service.ready else "unavailable"
            )
        all_ready = all(
            value in {"ready", "disabled"} for value in checks.values()
        )
        return {
            "status": "ready" if all_ready else "degraded",
            "service": "ai-services",
            "checks": checks,
        }

    return application


app = create_app()
