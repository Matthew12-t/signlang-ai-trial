"""FastAPI entrypoint for Isyara AI Services."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, Header, Request

from src.gloss.api import router as gloss_router
from src.gloss.service import GlossService
from src.shared.config import Settings, get_settings
from src.shared.errors import AppError, error_response, register_error_handlers
from src.shared.observability import configure_logging, install_request_middleware
from src.shared.providers.faster_whisper import FasterWhisperProvider
from src.shared.providers.huggingface import HuggingFaceChatProvider
from src.shared.providers.voxcpm import VoxCPMProvider
from src.stt.api import router as stt_router
from src.stt.service import STTService
from src.tts.api import router as tts_router
from src.tts.service import TTSService

logger = logging.getLogger(__name__)

_health_responses = {
    200: {
        "headers": {
            "X-Request-ID": {
                "description": "Caller-supplied request ID or a generated UUID.",
                "schema": {"type": "string"},
            }
        }
    }
}


def _canonical_request_path(request: Request) -> str:
    path = request.scope["path"]
    root_path = request.scope.get("root_path", "").rstrip("/")
    if not root_path:
        return path
    if path == root_path:
        return "/"
    if path.startswith(f"{root_path}/"):
        return path[len(root_path) :]
    return path


def create_app(
    settings: Settings | None = None,
    *,
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
        version="0.3.0",
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.stt_service = stt_service
    application.state.tts_service = tts_service

    gloss_provider = None
    if runtime_settings.gloss_mode == "qwen" and runtime_settings.hf_token is not None:
        gloss_provider = HuggingFaceChatProvider.from_settings(runtime_settings)
    application.state.gloss_service = GlossService(runtime_settings, gloss_provider)

    register_error_handlers(application)

    @application.middleware("http")
    async def authenticate_gloss(request: Request, call_next):
        configured_key = runtime_settings.internal_api_key
        is_gloss_normalize = (
            request.method == "POST"
            and _canonical_request_path(request) == "/v1/gloss/normalize"
        )
        if configured_key is not None and is_gloss_normalize:
            expected = configured_key.encode("utf-8")
            provided = request.headers.get("X-Internal-API-Key", "").encode("utf-8")
            if not hmac.compare_digest(expected, provided):
                return error_response(
                    request,
                    AppError("UNAUTHORIZED", "Authentication is required.", 401),
                )
        return await call_next(request)

    install_request_middleware(application)
    application.include_router(stt_router)
    application.include_router(tts_router)
    application.include_router(gloss_router)

    @application.get("/health/live", tags=["Health"], responses=_health_responses)
    async def live(
        _request_id: Annotated[
            str | None,
            Header(alias="X-Request-ID", description="Optional request ID."),
        ] = None,
    ) -> dict[str, str]:
        return {"status": "ok", "service": "isyara-ai-services"}

    @application.get("/health/ready", tags=["Health"], responses=_health_responses)
    async def ready(
        _request_id: Annotated[
            str | None,
            Header(alias="X-Request-ID", description="Optional request ID."),
        ] = None,
    ) -> dict[str, object]:
        checks: dict[str, str] = {}
        for service_name in ("stt", "tts"):
            if service_name not in runtime_settings.enabled_services:
                continue
            service = getattr(application.state, f"{service_name}_service", None)
            checks[service_name] = (
                "ready" if service is not None and service.ready else "unavailable"
            )
        if "gloss" in runtime_settings.enabled_services:
            checks["configuration"] = "ok"
            checks["gloss"] = (
                "llm_unavailable"
                if runtime_settings.gloss_mode == "qwen"
                and runtime_settings.hf_token is None
                else "ready"
            )
        all_ready = all(
            value in {"ready", "ok"} for value in checks.values()
        )
        return {
            "status": "ready" if all_ready else "degraded",
            "service": (
                "isyara-ai-services"
                if "gloss" in runtime_settings.enabled_services
                else "ai-services"
            ),
            "checks": checks,
        }

    return application


app = create_app()
