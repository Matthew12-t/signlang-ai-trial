from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from src.gloss.api import router as gloss_router
from src.gloss.service import GlossService
from src.shared.config import Settings, get_settings
from src.shared.errors import AppError, app_error_handler, request_validation_error_handler
from src.shared.observability import configure_logging
from src.shared.providers.huggingface import HuggingFaceChatProvider


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    application = FastAPI(title="Isyara AI Services", version="0.1.0")
    application.state.settings = resolved

    provider = None
    if resolved.gloss_mode == "qwen" and resolved.hf_token is not None:
        provider = HuggingFaceChatProvider.from_settings(resolved)
    application.state.gloss_service = GlossService(resolved, provider)

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next: object) -> object:
        supplied_request_id = request.headers.get("X-Request-ID", "").strip()
        request_id = (
            supplied_request_id
            if supplied_request_id and len(supplied_request_id) <= 128
            else str(uuid4())
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    application.add_exception_handler(AppError, app_error_handler)
    application.add_exception_handler(RequestValidationError, request_validation_error_handler)
    application.include_router(gloss_router)

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok", "service": "isyara-ai-services"}

    @application.get("/health/ready")
    async def ready() -> dict[str, object]:
        llm_ready = resolved.hf_token is not None
        gloss_status = "ready"
        status = "ready"
        if resolved.gloss_mode == "qwen" and not llm_ready:
            gloss_status = "llm_unavailable"
            status = "degraded"
        return {
            "status": status,
            "service": "isyara-ai-services",
            "checks": {"configuration": "ok", "gloss": gloss_status},
        }

    return application


app = create_app()
