import hmac
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from src.gloss.api import router as gloss_router
from src.gloss.service import GlossService
from src.shared.config import Settings, get_settings
from src.shared.errors import (
    AppError,
    app_error_handler,
    error_response,
    http_exception_handler,
    request_validation_error_handler,
)
from src.shared.observability import configure_logging, get_logger
from src.shared.providers.huggingface import HuggingFaceChatProvider


logger = get_logger(__name__)


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
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied_request_id = request.headers.get("X-Request-ID", "").strip()
        request_id = (
            supplied_request_id
            if supplied_request_id and len(supplied_request_id) <= 128
            else str(uuid4())
        )
        request.state.request_id = request_id

        configured_key = resolved.internal_api_key
        is_gloss_normalize = (
            request.method == "POST"
            and _canonical_request_path(request) == "/v1/gloss/normalize"
        )
        if configured_key is not None and is_gloss_normalize:
            expected = configured_key.get_secret_value().encode("utf-8")
            provided = request.headers.get("X-Internal-API-Key", "").encode("utf-8")
            if not hmac.compare_digest(expected, provided):
                logger.warning(
                    "gloss_auth request_id=%s mode=%s status=401",
                    request_id,
                    resolved.gloss_mode,
                )
                response = error_response(
                    request,
                    AppError("UNAUTHORIZED", "Authentication is required.", 401),
                )
                response.headers["X-Request-ID"] = request_id
                return response

        try:
            response = await call_next(request)
        except Exception:
            logger.error(
                "request_failed request_id=%s path=%s status=500",
                request_id,
                request.url.path,
            )
            response = error_response(
                request,
                AppError("INTERNAL_ERROR", "An internal error occurred.", 500),
            )
        response.headers["X-Request-ID"] = request_id
        return response

    application.add_exception_handler(AppError, app_error_handler)
    application.add_exception_handler(RequestValidationError, request_validation_error_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
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
