"""FastAPI routes for the stateless Sign Language Service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .cors import BrowserLocalCORSMiddleware
from .errors import SignServiceError, invalid_request, unauthorized
from .observability import configure_logging
from .service import SignPredictionService


def _request_id(request: Request) -> UUID:
    raw = request.headers.get("X-Request-ID")
    fallback = uuid4()
    if not raw:
        raise invalid_request("X-Request-ID header is required.")
    try:
        value = UUID(raw)
    except ValueError as exc:
        error = invalid_request("X-Request-ID must be a valid UUID.")
        error.request_id = fallback
        raise error from exc
    request.state.request_id = value
    return value


def _error_response(error: SignServiceError, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or error.request_id or uuid4()
    return JSONResponse(status_code=error.status_code, content=error.envelope(request_id))


def _authorize(request: Request, settings: Settings) -> None:
    if settings.internal_api_key and request.headers.get("X-Internal-API-Key") != settings.internal_api_key:
        raise unauthorized()


def create_app(
    settings: Settings | None = None,
    service: SignPredictionService | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_service = service or SignPredictionService(resolved_settings)
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_service.startup()
        yield

    app = FastAPI(
        title="Isyara Sign Language Service API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        BrowserLocalCORSMiddleware,
        allowed_origins=resolved_settings.allowed_origins,
        allow_private_network=resolved_settings.allow_private_network,
    )
    app.state.sign_service = resolved_service
    app.state.sign_settings = resolved_settings

    @app.exception_handler(SignServiceError)
    async def handle_sign_error(request: Request, exc: SignServiceError) -> JSONResponse:
        return _error_response(exc, request)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        error = SignServiceError("INFERENCE_FAILED", "Unexpected service failure.", 500, False)
        return _error_response(error, request)

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return resolved_service.liveness()

    @app.get("/health/ready")
    async def health_ready() -> JSONResponse:
        payload = resolved_service.readiness()
        if payload["status"] == "ready":
            return JSONResponse(status_code=200, content=payload)
        error = resolved_service.readiness_error(uuid4())
        error.details = {"checks": payload.get("checks", {})}
        return JSONResponse(status_code=503, content=error.envelope())

    @app.post("/v1/sign/predict")
    async def predict_sign(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        _authorize(request, resolved_settings)
        content_type = request.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise invalid_request("Content-Type must be multipart/form-data.")
        try:
            form = await request.form()
        except Exception as exc:
            error = invalid_request("Malformed multipart request.")
            error.request_id = request_id
            raise error from exc
        upload = form.get("video")
        if upload is None or not hasattr(upload, "read"):
            error = invalid_request("video is required.")
            error.request_id = request_id
            raise error
        top_k_raw = form.get("topK")
        vocabulary_version = form.get("vocabularyVersion")
        if top_k_raw is None or vocabulary_version is None:
            error = invalid_request("topK and vocabularyVersion are required.")
            error.request_id = request_id
            raise error
        try:
            top_k = int(str(top_k_raw))
        except ValueError as exc:
            error = invalid_request("topK must be an integer.")
            error.request_id = request_id
            raise error from exc
        video = await upload.read(resolved_settings.max_clip_bytes + 1)
        response = await resolved_service.predict(
            video,
            content_type=str(getattr(upload, "content_type", "") or ""),
            top_k=top_k,
            vocabulary_version=str(vocabulary_version),
            request_id=request_id,
        )
        return JSONResponse(status_code=200, content=response)

    return app
