from fastapi import FastAPI

from src.shared.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    application = FastAPI(title="Isyara AI Services", version="0.1.0")
    application.state.settings = resolved

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
