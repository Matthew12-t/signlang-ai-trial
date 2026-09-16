from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    internal_api_key: SecretStr | None = None
    log_level: str = "INFO"
    request_timeout_seconds: float = 15.0

    llm_backend: Literal["huggingface"] = "huggingface"
    llm_model: str = "Qwen/Qwen3-4B"
    hf_token: SecretStr | None = None
    hf_provider: str = "auto"
    hf_use_structured_output: bool = False

    gloss_mode: Literal["template", "qwen"] = "template"
    gloss_max_tokens: int = 64
    gloss_llm_max_output_tokens: int = 96


@lru_cache
def get_settings() -> Settings:
    return Settings()

