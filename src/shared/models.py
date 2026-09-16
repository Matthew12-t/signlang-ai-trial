from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConfirmedSignToken(APIModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    confirmed_at: datetime = Field(alias="confirmedAt")

    @field_validator("label")
    @classmethod
    def reject_blank_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label must not be blank")
        if not any(character.isalnum() for character in value):
            raise ValueError("label must contain alphanumeric content")
        return value


class ErrorDetail(APIModel):
    code: str
    message: str
    retryable: bool
    request_id: str = Field(alias="requestId")
    details: dict[str, object]


class ErrorEnvelope(APIModel):
    error: ErrorDetail


class GlossNormalizeRequest(APIModel):
    utterance_id: str = Field(alias="utteranceId", min_length=1, max_length=128)
    language: Literal["en"]
    tokens: list[ConfirmedSignToken] = Field(min_length=1)


class GlossNormalizeResponse(APIModel):
    request_id: str = Field(alias="requestId")
    utterance_id: str = Field(alias="utteranceId")
    text: str
    method: Literal["template", "qwen"]
    source_token_ids: list[str] = Field(alias="sourceTokenIds")
    warnings: list[str]
    latency_ms: int = Field(alias="latencyMs", ge=0)


class LLMGlossResult(APIModel):
    text: str = Field(min_length=1, max_length=500)
    source_token_ids: list[str] = Field(alias="sourceTokenIds")


class RuleNormalization(APIModel):
    text: str
    source_token_ids: list[str] = Field(alias="sourceTokenIds")
    warnings: list[str]

