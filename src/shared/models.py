"""Models shared by API, service, and inference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class TranscriptEntry(APIModel):
    id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    source: Literal["speech", "sign"]
    speaker: str | None = Field(default=None, max_length=128)
    text: str = Field(min_length=1, max_length=4000)
    started_at: datetime = Field(alias="startedAt")
    ended_at: datetime = Field(alias="endedAt")

    @model_validator(mode="after")
    def validate_time_order(self) -> "TranscriptEntry":
        if self.ended_at < self.started_at:
            raise ValueError("endedAt must not precede startedAt")
        return self


class RecallOptions(APIModel):
    max_answer_tokens: int = Field(default=160, alias="maxAnswerTokens", ge=32, le=512)
    require_evidence: bool = Field(default=True, alias="requireEvidence")


class RecallQueryRequest(APIModel):
    session_id: str = Field(alias="sessionId", min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=1000)
    language: Literal["en"] = "en"
    context_entries: list[TranscriptEntry] = Field(
        default_factory=list, alias="contextEntries", max_length=500
    )
    options: RecallOptions = Field(default_factory=RecallOptions)


class LLMEvidence(APIModel):
    entry_id: str = Field(alias="entryId", min_length=1)
    quote: str = Field(min_length=1)


NotFoundReason = Literal[
    "NOT_IN_CONTEXT", "EMPTY_CONTEXT", "EVIDENCE_VALIDATION_FAILED"
]


class LLMRecallResult(APIModel):
    answer: str = Field(min_length=1, max_length=2000)
    grounded: bool
    evidence: list[LLMEvidence]
    not_found_reason: NotFoundReason | None = Field(alias="notFoundReason")


class Evidence(APIModel):
    entry_id: str = Field(alias="entryId")
    quote: str
    started_at: datetime = Field(alias="startedAt")


class RecallQueryResponse(APIModel):
    request_id: str = Field(alias="requestId")
    answer: str
    grounded: bool
    evidence: list[Evidence]
    not_found_reason: NotFoundReason | None = Field(alias="notFoundReason")
    model: ModelDescriptor
    latency_ms: int = Field(alias="latencyMs", ge=0)


class ModelDescriptor(BaseModel):
    provider: str
    id: str


class TranscriptionSegment(BaseModel):
    text: str
    start_ms: int = Field(serialization_alias="startMs")
    end_ms: int = Field(serialization_alias="endMs")

    model_config = ConfigDict(populate_by_name=True)


class TranscriptionResponse(BaseModel):
    request_id: str = Field(serialization_alias="requestId")
    text: str
    language: str
    segments: list[TranscriptionSegment]
    model: ModelDescriptor
    latency_ms: int = Field(serialization_alias="latencyMs")

    model_config = ConfigDict(populate_by_name=True)


@dataclass(frozen=True, slots=True)
class ProviderTranscriptSegment:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class ProviderTranscript:
    text: str
    language: str
    segments: tuple[ProviderTranscriptSegment, ...]


@dataclass(frozen=True, slots=True)
class GeneratedAudio:
    samples: Any
    sample_rate: int


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    content: bytes
    media_type: str
    sample_rate: int
    model_id: str
    latency_ms: int
