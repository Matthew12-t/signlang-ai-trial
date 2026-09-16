"""Models shared by API, service, and inference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
