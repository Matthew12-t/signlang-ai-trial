"""Internal typed boundaries for video, model, and prediction data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Frame:
    data: Any
    timestamp_ms: float


@dataclass(frozen=True)
class DecodedVideo:
    frames: tuple[Frame, ...]
    duration_ms: float


@dataclass(frozen=True)
class ModelInputConfig:
    required_frames: int
    width: int
    height: int
    layout: str
    representation: str = "rgb"
    keypoint_count: int = 75


@dataclass(frozen=True)
class PreprocessedVideo:
    frames: tuple[Any, ...]
    duration_ms: float
    input_config: ModelInputConfig
    no_sign: bool = False


@dataclass(frozen=True)
class InputQuality:
    no_sign: bool = False


@dataclass(frozen=True)
class ModelPrediction:
    logits: tuple[float, ...]
    quality: InputQuality = InputQuality()


@dataclass(frozen=True)
class Candidate:
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence}


@dataclass(frozen=True)
class VocabularyLabel:
    index: int
    label: str
    display_text: str
    enabled: bool = True


@dataclass(frozen=True)
class Vocabulary:
    version: str
    model_version: str
    labels: tuple[VocabularyLabel, ...]

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = tuple(
            VocabularyLabel(
                index=int(item["index"]),
                label=str(item["label"]),
                display_text=str(item.get("displayText", item["label"])),
                enabled=bool(item.get("enabled", True)),
            )
            for item in payload["labels"]
        )
        vocabulary = cls(
            version=str(payload["version"]),
            model_version=str(payload["modelVersion"]),
            labels=labels,
        )
        vocabulary.validate()
        return vocabulary

    def validate(self) -> None:
        indices = [item.index for item in self.labels]
        if indices != list(range(len(indices))):
            raise ValueError("vocabulary indices must be unique and contiguous from zero")
        if not self.labels or any(not item.label for item in self.labels):
            raise ValueError("vocabulary must contain non-empty labels")
        if not any(item.enabled for item in self.labels):
            raise ValueError("vocabulary must enable at least one label")

    def labels_by_index(self) -> dict[int, VocabularyLabel]:
        return {item.index: item for item in self.labels}

    def enabled_labels_by_index(self) -> dict[int, VocabularyLabel]:
        return {item.index: item for item in self.labels if item.enabled}
