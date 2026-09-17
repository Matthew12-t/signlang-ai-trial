"""Context selection boundary for Conversation Recall."""

from __future__ import annotations

from typing import Protocol

from src.shared.models import TranscriptEntry


class RecallValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ContextRetriever(Protocol):
    def retrieve(
        self, entries: list[TranscriptEntry], *, max_chars: int
    ) -> list[TranscriptEntry]: ...


class ProvidedContextRetriever:
    def retrieve(
        self, entries: list[TranscriptEntry], *, max_chars: int
    ) -> list[TranscriptEntry]:
        ordered = sorted(entries, key=lambda item: item.sequence)
        if any(len(item.text) > max_chars for item in ordered):
            raise RecallValidationError("PAYLOAD_TOO_LARGE")

        selected = list(ordered)
        total = sum(len(item.text) for item in selected)
        while selected and total > max_chars:
            total -= len(selected.pop(0).text)
        return selected
