from datetime import UTC, datetime, timedelta

import pytest

from src.recall.retrieval import ProvidedContextRetriever, RecallValidationError
from src.shared.models import TranscriptEntry


def entry(entry_id: str, sequence: int, text: str) -> TranscriptEntry:
    started = datetime(2026, 9, 17, 10, 0, tzinfo=UTC) + timedelta(seconds=sequence)
    return TranscriptEntry(
        id=entry_id,
        sequence=sequence,
        source="speech",
        speaker="Participant",
        text=text,
        startedAt=started,
        endedAt=started + timedelta(seconds=1),
    )


def test_retriever_orders_entries_by_sequence() -> None:
    result = ProvidedContextRetriever().retrieve(
        [entry("two", 2, "Second"), entry("one", 1, "First")],
        max_chars=100,
    )

    assert [item.id for item in result] == ["one", "two"]


def test_retriever_removes_oldest_whole_entries_until_text_fits() -> None:
    result = ProvidedContextRetriever().retrieve(
        [entry("one", 1, "12345"), entry("two", 2, "67890")],
        max_chars=5,
    )

    assert [(item.id, item.text) for item in result] == [("two", "67890")]


def test_retriever_rejects_single_entry_larger_than_limit() -> None:
    with pytest.raises(RecallValidationError, match="PAYLOAD_TOO_LARGE"):
        ProvidedContextRetriever().retrieve(
            [entry("one", 1, "123456")],
            max_chars=5,
        )


def test_retriever_keeps_empty_context_empty() -> None:
    assert ProvidedContextRetriever().retrieve([], max_chars=5) == []
