"""Local validation for model-produced Recall evidence."""

from __future__ import annotations

import unicodedata

from src.shared.models import Evidence, LLMRecallResult, RecallAnswer, TranscriptEntry


_NOT_FOUND = "I could not find that information in the transcript."
_INVALID_EVIDENCE = "I could not verify that answer from the transcript."


def not_found(reason: str, *, invalid_evidence: bool = False) -> RecallAnswer:
    return RecallAnswer(
        answer=_INVALID_EVIDENCE if invalid_evidence else _NOT_FOUND,
        grounded=False,
        evidence=[],
        notFoundReason=reason,
    )


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_recall_evidence(
    result: LLMRecallResult, entries: list[TranscriptEntry]
) -> RecallAnswer:
    if not result.grounded:
        if result.evidence or result.not_found_reason != "NOT_IN_CONTEXT":
            return not_found("EVIDENCE_VALIDATION_FAILED", invalid_evidence=True)
        return RecallAnswer(
            answer=result.answer,
            grounded=False,
            evidence=[],
            notFoundReason="NOT_IN_CONTEXT",
        )

    if not result.evidence or result.not_found_reason is not None:
        return not_found("EVIDENCE_VALIDATION_FAILED", invalid_evidence=True)

    by_id = {entry.id: entry for entry in entries}
    evidence: list[Evidence] = []
    for item in result.evidence:
        entry = by_id.get(item.entry_id)
        if entry is None or _normalize(item.quote) not in _normalize(entry.text):
            return not_found("EVIDENCE_VALIDATION_FAILED", invalid_evidence=True)
        evidence.append(
            Evidence(entryId=entry.id, quote=item.quote, startedAt=entry.started_at)
        )

    return RecallAnswer(
        answer=result.answer,
        grounded=True,
        evidence=evidence,
        notFoundReason=None,
    )

