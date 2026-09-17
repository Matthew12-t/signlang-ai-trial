"""Conversation Recall prompt construction."""

from __future__ import annotations

import json

from src.shared.models import ChatMessage, LLMRecallResult, RecallQueryRequest, TranscriptEntry


def build_recall_messages(
    request: RecallQueryRequest, entries: list[TranscriptEntry]
) -> list[ChatMessage]:
    schema = json.dumps(LLMRecallResult.model_json_schema(), separators=(",", ":"))
    system = (
        "Answer only from the supplied transcript entries. Do not use outside knowledge. "
        "If the answer is unsupported, return grounded=false, evidence=[], and "
        "notFoundReason=NOT_IN_CONTEXT. A grounded answer must cite existing entryId "
        "values and quote text from those entries. Keep the answer concise and in English. "
        f"Return only JSON matching this response JSON schema: {schema}"
    )
    payload = {
        "question": request.query,
        "language": request.language,
        "options": request.options.model_dump(by_alias=True),
        "transcriptEntries": [
            entry.model_dump(by_alias=True, mode="json") for entry in entries
        ],
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
    ]

