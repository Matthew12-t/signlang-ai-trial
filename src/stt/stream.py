"""Helpers for REST-based rolling STT chunks.

The Application Server owns stream IDs, revisions, overlap de-duplication, and
the decision to persist only committed results. This module intentionally does
not introduce a WebSocket transport.
"""

from __future__ import annotations


def merge_overlapping_text(previous: str, current: str) -> str:
    """Merge two provisional transcripts using their longest word overlap."""

    previous_words = previous.split()
    current_words = current.split()
    max_overlap = min(len(previous_words), len(current_words))
    for size in range(max_overlap, 0, -1):
        left = [word.casefold() for word in previous_words[-size:]]
        right = [word.casefold() for word in current_words[:size]]
        if left == right:
            return " ".join(previous_words + current_words[size:])
    return " ".join(previous_words + current_words).strip()
