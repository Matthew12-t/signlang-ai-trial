"""Conversation Recall application service."""

from __future__ import annotations

from pydantic import ValidationError

from src.recall.prompt import build_recall_messages
from src.recall.retrieval import ContextRetriever, RecallValidationError
from src.recall.validation import not_found, validate_recall_evidence
from src.shared.errors import AppError
from src.shared.models import LLMRecallResult, RecallAnswer, RecallQueryRequest
from src.shared.providers import (
    ChatProvider,
    ProviderBadResponse,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


class RecallService:
    def __init__(
        self,
        *,
        provider: ChatProvider | None,
        retriever: ContextRetriever,
        model: str,
        max_context_chars: int,
        max_answer_tokens: int,
        temperature: float,
    ) -> None:
        self.provider = provider
        self.retriever = retriever
        self.model = model
        self.max_context_chars = max_context_chars
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature

    async def query(self, request: RecallQueryRequest) -> RecallAnswer:
        try:
            entries = self.retriever.retrieve(
                request.context_entries, max_chars=self.max_context_chars
            )
        except RecallValidationError:
            raise AppError(
                "PAYLOAD_TOO_LARGE", "The supplied context is too large.", 413
            ) from None

        if not entries:
            return not_found("EMPTY_CONTEXT")
        if self.provider is None:
            raise AppError(
                "MODEL_UNAVAILABLE", "The Recall model is unavailable.", 503, True
            )

        try:
            raw = await self.provider.complete_json(
                build_recall_messages(request, entries),
                LLMRecallResult.model_json_schema(),
                max_tokens=min(
                    request.options.max_answer_tokens, self.max_answer_tokens
                ),
                temperature=self.temperature,
            )
            result = LLMRecallResult.model_validate(raw)
        except ProviderRateLimited:
            raise AppError("RATE_LIMITED", "The provider rate limit was reached.", 429, True) from None
        except ProviderTimeout:
            raise AppError("HF_TIMEOUT", "The provider request timed out.", 504, True) from None
        except ProviderUnavailable:
            raise AppError("MODEL_UNAVAILABLE", "The Recall model is unavailable.", 503, True) from None
        except (ProviderBadResponse, ValidationError):
            raise AppError("UPSTREAM_BAD_RESPONSE", "The provider response was invalid.", 502) from None

        return validate_recall_evidence(result, entries)

