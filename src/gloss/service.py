"""Gloss normalization policy and hosted-provider fallback service."""

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from src.gloss.rules import normalize_with_rules
from src.shared.config import Settings
from src.shared.models import ChatMessage, GlossNormalizeRequest, LLMGlossResult, RuleNormalization
from src.shared.providers import (
    ChatProvider,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
)


SYSTEM_INSTRUCTION = (
    "Convert only the supplied confirmed sign-token labels into one concise English sentence. "
    "Do not add facts, people, objects, times, or intent absent from the labels. "
    "Return only data matching the requested schema and repeat every sourceTokenId exactly "
    "once in the original order."
)

_HIGH_CONFIDENCE_NON_ENGLISH_PHRASES = (
    "terima kasih",
    "muchas gracias",
    "buenos dias",
    "bonjour",
    "merci beaucoup",
    "danke schon",
)
_COMMON_ENGLISH_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "do",
    "hello",
    "i",
    "is",
    "it",
    "me",
    "my",
    "no",
    "not",
    "please",
    "repeat",
    "that",
    "thank",
    "thanks",
    "the",
    "this",
    "understand",
    "we",
    "yes",
    "you",
    "your",
}
_SAFE_OUTPUT_CHARACTERS = re.compile(r"^[A-Za-z0-9 .,!?;:'\"()&+/\-]+$")


@dataclass(frozen=True, slots=True)
class GlossServiceResult:
    text: str
    method: str
    source_token_ids: list[str]
    warnings: list[str]


class GlossValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class GlossService:
    def __init__(self, settings: Settings, provider: ChatProvider | None = None) -> None:
        self._settings = settings
        self._provider = provider

    async def normalize(self, request: GlossNormalizeRequest) -> GlossServiceResult:
        if request.language != "en":
            raise GlossValidationError("UNSUPPORTED_LANGUAGE")
        if len(request.tokens) > self._settings.gloss_max_tokens:
            raise GlossValidationError("PAYLOAD_TOO_LARGE")

        template = normalize_with_rules(request.tokens)
        if self._settings.gloss_mode == "template":
            return self._template_result(template)
        if self._provider is None:
            return self._fallback(template, "LLM_FALLBACK_PROVIDER_UNAVAILABLE")

        try:
            response_schema = LLMGlossResult.model_json_schema(by_alias=True)
            provider_data = await self._provider.complete_json(
                self._messages(request, response_schema, template.text),
                response_schema,
                max_tokens=self._settings.gloss_llm_max_output_tokens,
                temperature=0,
            )
        except ProviderTimeout:
            return self._fallback(template, "LLM_FALLBACK_PROVIDER_TIMEOUT")
        except ProviderRateLimited:
            return self._fallback(template, "LLM_FALLBACK_RATE_LIMITED")
        except ProviderError:
            return self._fallback(template, "LLM_FALLBACK_PROVIDER_UNAVAILABLE")

        try:
            llm_result = LLMGlossResult.model_validate(provider_data)
        except ValidationError:
            return self._fallback(template, "LLM_FALLBACK_INVALID_RESPONSE")
        if llm_result.source_token_ids != [token.id for token in request.tokens]:
            return self._fallback(template, "LLM_FALLBACK_INVALID_PROVENANCE")
        if not is_plausible_english_output(
            llm_result.text,
            [token.label for token in request.tokens],
        ):
            return self._fallback(template, "LLM_FALLBACK_INVALID_RESPONSE")

        return GlossServiceResult(
            text=llm_result.text,
            method="qwen",
            source_token_ids=llm_result.source_token_ids,
            warnings=[],
        )

    def _messages(
        self,
        request: GlossNormalizeRequest,
        response_schema: dict[str, object],
        example_text: str,
    ) -> list[ChatMessage]:
        tokens = [{"id": token.id, "label": token.label} for token in request.tokens]
        example = {
            "text": example_text,
            "sourceTokenIds": [token.id for token in request.tokens],
        }
        schema_json = json.dumps(response_schema, separators=(",", ":"))
        example_json = json.dumps(example, separators=(",", ":"))
        return [
            ChatMessage(
                role="system",
                content=(
                    f"{SYSTEM_INSTRUCTION}\n"
                    f"Response JSON Schema: {schema_json}\n"
                    f"Valid response example: {example_json}"
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps({"tokens": tokens}, separators=(",", ":")),
            ),
        ]

    @staticmethod
    def _template_result(template: RuleNormalization) -> GlossServiceResult:
        return GlossServiceResult(
            text=template.text,
            method="template",
            source_token_ids=template.source_token_ids,
            warnings=template.warnings,
        )

    def _fallback(self, template: RuleNormalization, warning: str) -> GlossServiceResult:
        result = self._template_result(template)
        return GlossServiceResult(
            text=result.text,
            method=result.method,
            source_token_ids=result.source_token_ids,
            warnings=[*result.warnings, warning],
        )


def is_plausible_english_output(text: str, source_labels: list[str]) -> bool:
    """Conservatively validate hosted output before it can reach callers.

    This intentionally favors deterministic fallback over accepting uncertain output.
    It is a small safety heuristic, not a general-purpose language detector.
    """
    if (
        not text
        or not text.isascii()
        or not text.isprintable()
        or _SAFE_OUTPUT_CHARACTERS.fullmatch(text) is None
    ):
        return False

    lowered = text.casefold()
    if any(phrase in lowered for phrase in _HIGH_CONFIDENCE_NON_ENGLISH_PHRASES):
        return False

    output_words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", lowered))
    if not output_words:
        return False
    if output_words & _COMMON_ENGLISH_WORDS:
        return True

    source_words = {
        word
        for label in source_labels
        for word in re.findall(r"[a-z]+", label.casefold())
    }
    return bool(output_words & source_words)

