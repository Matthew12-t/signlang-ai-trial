"""Gloss normalization policy and hosted-provider fallback service."""

import json
from dataclasses import dataclass

from pydantic import ValidationError

from src.gloss.rules import canonicalize_label, normalize_with_rules
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
            provider_data = await self._provider.complete_json(
                self._messages(request),
                LLMGlossResult.model_json_schema(by_alias=True),
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

        return GlossServiceResult(
            text=llm_result.text,
            method="qwen",
            source_token_ids=llm_result.source_token_ids,
            warnings=[],
        )

    def _messages(self, request: GlossNormalizeRequest) -> list[ChatMessage]:
        tokens = [
            {"id": token.id, "label": canonicalize_label(token.label)}
            for token in request.tokens
        ]
        return [
            ChatMessage(role="system", content=SYSTEM_INSTRUCTION),
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

