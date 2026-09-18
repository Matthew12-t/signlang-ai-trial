"""Hugging Face provider adapters."""

import asyncio
import json
import logging

import httpx
import requests
from huggingface_hub import InferenceClient

from src.shared.config import Settings
from src.shared.models import ChatMessage
from src.shared.providers.base import (
    ProviderBadResponse,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)

logger = logging.getLogger(__name__)


class HuggingFaceChatProvider:
    def __init__(
        self,
        *,
        client: object,
        model: str,
        use_structured_output: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._use_structured_output = use_structured_output

    @classmethod
    def from_settings(
        cls, settings: Settings, *, model: str | None = None
    ) -> "HuggingFaceChatProvider":
        if settings.hf_token is None:
            raise ProviderUnavailable("HF token is not configured")

        client = InferenceClient(
            provider=settings.hf_provider,
            api_key=settings.hf_token,
            timeout=settings.request_timeout_seconds,
        )
        return cls(
            client=client,
            model=model or settings.llm_model,
            use_structured_output=settings.hf_use_structured_output,
        )

    async def complete_json(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, object],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._use_structured_output:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "isyara_response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        try:
            response = await asyncio.to_thread(self._client.chat_completion, **request)
        except (TimeoutError, httpx.TimeoutException, requests.exceptions.Timeout):
            logger.warning("Chat provider timed out model=%s", self._model)
            raise ProviderTimeout("Provider request timed out") from None
        except Exception as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            # Only the exception type and status are logged. Provider errors can
            # carry the request URL, which embeds the API key.
            logger.warning(
                "Chat provider call failed model=%s error=%s status=%s",
                self._model,
                type(error).__name__,
                status_code,
            )
            if status_code == 429:
                raise ProviderRateLimited("Provider rate limit exceeded") from None
            if status_code == 402:
                raise ProviderUnavailable("Provider credits are exhausted") from None
            if isinstance(status_code, int) and 500 <= status_code < 600:
                raise ProviderUnavailable("Provider is unavailable") from None
            raise ProviderBadResponse("Provider returned an invalid response") from None

        content = self._content_of(response, max_tokens=max_tokens)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Chat provider returned non-JSON content model=%s", self._model)
            raise ProviderBadResponse("Provider returned invalid JSON") from None

        if not isinstance(parsed, dict):
            raise ProviderBadResponse(
                "Provider returned a JSON value other than an object"
            ) from None
        return parsed

    def _content_of(self, response: object, *, max_tokens: int) -> str:
        """Pull the message text out of a completion, naming why it is missing.

        A reasoning model that spends its whole budget before answering returns
        ``content=None`` with ``finish_reason="length"``. Parsing that as JSON
        raises ``TypeError``, which used to surface as the same opaque bad
        response as genuinely malformed output.
        """

        choices = getattr(response, "choices", None) or ()
        if not choices:
            logger.warning("Chat provider returned no choices model=%s", self._model)
            raise ProviderBadResponse(
                "Provider returned no completion choices"
            ) from None

        choice = choices[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        if content is not None:
            return content

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            logger.warning(
                "Chat provider hit the token limit before answering "
                "model=%s max_tokens=%s",
                self._model,
                max_tokens,
            )
            raise ProviderBadResponse(
                "Provider stopped at the token limit before returning content"
            ) from None

        logger.warning(
            "Chat provider returned empty content model=%s finish_reason=%s",
            self._model,
            finish_reason,
        )
        raise ProviderBadResponse("Provider returned an empty response") from None

