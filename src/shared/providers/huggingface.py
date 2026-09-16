"""Hugging Face provider adapters."""

import asyncio
import json

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
    def from_settings(cls, settings: Settings) -> "HuggingFaceChatProvider":
        if settings.hf_token is None:
            raise ProviderUnavailable("HF token is not configured")

        client = InferenceClient(
            provider=settings.hf_provider,
            api_key=settings.hf_token.get_secret_value(),
            timeout=settings.request_timeout_seconds,
        )
        return cls(
            client=client,
            model=settings.llm_model,
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
            parsed = json.loads(response.choices[0].message.content)
        except (TimeoutError, httpx.TimeoutException, requests.exceptions.Timeout):
            raise ProviderTimeout("Provider request timed out") from None
        except json.JSONDecodeError:
            raise ProviderBadResponse("Provider returned invalid JSON") from None
        except Exception as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code == 429:
                raise ProviderRateLimited("Provider rate limit exceeded") from None
            if isinstance(status_code, int) and 500 <= status_code < 600:
                raise ProviderUnavailable("Provider is unavailable") from None
            raise ProviderBadResponse("Provider returned an invalid response") from None

        if not isinstance(parsed, dict):
            raise ProviderBadResponse("Provider returned a JSON value other than an object")
        return parsed

