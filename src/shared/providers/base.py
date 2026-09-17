"""Provider-neutral interfaces for chat completion adapters."""

from typing import Protocol

from src.shared.models import ChatMessage


class ProviderError(Exception):
    """Base exception for provider failures exposed to application services."""


class ProviderTimeout(ProviderError):
    """The provider did not respond before the configured timeout."""


class ProviderRateLimited(ProviderError):
    """The provider rejected the request because of rate limiting."""


class ProviderUnavailable(ProviderError):
    """The provider is unavailable or is not configured."""


class ProviderBadResponse(ProviderError):
    """The provider returned an invalid response or an unclassified error."""


class ChatProvider(Protocol):
    async def complete_json(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, object],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        raise NotImplementedError
