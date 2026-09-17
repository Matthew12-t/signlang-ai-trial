"""Shared hosted-chat interfaces; local STT/TTS adapters live beside them."""

from src.shared.providers.base import (
    ChatProvider,
    ProviderBadResponse,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)

__all__ = [
    "ChatProvider",
    "ProviderBadResponse",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderTimeout",
    "ProviderUnavailable",
]
