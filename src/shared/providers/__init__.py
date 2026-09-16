"""Provider-neutral chat interfaces and provider errors."""

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
