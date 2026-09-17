import pytest

from src.shared.config import Settings


def test_required_internal_api_key_must_be_configured(monkeypatch) -> None:
    monkeypatch.setenv("REQUIRE_INTERNAL_API_KEY", "true")
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="INTERNAL_API_KEY must be set"):
        Settings.from_env()
