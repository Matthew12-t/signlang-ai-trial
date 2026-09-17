from src.sign.config import Settings


def test_empty_thresholds_keep_readiness_invalid() -> None:
    errors = Settings().validation_errors()
    assert "SIGN_CONFIDENT_THRESHOLD must be configured" in errors
    assert "SIGN_UNKNOWN_THRESHOLD must be configured" in errors
    assert "SIGN_MIN_TOP1_TOP2_MARGIN must be configured" in errors


def test_allowed_origins_are_parsed_and_deduplicated(monkeypatch) -> None:
    monkeypatch.setenv(
        "SIGN_ALLOWED_ORIGINS",
        " http://localhost:5173,https://demo.example,http://localhost:5173 ",
    )
    monkeypatch.setenv("SIGN_ALLOW_PRIVATE_NETWORK", "true")

    settings = Settings.from_env()

    assert settings.allowed_origins == ("http://localhost:5173", "https://demo.example")
    assert settings.allow_private_network is True


def test_wildcard_origin_is_rejected() -> None:
    settings = Settings(allowed_origins=("*",))

    assert "SIGN_ALLOWED_ORIGINS must not contain '*'" in settings.validation_errors()


def test_private_network_requires_an_origin_allowlist() -> None:
    settings = Settings(allow_private_network=True)

    assert "SIGN_ALLOW_PRIVATE_NETWORK requires SIGN_ALLOWED_ORIGINS" in settings.validation_errors()
