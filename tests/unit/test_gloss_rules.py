import pytest
from pydantic import ValidationError

from src.gloss.rules import normalize_with_rules
from src.shared.models import ConfirmedSignToken, GlossNormalizeRequest


def token(token_id: str, label: str) -> ConfirmedSignToken:
    return ConfirmedSignToken(id=token_id, label=label, confirmedAt="2026-09-17T10:00:00Z")


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["I", "NOT_UNDERSTAND"], "I do not understand."),
        (["YOU", "REPEAT", "PLEASE"], "Could you repeat that, please?"),
        (["REPEAT", "PLEASE"], "Please repeat that."),
        (["THANK_YOU"], "Thank you."),
        (["HELLO"], "Hello."),
        (["YES"], "Yes."),
        (["NO"], "No."),
    ],
)
def test_exact_templates(labels: list[str], expected: str) -> None:
    result = normalize_with_rules([token(str(index), label) for index, label in enumerate(labels)])
    assert result.text == expected
    assert result.warnings == []


def test_labels_are_canonicalized_before_matching() -> None:
    result = normalize_with_rules([token("1", "i"), token("2", "not-understand")])
    assert result.text == "I do not understand."


def test_unmatched_input_is_preserved_and_warned() -> None:
    result = normalize_with_rules([token("1", "NEED_HELP"), token("2", "NOW")])
    assert result.text == "Need help now."
    assert result.warnings == ["UNMATCHED_TEMPLATE"]


def test_unmatched_labels_preserve_meaningful_punctuation_and_unicode() -> None:
    result = normalize_with_rules([token("1", "C++"), token("2", "Café")])

    assert result.text == "C++ café."
    assert result.source_token_ids == ["1", "2"]
    assert result.warnings == ["UNMATCHED_TEMPLATE"]


def test_meaningful_punctuation_cannot_accidentally_select_an_exact_template() -> None:
    result = normalize_with_rules([token("1", "NO++")])

    assert result.text == "No++."
    assert result.source_token_ids == ["1"]
    assert result.warnings == ["UNMATCHED_TEMPLATE"]


@pytest.mark.parametrize("label", ["___", "???", "--", "  _?  "])
def test_token_label_requires_usable_alphanumeric_content(label: str) -> None:
    with pytest.raises(ValidationError):
        token("1", label)


def test_request_rejects_unsupported_language() -> None:
    with pytest.raises(ValidationError):
        GlossNormalizeRequest(
            utteranceId="u1",
            language="id",
            tokens=[token("1", "HELLO")],
        )
