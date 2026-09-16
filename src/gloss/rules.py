import re

from src.shared.models import ConfirmedSignToken, RuleNormalization


EXACT_TEMPLATES: dict[tuple[str, ...], str] = {
    ("I", "NOT_UNDERSTAND"): "I do not understand.",
    ("YOU", "REPEAT", "PLEASE"): "Could you repeat that, please?",
    ("REPEAT", "PLEASE"): "Please repeat that.",
    ("THANK_YOU",): "Thank you.",
    ("HELLO",): "Hello.",
    ("YES",): "Yes.",
    ("NO",): "No.",
}


def canonicalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.strip().upper()).strip("_")


def normalize_with_rules(tokens: list[ConfirmedSignToken]) -> RuleNormalization:
    labels = tuple(canonicalize_label(token.label) for token in tokens)
    source_ids = [token.id for token in tokens]
    exact = EXACT_TEMPLATES.get(labels)
    if exact is not None:
        return RuleNormalization(text=exact, sourceTokenIds=source_ids, warnings=[])

    words = " ".join(label.replace("_", " ").lower() for label in labels).strip()
    text = f"{words[:1].upper()}{words[1:]}."
    return RuleNormalization(
        text=text,
        sourceTokenIds=source_ids,
        warnings=["UNMATCHED_TEMPLATE"],
    )
