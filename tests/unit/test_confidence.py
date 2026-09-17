from src.sign.confidence import ConfidencePolicy, classify_prediction
from src.sign.schemas import Candidate, InputQuality


POLICY = ConfidencePolicy(0.80, 0.40, 0.20)


def test_confident_requires_threshold_and_margin() -> None:
    decision = classify_prediction(
        [Candidate("REPEAT", 0.86), Candidate("AGAIN", 0.09)],
        InputQuality(),
        POLICY,
    )
    assert decision.status == "CONFIDENT"
    assert decision.prediction == "REPEAT"
    assert decision.requires_confirmation is False


def test_ambiguous_when_margin_is_small() -> None:
    decision = classify_prediction(
        [Candidate("REPEAT", 0.84), Candidate("AGAIN", 0.70)],
        InputQuality(),
        POLICY,
    )
    assert decision.status == "AMBIGUOUS"
    assert decision.requires_confirmation is True


def test_unknown_when_top1_is_below_threshold() -> None:
    decision = classify_prediction(
        [Candidate("REPEAT", 0.30), Candidate("AGAIN", 0.20)],
        InputQuality(),
        POLICY,
    )
    assert decision.status == "UNKNOWN"
    assert decision.prediction is None


def test_no_sign_overrides_model_candidates() -> None:
    decision = classify_prediction(
        [Candidate("REPEAT", 0.99)], InputQuality(no_sign=True), POLICY
    )
    assert decision.status == "NO_SIGN"
    assert decision.prediction is None
