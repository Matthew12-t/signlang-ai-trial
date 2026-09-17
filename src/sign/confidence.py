"""Confidence-aware classification kept independent from the model adapter."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import Candidate, InputQuality


@dataclass(frozen=True)
class ConfidencePolicy:
    confident_threshold: float
    unknown_threshold: float
    min_top1_top2_margin: float


@dataclass(frozen=True)
class PredictionDecision:
    status: str
    prediction: str | None
    requires_confirmation: bool


def classify_prediction(
    candidates: list[Candidate],
    quality: InputQuality,
    policy: ConfidencePolicy,
) -> PredictionDecision:
    if quality.no_sign or not candidates:
        return PredictionDecision("NO_SIGN", None, False)
    top1 = candidates[0].confidence
    top2 = candidates[1].confidence if len(candidates) > 1 else 0.0
    margin = top1 - top2
    if top1 < policy.unknown_threshold:
        return PredictionDecision("UNKNOWN", None, False)
    if top1 >= policy.confident_threshold and margin >= policy.min_top1_top2_margin:
        return PredictionDecision("CONFIDENT", candidates[0].label, False)
    return PredictionDecision("AMBIGUOUS", candidates[0].label, True)
