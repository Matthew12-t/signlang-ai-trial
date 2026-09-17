"""Application service for stateless isolated-sign prediction."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from uuid import UUID, uuid4

from .confidence import ConfidencePolicy, classify_prediction
from .config import Settings
from .errors import (
    SignServiceError,
    VideoDecodeError,
    inference_busy,
    inference_failed,
    inference_timeout,
    model_not_ready,
    vocabulary_mismatch,
)
from .model import SignRecognitionModel, build_model
from .observability import log_prediction
from .preprocessing import KeypointExtractor, MediaPipeHolisticExtractor, preprocess_frames
from .schemas import Candidate, DecodedVideo, ModelInputConfig, Vocabulary
from .video import OpenCVVideoDecoder, VideoDecoder, validate_video_request


class InferenceLimiter:
    def __init__(self, concurrency: int, queue_size: int) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)
        self.queue_size = queue_size
        self.waiting = 0

    async def __aenter__(self):
        if self.semaphore.locked() and self.waiting >= self.queue_size:
            raise inference_busy()
        self.waiting += 1
        try:
            await self.semaphore.acquire()
            return self
        finally:
            self.waiting -= 1

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.semaphore.release()


def _softmax(logits: tuple[float, ...]) -> list[float]:
    if not logits:
        return []
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


class SignPredictionService:
    def __init__(
        self,
        settings: Settings,
        *,
        decoder: VideoDecoder | None = None,
        model: SignRecognitionModel | None = None,
        extractor: KeypointExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.decoder = decoder or OpenCVVideoDecoder()
        self.model = model
        self.extractor = extractor
        self.vocabulary: Vocabulary | None = None
        self._ready = False
        self._checks: dict[str, str] = {}
        self._limiter = InferenceLimiter(settings.max_concurrency, settings.queue_size)

    def startup(self) -> None:
        self._ready = False
        self._checks = {}
        config_errors = self.settings.validation_errors()
        if config_errors:
            self._checks["configuration"] = "; ".join(config_errors)
            return
        try:
            self.vocabulary = Vocabulary.from_file(Path(self.settings.vocabulary_path))
            if self.vocabulary.version != self.settings.vocabulary_version:
                raise ValueError("vocabulary version does not match configuration")
            if self.vocabulary.model_version != self.settings.model_version and self.settings.model_backend == "signbart":
                raise ValueError("vocabulary modelVersion does not match SIGN_MODEL_VERSION")
            self._checks["vocabulary"] = self.vocabulary.version
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._checks["vocabulary"] = str(exc)
            return
        try:
            if self.settings.model_backend == "signbart":
                self.extractor = self.extractor or MediaPipeHolisticExtractor()
            self.model = self.model or build_model(self.settings, self.vocabulary)
            self.model.load(self.settings.model_path, self.settings.device)
            output_features = getattr(self.model, "_output_features", None)
            if output_features is not None and output_features != len(self.vocabulary.labels):
                raise ValueError("model output dimension does not match vocabulary labels")
            self.model.warmup()
            self._checks["model"] = "ready" if self.settings.model_backend == "signbart" else "mock"
            self._checks["device"] = self.settings.device if self.settings.model_backend == "signbart" else "cpu"
            self._checks["warmup"] = "ready"
            self._ready = True
        except SignServiceError as exc:
            self._checks["model"] = exc.message
        except Exception as exc:  # pragma: no cover - external model boundary
            self._checks["model"] = str(exc)

    def liveness(self) -> dict[str, str]:
        return {"status": "alive", "service": "sign-language"}

    def readiness(self) -> dict[str, object]:
        return {
            "status": "ready" if self._ready else "not_ready",
            "service": "sign-language",
            "checks": self._checks,
        }

    def readiness_error(self, request_id: UUID) -> SignServiceError:
        return model_not_ready(
            "Sign Language Service model is not ready.",
            request_id=request_id,
            details={"checks": self._checks},
        )

    def _require_ready(self) -> None:
        if not self._ready or self.model is None or self.vocabulary is None:
            raise model_not_ready("Sign Language Service model is not ready.", details={"checks": self._checks})

    def _rank_candidates(self, logits: tuple[float, ...], top_k: int) -> list[Candidate]:
        assert self.vocabulary is not None
        probabilities = _softmax(logits)
        labels = self.vocabulary.enabled_labels_by_index()
        ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
        candidates = [
            Candidate(label=labels[index].label, confidence=round(float(probability), 6))
            for index, probability in ranked
            if index in labels
        ][:top_k]
        return candidates

    async def predict(
        self,
        video: bytes,
        *,
        content_type: str,
        top_k: int,
        vocabulary_version: str,
        request_id: UUID,
    ) -> dict[str, object]:
        self._require_ready()
        assert self.model is not None
        assert self.vocabulary is not None
        if vocabulary_version != self.vocabulary.version:
            raise vocabulary_mismatch(
                "Requested vocabulary version does not match the loaded vocabulary.",
                details={"loadedVersion": self.vocabulary.version},
            )
        if not 1 <= top_k <= 5:
            raise SignServiceError("INVALID_REQUEST", "topK must be between 1 and 5.", 400, False)
        prediction_id = f"pred_{uuid4().hex}"
        started = time.perf_counter()
        try:
            validate_video_request(video, content_type, max_bytes=self.settings.max_clip_bytes)
            decoded = self.decoder.decode(video, content_type)
        except VideoDecodeError:
            return self._response(
                request_id=request_id,
                prediction_id=prediction_id,
                candidates=[],
                status="INVALID_INPUT",
                prediction=None,
                confidence=0.0,
                started=started,
            )
        if decoded.duration_ms > self.settings.max_clip_duration_ms:
            raise SignServiceError(
                "PAYLOAD_TOO_LARGE",
                "Video exceeds the configured duration limit.",
                413,
                False,
                details={"maxDurationMs": self.settings.max_clip_duration_ms},
            )
        config = ModelInputConfig(
            required_frames=self.settings.required_frames,
            width=self.settings.input_width,
            height=self.settings.input_height,
            layout=self.settings.input_layout,
            representation="skeleton" if self.settings.model_backend == "signbart" else "rgb",
        )
        try:
            prepared = preprocess_frames(decoded, config, extractor=self.extractor)
        except (ValueError, TypeError) as exc:
            return self._response(
                request_id=request_id,
                prediction_id=prediction_id,
                candidates=[],
                status="INVALID_INPUT",
                prediction=None,
                confidence=0.0,
                started=started,
            )
        try:
            async with self._limiter:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.model.predict_logits, prepared),
                    timeout=self.settings.inference_timeout_ms / 1000,
                )
        except SignServiceError:
            raise
        except TimeoutError as exc:
            raise inference_timeout() from exc
        except Exception as exc:  # pragma: no cover - external model boundary
            raise inference_failed("Sign inference failed.") from exc
        candidates = self._rank_candidates(result.logits, top_k)
        policy = ConfidencePolicy(
            confident_threshold=self.settings.confident_threshold,  # type: ignore[arg-type]
            unknown_threshold=self.settings.unknown_threshold,  # type: ignore[arg-type]
            min_top1_top2_margin=self.settings.min_top1_top2_margin,  # type: ignore[arg-type]
        )
        decision = classify_prediction(candidates, result.quality, policy)
        confidence = candidates[0].confidence if candidates else 0.0
        response = self._response(
            request_id=request_id,
            prediction_id=prediction_id,
            candidates=candidates,
            status=decision.status,
            prediction=decision.prediction,
            confidence=confidence,
            started=started,
            model_version=self.model.model_version,
        )
        log_prediction(
            requestId=str(request_id),
            predictionId=prediction_id,
            status=decision.status,
            modelVersion=self.model.model_version,
            vocabularyVersion=self.vocabulary.version,
            durationMs=round(decoded.duration_ms, 2),
            frameCount=len(decoded.frames),
            latencyMs=response["latencyMs"],
        )
        return response

    def _response(
        self,
        *,
        request_id: UUID,
        prediction_id: str,
        candidates: list[Candidate],
        status: str,
        prediction: str | None,
        confidence: float,
        started: float,
        model_version: str | None = None,
    ) -> dict[str, object]:
        assert self.vocabulary is not None
        return {
            "requestId": str(request_id),
            "predictionId": prediction_id,
            "prediction": prediction,
            "confidence": confidence,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "status": status,
            "requiresConfirmation": status == "AMBIGUOUS",
            "modelVersion": model_version or (self.model.model_version if self.model else self.settings.model_version),
            "vocabularyVersion": self.vocabulary.version,
            "latencyMs": max(0, round((time.perf_counter() - started) * 1000)),
        }
