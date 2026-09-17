"""Local VoxCPM2 runtime adapter."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.shared.config import Settings
from src.shared.errors import ServiceError
from src.shared.models import GeneratedAudio

logger = logging.getLogger(__name__)


class VoxCPMProvider:
    provider_name = "local-voxcpm"

    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.tts_model
        self._settings = settings
        self._model: Any | None = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.tts_max_concurrency)

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def load(self) -> None:
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                self._model = await asyncio.to_thread(self._load_sync)
                self._load_error = None
                logger.info("Loaded TTS model %s", self.model_id)
            except Exception as error:
                self._load_error = str(error)
                logger.exception("Unable to load TTS model %s", self.model_id)
                raise

    def _load_sync(self) -> Any:
        try:
            from voxcpm import VoxCPM
        except ImportError as error:
            raise ServiceError(
                code="MODEL_DEPENDENCY_MISSING",
                message="The voxcpm runtime is not installed.",
                status_code=503,
            ) from error

        return VoxCPM.from_pretrained(
            self.model_id,
            device=self._settings.tts_device,
            optimize=self._settings.tts_optimize,
            load_denoiser=self._settings.tts_load_denoiser,
        )

    async def synthesize(self, text: str) -> GeneratedAudio:
        if self._model is None:
            try:
                await self.load()
            except ServiceError:
                raise
            except Exception as error:
                raise ServiceError(
                    code="MODEL_UNAVAILABLE",
                    message="The Text-to-Speech model could not be loaded.",
                    status_code=503,
                    retryable=True,
                ) from error

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._settings.model_queue_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise ServiceError(
                code="MODEL_BUSY",
                message="The Text-to-Speech model is processing another request.",
                status_code=503,
                retryable=True,
            ) from error

        try:
            return await asyncio.to_thread(self._synthesize_sync, text)
        except ServiceError:
            raise
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise ServiceError(
                    code="MODEL_OUT_OF_MEMORY",
                    message="The Text-to-Speech model ran out of GPU memory.",
                    status_code=503,
                ) from error
            raise ServiceError(
                code="INFERENCE_FAILED",
                message="Text-to-Speech inference failed.",
                status_code=502,
                retryable=True,
            ) from error
        except Exception as error:
            raise ServiceError(
                code="UPSTREAM_BAD_RESPONSE",
                message="The Text-to-Speech model returned unusable audio.",
                status_code=502,
                retryable=True,
            ) from error
        finally:
            self._semaphore.release()

    def _synthesize_sync(self, text: str) -> GeneratedAudio:
        import numpy as np
        import torch

        torch.manual_seed(self._settings.tts_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._settings.tts_seed)

        waveform = self._model.generate(
            text=text,
            cfg_value=self._settings.tts_cfg_value,
            inference_timesteps=self._settings.tts_inference_timesteps,
            normalize=self._settings.tts_normalize,
            denoise=False,
            retry_badcase=False,
        )
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ServiceError(
                code="UPSTREAM_BAD_RESPONSE",
                message="The Text-to-Speech model returned empty audio.",
                status_code=502,
                retryable=True,
            )
        sample_rate = int(self._model.tts_model.sample_rate)
        return GeneratedAudio(samples=samples, sample_rate=sample_rate)
