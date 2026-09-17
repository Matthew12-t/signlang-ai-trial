"""Explicit mock and upstream-compatible SignBart model adapters."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from .config import Settings
from .errors import SignServiceError, inference_failed, model_not_ready
from .schemas import InputQuality, ModelInputConfig, ModelPrediction, PreprocessedVideo, Vocabulary


class SignRecognitionModel(Protocol):
    model_version: str
    input_config: ModelInputConfig

    def load(self, checkpoint_path: Path | None, device: str) -> None: ...
    def warmup(self) -> None: ...
    def predict_logits(self, inputs: PreprocessedVideo) -> ModelPrediction: ...


class MockSignRecognitionModel:
    """Deterministic fixture; never represents a real model prediction."""

    model_version = "mock-sign-v0"

    def __init__(self, vocabulary: Vocabulary, input_config: ModelInputConfig) -> None:
        self.vocabulary = vocabulary
        self.input_config = input_config

    def load(self, checkpoint_path: Path | None, device: str) -> None:
        return None

    def warmup(self) -> None:
        return None

    def predict_logits(self, inputs: PreprocessedVideo) -> ModelPrediction:
        if not inputs.frames:
            return ModelPrediction((), quality=InputQuality(no_sign=True))
        # Deliberately deterministic and clearly identified as mock output.
        logits = tuple(
            6.0 if index == 0 else 0.5 - index * 0.05
            for index in range(len(self.vocabulary.labels))
        )
        return ModelPrediction(logits, quality=InputQuality(no_sign=inputs.no_sign))


class SignBartModelAdapter:
    """Load and run the upstream SignBart skeleton model.

    The upstream implementation defines ``SignBart(config)`` and expects
    ``keypoints`` shaped ``(B, T, 75, 2)`` plus an ``(B, T)`` attention mask.
    Checkpoints are not vendored. Configure SIGN_SIGNBART_REPO and
    SIGN_SIGNBART_CONFIG for the upstream checkout, or provide a custom loader
    as ``module:function`` through SIGN_SIGNBART_LOADER.
    """

    def __init__(self, settings: Settings, input_config: ModelInputConfig) -> None:
        self.settings = settings
        self.input_config = input_config
        self.model_version = settings.model_version
        self._model: Any = None
        self._torch: Any = None

    def load(self, checkpoint_path: Path | None, device: str) -> None:
        if checkpoint_path is None:
            raise model_not_ready("SignBart checkpoint is not configured.")
        if not checkpoint_path.exists():
            raise model_not_ready(
                "SignBart checkpoint was not found.",
                details={"checkpointConfigured": True},
            )
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise model_not_ready("PyTorch is required for the SignBart backend.") from exc
        self._torch = torch
        if self.settings.signbart_loader:
            loader = self._load_callable(self.settings.signbart_loader)
            self._model = loader(checkpoint_path, device)
        elif self.settings.signbart_repo and self.settings.signbart_config:
            self._model = self._load_upstream_model(
                self.settings.signbart_repo,
                self.settings.signbart_config,
                checkpoint_path,
                device,
            )
        elif checkpoint_path.suffix.lower() in {".jit", ".torchscript", ".ts"}:
            try:
                self._model = torch.jit.load(str(checkpoint_path), map_location=device)
            except Exception as exc:  # pragma: no cover - external artifact boundary
                raise model_not_ready("TorchScript SignBart checkpoint could not be loaded.") from exc
        else:
            raise model_not_ready(
                "A .pth SignBart checkpoint requires SIGN_SIGNBART_REPO and SIGN_SIGNBART_CONFIG."
            )
        if hasattr(self._model, "eval"):
            self._model.eval()
        self._validate_output_dimension()

    @staticmethod
    def _load_callable(reference: str):
        try:
            module_name, function_name = reference.split(":", 1)
            return getattr(importlib.import_module(module_name), function_name)
        except (ValueError, ImportError, AttributeError) as exc:
            raise model_not_ready(
                "SIGN_SIGNBART_LOADER must reference an importable module:function."
            ) from exc

    def _load_upstream_model(
        self,
        repo_path: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
    ) -> Any:
        if not repo_path.is_dir() or not config_path.is_file():
            raise model_not_ready("SignBart repository or model config was not found.")
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise model_not_ready("PyYAML is required to load the SignBart config.") from exc
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            module = self._import_upstream_module(repo_path)
            model = module.SignBart(config)
        except (OSError, KeyError, TypeError, ValueError, ImportError) as exc:
            raise model_not_ready("SignBart source or config could not be loaded.") from exc
        try:
            checkpoint = self._load_checkpoint(checkpoint_path, device)
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
                checkpoint = checkpoint["model"]
            if not isinstance(checkpoint, dict):
                raise TypeError("checkpoint does not contain a state dict")
            result = model.load_state_dict(checkpoint, strict=False)
            allowed_wrapper_keys = {
                "encoder.embed_tokens.weight",
                "decoder.embed_tokens.weight",
            }
            unexpected_keys = [
                key for key in result.unexpected_keys if key not in allowed_wrapper_keys
            ]
            if result.missing_keys or unexpected_keys:
                raise ValueError("checkpoint keys do not match the SignBart architecture")
            model.to(device)
            return model
        except SignServiceError:
            raise
        except Exception as exc:  # pragma: no cover - external artifact boundary
            raise model_not_ready("SignBart checkpoint does not match the configured model.") from exc

    @staticmethod
    def _import_upstream_module(repo_path: Path) -> Any:
        """Import upstream despite its mixed absolute and relative imports."""
        package_name = f"_isyara_signbart_{abs(hash(repo_path.resolve()))}"
        package = sys.modules.get(package_name)
        if package is None:
            package = ModuleType(package_name)
            package.__path__ = [str(repo_path)]  # type: ignore[attr-defined]
            package.__package__ = package_name
            sys.modules[package_name] = package

        aliases = ("attention", "layers", "utils", "encoder", "decoder", "model")
        previous_aliases: dict[str, Any] = {}
        try:
            for name in aliases:
                qualified_name = f"{package_name}.{name}"
                imported = importlib.import_module(qualified_name)
                previous_aliases[name] = sys.modules.get(name)
                sys.modules[name] = imported
            return sys.modules[f"{package_name}.model"]
        finally:
            for name, previous in previous_aliases.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    def _load_checkpoint(self, checkpoint_path: Path, device: str) -> Any:
        if checkpoint_path.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file  # type: ignore[import-not-found]
            except ImportError as exc:
                raise model_not_ready(
                    "safetensors is required to load this SignBart checkpoint."
                ) from exc
            return load_file(str(checkpoint_path), device=device)
        try:
            return self._torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        except TypeError:  # Older PyTorch versions do not accept weights_only.
            return self._torch.load(str(checkpoint_path), map_location=device)

    def _validate_output_dimension(self) -> None:
        if self._model is None:
            return
        head = getattr(self._model, "classification_head", None)
        output_projection = getattr(head, "out_proj", None)
        output_features = getattr(output_projection, "out_features", None)
        if output_features is not None and self.settings.model_backend == "signbart":
            # The vocabulary size is checked by the service once the vocabulary is loaded.
            self._output_features = int(output_features)

    def _to_model_inputs(self, inputs: PreprocessedVideo) -> tuple[Any, Any]:
        if self._torch is None:
            raise model_not_ready("SignBart model has not been loaded.")
        try:
            import numpy as np  # type: ignore[import-not-found]
            if self.input_config.representation == "skeleton":
                array = np.stack(inputs.frames, axis=0).astype(np.float32)
                expected = (75, 2)
                if array.ndim != 3 or tuple(array.shape[1:]) != expected:
                    raise ValueError("SignBart keypoints must have shape (T, 75, 2)")
                keypoints = self._torch.from_numpy(array[None, ...]).float()
                attention_mask = self._torch.ones(
                    (1, array.shape[0]), dtype=self._torch.float32
                )
                return keypoints.to(self.settings.device), attention_mask.to(self.settings.device)
            array = np.stack(inputs.frames, axis=0)
            if array.ndim != 4 or array.shape[-1] != 3:
                raise ValueError("preprocessed frames must have shape T,H,W,C")
            array = array.transpose(0, 3, 1, 2)
            if self.input_config.layout == "NCTHW":
                array = array.transpose(1, 0, 2, 3)[None, ...]
            else:
                array = array[None, ...]
            tensor = self._torch.from_numpy(array)
            return tensor.float().to(self.settings.device), self._torch.ones(
                (1, array.shape[1]), dtype=self._torch.float32, device=self.settings.device
            )
        except ImportError as exc:
            raise model_not_ready("NumPy is required for the SignBart backend.") from exc
        except (TypeError, ValueError) as exc:
            raise inference_failed("Preprocessing output is incompatible with SignBart.") from exc

    def warmup(self) -> None:
        if self._model is None:
            raise model_not_ready("SignBart model has not been loaded.")
        warmup_method = getattr(self._model, "warmup", None)
        if callable(warmup_method):
            warmup_method()
            return
        with self._torch.inference_mode():
            if self.input_config.representation == "skeleton":
                keypoints = self._torch.zeros(
                    (1, self.input_config.required_frames, 75, 2),
                    device=self.settings.device,
                )
                mask = self._torch.ones(
                    (1, self.input_config.required_frames),
                    dtype=self._torch.float32,
                    device=self.settings.device,
                )
                self._model(keypoints, mask)
            else:
                if self.input_config.layout == "NCTHW":
                    shape = (1, 3, self.input_config.required_frames, self.input_config.height, self.input_config.width)
                else:
                    shape = (1, self.input_config.required_frames, 3, self.input_config.height, self.input_config.width)
                self._model(self._torch.zeros(shape, device=self.settings.device))

    def predict_logits(self, inputs: PreprocessedVideo) -> ModelPrediction:
        if self._model is None:
            raise model_not_ready("SignBart model has not been loaded.")
        tensor, attention_mask = self._to_model_inputs(inputs)
        with self._torch.inference_mode():
            if self.input_config.representation == "skeleton":
                output = self._model(tensor, attention_mask)
            else:
                output = self._model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        elif isinstance(output, dict):
            output = output.get("logits")
        elif hasattr(output, "logits"):
            output = output.logits
        if output is None or not hasattr(output, "detach"):
            raise inference_failed("SignBart output did not contain logits.")
        values = output.detach().float().reshape(-1).tolist()
        return ModelPrediction(tuple(float(value) for value in values), InputQuality(inputs.no_sign))


def build_model(settings: Settings, vocabulary: Vocabulary) -> SignRecognitionModel:
    representation = "skeleton" if settings.model_backend == "signbart" else "rgb"
    input_config = ModelInputConfig(
        required_frames=settings.required_frames,
        width=settings.input_width,
        height=settings.input_height,
        layout=settings.input_layout,
        representation=representation,
    )
    if settings.model_backend == "mock":
        return MockSignRecognitionModel(vocabulary, input_config)
    return SignBartModelAdapter(settings, input_config)
