"""Frame sampling and SignBart skeleton preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .errors import model_not_ready
from .schemas import DecodedVideo, Frame, ModelInputConfig, PreprocessedVideo


class KeypointExtractor(Protocol):
    def extract(self, frames: tuple[Frame, ...]) -> tuple[Any, bool]: ...


@dataclass(frozen=True)
class SkeletonLayout:
    """MediaPipe Holistic's 75-point layout used by the upstream SignBart code."""

    body: tuple[int, ...] = tuple(range(11, 17))
    left_hand: tuple[int, ...] = tuple(range(33, 54))
    right_hand: tuple[int, ...] = tuple(range(54, 75))

    @property
    def parts(self) -> tuple[tuple[int, ...], ...]:
        return (self.body, self.left_hand, self.right_hand)


SIGNBART_LAYOUT = SkeletonLayout()


def sample_frames(frames: list[Frame] | tuple[Frame, ...], required_frames: int) -> tuple[Frame, ...]:
    """Select frames uniformly, retaining both clip boundaries when possible."""
    if required_frames < 1:
        raise ValueError("required_frames must be positive")
    if not frames:
        return ()
    if len(frames) <= required_frames:
        return tuple(frames)
    if required_frames == 1:
        return (frames[0],)
    indices = [round(i * (len(frames) - 1) / (required_frames - 1)) for i in range(required_frames)]
    return tuple(frames[index] for index in indices)


def normalize_skeleton(keypoints: Any, layout: SkeletonLayout = SIGNBART_LAYOUT) -> Any:
    """Match the upstream local-bounding-box normalization for each component."""
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise model_not_ready("NumPy is required for SignBart skeleton preprocessing.") from exc

    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (75, 2):
        raise ValueError("SignBart keypoints must have shape (T, 75, 2)")
    normalized = np.clip(array, 0.0, 1.0).copy()
    for part in layout.parts:
        for frame_index in range(normalized.shape[0]):
            component = normalized[frame_index, list(part), :]
            valid = np.any(component > 0.0, axis=1)
            if not np.any(valid):
                continue
            points = component[valid]
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            width = max_x - min_x
            height = max_y - min_y
            if width > height:
                delta_x = 0.05 * width
                delta_y = delta_x + ((width - height) / 2)
            else:
                delta_y = 0.05 * height
                delta_x = delta_y + ((height - width) / 2)
            start = np.clip(np.array([min_x - delta_x, min_y - delta_y]), 0.0, 1.0)
            end = np.clip(np.array([max_x + delta_x, max_y + delta_y]), 0.0, 1.0)
            denominator = end - start
            safe_denominator = np.where(denominator == 0.0, 1.0, denominator)
            transformed = np.clip((component - start) / safe_denominator, 0.0, 1.0)
            transformed[~valid] = 0.0
            normalized[frame_index, list(part), :] = transformed
    return normalized


class MediaPipeHolisticExtractor:
    """Extract the 75-point ``(x, y)`` skeleton expected by SignBart."""

    def __init__(self) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
            import mediapipe as mp  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise model_not_ready(
                "MediaPipe and OpenCV are required for SignBart video preprocessing."
            ) from exc
        self.cv2 = cv2
        self.mp = mp
        self.np = np

    def extract(self, frames: tuple[Frame, ...]) -> tuple[Any, bool]:
        holistic_module = self.mp.solutions.holistic
        keypoint_frames = self.np.zeros((len(frames), 75, 2), dtype=self.np.float32)
        with holistic_module.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            for frame_index, frame in enumerate(frames):
                image = self.np.asarray(frame.data)
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("decoded frames must have shape H,W,3")
                rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                if result.pose_landmarks:
                    for index, landmark in enumerate(result.pose_landmarks.landmark[:33]):
                        keypoint_frames[frame_index, index] = [landmark.x, landmark.y]
                if result.left_hand_landmarks:
                    for index, landmark in enumerate(result.left_hand_landmarks.landmark[:21]):
                        keypoint_frames[frame_index, 33 + index] = [landmark.x, landmark.y]
                if result.right_hand_landmarks:
                    for index, landmark in enumerate(result.right_hand_landmarks.landmark[:21]):
                        keypoint_frames[frame_index, 54 + index] = [landmark.x, landmark.y]
        clipped = self.np.clip(keypoint_frames, 0.0, 1.0)
        return clipped, not bool(self.np.any(clipped > 0.0))


def _resize_and_normalize(frame: Any, config: ModelInputConfig) -> Any:
    """Convert an OpenCV BGR frame to normalized RGB for the explicit mock path."""
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return frame
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("decoded frames must have three color channels")
    resized = cv2.resize(array, (config.width, config.height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def preprocess_frames(
    decoded: DecodedVideo,
    config: ModelInputConfig,
    *,
    extractor: KeypointExtractor | None = None,
) -> PreprocessedVideo:
    if not decoded.frames:
        raise ValueError("cannot preprocess an empty video")
    sampled = sample_frames(decoded.frames, config.required_frames)
    if config.representation == "skeleton":
        if extractor is None:
            raise ValueError("a keypoint extractor is required for skeleton preprocessing")
        keypoints, no_sign = extractor.extract(sampled)
        normalized = normalize_skeleton(keypoints)
        return PreprocessedVideo(
            frames=tuple(normalized),
            duration_ms=decoded.duration_ms,
            input_config=config,
            no_sign=no_sign,
        )
    processed = tuple(_resize_and_normalize(frame.data, config) for frame in sampled)
    return PreprocessedVideo(
        frames=processed,
        duration_ms=decoded.duration_ms,
        input_config=config,
    )
