"""Video validation and transient decoding boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from .errors import VideoDecodeError, payload_too_large, unsupported_media_type
from .schemas import DecodedVideo, Frame

SUPPORTED_CONTENT_TYPES = frozenset({"video/webm", "video/mp4"})


class VideoDecoder(Protocol):
    def decode(self, video: bytes, content_type: str) -> DecodedVideo: ...


def base_media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def validate_video_request(video: bytes, content_type: str, *, max_bytes: int) -> None:
    media_type = base_media_type(content_type)
    if media_type not in SUPPORTED_CONTENT_TYPES:
        raise unsupported_media_type("Only video/webm and video/mp4 are supported.")
    if not video:
        raise VideoDecodeError("Video payload is empty.")
    if len(video) > max_bytes:
        raise payload_too_large(
            "Video exceeds the configured byte limit.",
            details={"maxBytes": max_bytes},
        )


class OpenCVVideoDecoder:
    """Decode using OpenCV while keeping the temporary media file request-scoped."""

    def decode(self, video: bytes, content_type: str) -> DecodedVideo:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VideoDecodeError(
                "Video decoder dependency is not installed.",
            ) from exc

        suffix = ".webm" if base_media_type(content_type) == "video/webm" else ".mp4"
        temp_path: Path | None = None
        capture = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(video)
                temp_path = Path(handle.name)
            capture = cv2.VideoCapture(str(temp_path))
            if not capture.isOpened():
                raise VideoDecodeError("Video could not be decoded.")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frames: list[Frame] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = (len(frames) / fps * 1000) if fps > 0 else float(len(frames))
                frames.append(Frame(data=frame, timestamp_ms=timestamp_ms))
            if not frames:
                raise VideoDecodeError("Video contains no decodable frames.")
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or len(frames))
            duration_ms = (frame_count / fps * 1000) if fps > 0 else frames[-1].timestamp_ms
            if duration_ms <= 0:
                duration_ms = float(len(frames))
            # Ensure the optional numpy dependency was actually used for the decode boundary.
            if not isinstance(frames[0].data, np.ndarray):
                raise VideoDecodeError("Video decoder returned an unsupported frame type.")
            return DecodedVideo(tuple(frames), duration_ms)
        finally:
            if capture is not None:
                capture.release()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
