import numpy as np

from src.sign.preprocessing import sample_frames
from src.sign.preprocessing import normalize_skeleton
from src.sign.schemas import Frame


def test_uniform_sampling_keeps_clip_boundaries() -> None:
    frames = tuple(Frame(index, float(index)) for index in range(10))
    sampled = sample_frames(frames, 4)
    assert [frame.data for frame in sampled] == [0, 3, 6, 9]


def test_short_clip_is_not_duplicated() -> None:
    frames = tuple(Frame(index, float(index)) for index in range(3))
    assert sample_frames(frames, 8) == frames


def test_signbart_normalization_preserves_shape_and_bounds() -> None:
    keypoints = np.zeros((1, 75, 2), dtype=np.float32)
    keypoints[0, 11:17] = [[0.2, 0.3], [0.4, 0.3], [0.2, 0.6], [0.4, 0.6], [0.3, 0.45], [0.35, 0.5]]
    normalized = normalize_skeleton(keypoints)
    assert normalized.shape == (1, 75, 2)
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0
    assert np.all(normalized[0, 33:54] == 0.0)
