from pathlib import Path

import pytest

from src.sign.schemas import Vocabulary


def test_mvp_vocabulary_matches_wlasl100_output_space() -> None:
    vocabulary = Vocabulary.from_file(Path("config/mvp-en-v1.json"))

    assert vocabulary.version == "mvp-en-v1"
    assert vocabulary.model_version == "signbart-wlasl100-hf-8f98ae7"
    assert len(vocabulary.labels) == 100
    assert len(vocabulary.enabled_labels_by_index()) == 20
    assert vocabulary.labels[0].label == "BOOK"
    assert vocabulary.labels[99].label == "THURSDAY"


def test_vocabulary_rejects_all_disabled_labels(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text(
        '{"version":"v1","modelVersion":"m1","labels":['
        '{"index":0,"label":"BOOK","enabled":false}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enable at least one label"):
        Vocabulary.from_file(path)
