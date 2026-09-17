import pytest
from types import SimpleNamespace

from src.sign.config import Settings
from src.sign.errors import SignServiceError
from src.sign.model import SignBartModelAdapter
from src.sign.schemas import ModelInputConfig, Vocabulary, VocabularyLabel
from src.sign.service import SignPredictionService


def test_signbart_without_checkpoint_is_not_ready() -> None:
    settings = Settings(model_backend="signbart", model_version="signbart-mvp-v1")
    adapter = SignBartModelAdapter(
        settings,
        ModelInputConfig(required_frames=16, width=224, height=224, layout="NCTHW"),
    )
    with pytest.raises(SignServiceError) as raised:
        adapter.load(None, "cuda:0")
    assert raised.value.code == "MODEL_NOT_READY"


def test_candidate_ranking_only_exposes_enabled_labels() -> None:
    settings = Settings()
    service = SignPredictionService(settings)
    service.vocabulary = Vocabulary(
        version="mvp-en-v1",
        model_version="model-v1",
        labels=(
            VocabularyLabel(0, "DISABLED_TOP", "Disabled top", enabled=False),
            VocabularyLabel(1, "ENABLED_SECOND", "Enabled second", enabled=True),
            VocabularyLabel(2, "ENABLED_THIRD", "Enabled third", enabled=True),
        ),
    )

    candidates = service._rank_candidates((10.0, 9.0, 8.0), top_k=2)

    assert [candidate.label for candidate in candidates] == ["ENABLED_SECOND", "ENABLED_THIRD"]
    assert candidates[0].confidence < 0.5


def test_output_dimension_is_recorded_from_signbart_head() -> None:
    settings = Settings(model_backend="signbart")
    adapter = SignBartModelAdapter(
        settings,
        ModelInputConfig(required_frames=16, width=224, height=224, layout="NCTHW"),
    )
    adapter._model = SimpleNamespace(
        classification_head=SimpleNamespace(out_proj=SimpleNamespace(out_features=100))
    )

    adapter._validate_output_dimension()

    assert adapter._output_features == 100
