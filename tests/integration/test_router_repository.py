"""Query router model persisted as a JSON file."""

import json
from pathlib import Path

import pytest

from src.adapters.outbound.json_router_repository import (
    JsonRouterRepository,
)
from src.domain.errors import CorruptedDataError
from src.domain.priors import FEATURE_NAMES, RouterModel

MODEL = RouterModel(
    weights=[0.5, -1.0] + [0.0] * (len(FEATURE_NAMES) - 2) + [2.0, -3.0],
    bias=0.25,
    vocabulary=["default", "docker"],
)


def test_round_trip(tmp_path: Path) -> None:
    """A saved model is loaded back identical, from a new directory."""
    path = tmp_path / "models" / "router.json"
    repository = JsonRouterRepository(str(path))
    assert repository.save(MODEL) == str(path)
    assert path.is_file()
    loaded = JsonRouterRepository(str(path)).load()
    assert loaded == MODEL
    assert loaded is not None
    query = "What is the default of foo_bar?"
    assert loaded.p_code(query) == MODEL.p_code(query)


def test_save_overwrites_the_previous_model(tmp_path: Path) -> None:
    """Only the last saved model is kept."""
    repository = JsonRouterRepository(str(tmp_path / "router.json"))
    repository.save(MODEL)
    other = RouterModel(weights=[1.0] * len(FEATURE_NAMES))
    repository.save(other)
    assert repository.load() == other


def test_missing_file_is_no_model(tmp_path: Path) -> None:
    """Without a file (or with a directory) there is no router."""
    assert JsonRouterRepository(str(tmp_path / "nope.json")).load() is None
    assert JsonRouterRepository(str(tmp_path)).load() is None


@pytest.mark.parametrize("content", [
    "{not json", "[]", '{"bias": 1.0}', '{"weights": "many"}', "",
])
def test_corrupt_file_is_reported(tmp_path: Path, content: str) -> None:
    """A file that is not a router model is a corrupted-data error."""
    path = tmp_path / "router.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CorruptedDataError, match="Invalid router model"):
        JsonRouterRepository(str(path)).load()


@pytest.mark.parametrize("weights, vocabulary", [
    (len(FEATURE_NAMES) - 1, []),
    (len(FEATURE_NAMES) + 1, []),
    (len(FEATURE_NAMES), ["docker"]),
    (0, []),
])
def test_wrong_number_of_weights_is_reported(
    tmp_path: Path, weights: int, vocabulary: list[str]
) -> None:
    """A model trained with other features must be trained again."""
    path = tmp_path / "router.json"
    path.write_text(json.dumps({
        "weights": [0.0] * weights, "vocabulary": vocabulary,
    }), encoding="utf-8")
    with pytest.raises(CorruptedDataError, match="train_router"):
        JsonRouterRepository(str(path)).load()
