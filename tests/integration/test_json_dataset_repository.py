"""JSON dataset repository adapter."""

import json
from pathlib import Path

import pytest

from src.adapters.outbound.json_dataset_repository import (
    JsonDatasetRepository,
)
from src.domain.errors import CorruptedDataError, NotFoundError, RagError
from src.domain.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)
from tests.samples import write_json

DATASET = {"rag_questions": [
    {"question_id": "1", "question": "a?", "answer": "x",
     "sources": [{"file_path": "f.py", "first_character_index": 0,
                  "last_character_index": 4}]},
    {"question_id": "2", "question": "b?"},
]}


def test_read_valid(tmp_path: Path) -> None:
    """A valid file is parsed into the model."""
    path = write_json(tmp_path / "d.json", DATASET)
    dataset = JsonDatasetRepository().read(path, RagDataset)
    assert len(dataset.rag_questions) == 2
    assert isinstance(dataset.rag_questions[0], AnsweredQuestion)


def test_missing_file(tmp_path: Path) -> None:
    """A missing file or a directory is a NotFoundError."""
    repo = JsonDatasetRepository()
    with pytest.raises(NotFoundError, match="File not found"):
        repo.read(str(tmp_path / "missing.json"), RagDataset)
    with pytest.raises(NotFoundError):
        repo.read(str(tmp_path), RagDataset)


@pytest.mark.parametrize("content", ["{not json", "", "[1, 2", "nul"])
def test_malformed_json(tmp_path: Path, content: str) -> None:
    """Invalid JSON is reported as malformed."""
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(CorruptedDataError, match="Malformed JSON"):
        JsonDatasetRepository().read(str(path), RagDataset)


@pytest.mark.parametrize("data", [
    {}, [], {"rag_questions": [{"question_id": "1"}]},
    {"rag_questions": "nope"},
])
def test_wrong_schema(tmp_path: Path, data: object) -> None:
    """Valid JSON of the wrong shape names the expected model."""
    path = write_json(tmp_path / "w.json", data)
    with pytest.raises(CorruptedDataError, match="RagDataset format"):
        JsonDatasetRepository().read(path, RagDataset)


def test_not_utf8(tmp_path: Path) -> None:
    """A binary file is rejected cleanly."""
    path = tmp_path / "bin.json"
    path.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(CorruptedDataError):
        JsonDatasetRepository().read(str(path), RagDataset)


def test_write_creates_directories(tmp_path: Path) -> None:
    """Writing creates the directory and round-trips the model."""
    repo = JsonDatasetRepository()
    results = StudentSearchResults(k=1, search_results=[
        MinimalSearchResults(question_id="1", question="é?",
                             retrieved_sources=[]),
    ])
    out = repo.write(str(tmp_path / "a" / "b"), "r.json", results)
    assert out == str(tmp_path / "a" / "b" / "r.json")
    text = Path(out).read_text(encoding="utf-8")
    assert "é?" in text and text.endswith("\n")
    assert json.loads(text)["k"] == 1
    assert repo.read(out, StudentSearchResults) == results


def test_write_failure(tmp_path: Path) -> None:
    """An unwritable destination raises a RagError."""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    results = StudentSearchResults(k=1, search_results=[])
    with pytest.raises(RagError, match="Cannot write"):
        JsonDatasetRepository().write(str(blocker / "sub"), "r.json",
                                      results)
