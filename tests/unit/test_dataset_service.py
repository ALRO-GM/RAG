"""DatasetService: batch search, batch answering and evaluation."""

import json
from typing import Any, Dict, List

import pytest

from src.application.answering import AnswerService
from src.application.datasets import DatasetService
from src.application.search import SearchService
from src.domain.errors import CorruptedDataError, InvalidInputError
from src.domain.errors import NotFoundError
from src.domain.models import Chunk
from src.domain.text_analysis import tokenize
from tests.fakes import (
    FakeEmbedder,
    FakeGenerator,
    FakeSourceReader,
    InMemoryDatasetRepository,
    InMemoryIndexRepository,
    RecordingProgress,
    make_snapshot,
)

SOURCE_TEXT = "0123456789" * 20


def source(path: str, first: int, last: int) -> Dict[str, Any]:
    """Return a source as JSON-ready data."""
    return {"file_path": path, "first_character_index": first,
            "last_character_index": last}


def dataset_json() -> str:
    """Return a dataset with two answered and one unanswered question."""
    return json.dumps({"rag_questions": [
        {"question_id": "q1", "question": "FP8 quantization memory",
         "answer": "x", "sources": [source("b.md", 0, 37)]},
        {"question_id": "q2", "question": "KVCacheManager blocks",
         "answer": "y", "sources": [source("a.py", 40, 77)]},
        {"question_id": "q3", "question": "   "},
    ]})


def results_json(sources: List[List[Dict[str, Any]]], k: int = 3) -> str:
    """Return saved search results for q1, q2... with ``sources``."""
    return json.dumps({"k": k, "search_results": [
        {"question_id": f"q{i}", "question": f"question {i}",
         "retrieved_sources": s}
        for i, s in enumerate(sources, start=1)
    ]})


@pytest.fixture()
def repo() -> InMemoryDatasetRepository:
    """Return a dataset repository holding the sample dataset."""
    return InMemoryDatasetRepository({"in/dataset.json": dataset_json()})


@pytest.fixture()
def datasets(
    chunks: List[Chunk], repo: InMemoryDatasetRepository
) -> DatasetService:
    """Return a DatasetService wired with fakes."""
    index = InMemoryIndexRepository()
    index.snapshot = make_snapshot(chunks)
    search = SearchService(index, tokenize, FakeEmbedder())
    answers = AnswerService(search, FakeGenerator())
    reader = FakeSourceReader({"src.txt": SOURCE_TEXT})
    return DatasetService(repo, search, answers, reader, RecordingProgress())


def test_search_dataset(
    datasets: DatasetService, repo: InMemoryDatasetRepository,
    chunks: List[Chunk],
) -> None:
    """Every question is searched and the results are written."""
    report = datasets.search_dataset("in/dataset.json", 2, "out")
    assert report.n_questions == 3
    assert report.output_path == "out/dataset.json"
    assert report.mode == "bm25"
    saved = repo.json("out/dataset.json")
    assert isinstance(saved, dict)
    assert saved["k"] == 2
    results = saved["search_results"]
    assert [r["question_id"] for r in results] == ["q1", "q2", "q3"]
    assert results[0]["retrieved_sources"][0] == (
        chunks[2].to_source().model_dump()
    )
    assert all(len(r["retrieved_sources"]) <= 2 for r in results)
    assert results[2]["retrieved_sources"] == []


def test_search_dataset_errors(datasets: DatasetService,
                               repo: InMemoryDatasetRepository) -> None:
    """Bad k, missing and malformed files are user-facing errors."""
    with pytest.raises(InvalidInputError):
        datasets.search_dataset("in/dataset.json", 0, "out")
    with pytest.raises(NotFoundError):
        datasets.search_dataset("missing.json", 2, "out")
    repo.files["bad.json"] = '{"rag_questions": [{"nope": 1}]}'
    with pytest.raises(CorruptedDataError):
        datasets.search_dataset("bad.json", 2, "out")


def test_answer_dataset(
    datasets: DatasetService, repo: InMemoryDatasetRepository
) -> None:
    """Sources are read back and answered; all sources are kept."""
    repo.files["res/r.json"] = results_json([
        [source("src.txt", 0, 10), source("src.txt", 10, 20),
         source("src.txt", 20, 30)],
        [source("gone.txt", 0, 10)],
        [],
    ])
    report = datasets.answer_dataset("res/r.json", "ans", k=2)
    assert report.n_questions == 3
    assert report.output_path == "ans/r.json"
    saved = repo.json("ans/r.json")
    assert isinstance(saved, dict)
    first, second, third = saved["search_results"]
    assert len(first["retrieved_sources"]) == 3
    assert first["answer"] == "answer to 'question 1' from 2 chunk(s)"
    assert second["answer"] == third["answer"]
    assert "do not contain enough information" in second["answer"]
    assert isinstance(datasets.sources, FakeSourceReader)
    assert [len(c) for c in datasets.sources.calls] == [2, 1, 0]
    assert isinstance(datasets.progress, RecordingProgress)
    assert ("Answering", 3) in datasets.progress.loops


def test_answer_dataset_default_k_uses_saved_k(
    datasets: DatasetService, repo: InMemoryDatasetRepository
) -> None:
    """Without k, the k stored in the results file limits the context."""
    many = [source("src.txt", i * 10, i * 10 + 10) for i in range(5)]
    repo.files["r.json"] = results_json([many], k=4)
    datasets.answer_dataset("r.json", "ans")
    assert isinstance(datasets.sources, FakeSourceReader)
    assert len(datasets.sources.calls[0]) == 4


def test_answer_dataset_bad_k(
    datasets: DatasetService, repo: InMemoryDatasetRepository
) -> None:
    """An explicit k must be positive."""
    repo.files["r.json"] = results_json([[]])
    with pytest.raises(InvalidInputError):
        datasets.answer_dataset("r.json", "ans", k=-1)


def test_evaluate(
    datasets: DatasetService, repo: InMemoryDatasetRepository
) -> None:
    """Recall is computed for the ks up to k, k included."""
    repo.files["r.json"] = results_json([
        [source("b.md", 0, 37)],
        [source("x.md", 0, 5), source("a.py", 40, 77)],
    ], k=2)
    report, warnings = datasets.evaluate(
        "r.json", "in/dataset.json", k=2, ks=(1, 3)
    )
    assert warnings == []
    assert report.recall == {1: 0.5, 2: 1.0}
    assert report.n_questions == 2
    assert report.n_missing == 0


def test_evaluate_default_ks(
    datasets: DatasetService, repo: InMemoryDatasetRepository
) -> None:
    """The default ks are 1, 3, 5, 10."""
    repo.files["r.json"] = results_json([[source("b.md", 0, 37)]])
    report, _ = datasets.evaluate("r.json", "in/dataset.json")
    assert list(report.recall) == [1, 3, 5, 10]
    assert report.n_missing == 1
    assert report.recall[10] == 0.5


def test_evaluate_warnings(
    datasets: DatasetService, repo: InMemoryDatasetRepository
) -> None:
    """Too-long sources and too many sources are reported."""
    repo.files["r.json"] = results_json([
        [source("b.md", 0, 5000)],
        [source("a.py", 0, 10), source("a.py", 10, 20),
         source("a.py", 20, 30)],
    ])
    _, warnings = datasets.evaluate(
        "r.json", "in/dataset.json", k=2, max_context_length=2000
    )
    assert len(warnings) == 2
    assert "1 source(s) longer than 2000" in warnings[0]
    assert "1 question(s) with more than 2 sources" in warnings[1]


def test_evaluate_errors(
    datasets: DatasetService, repo: InMemoryDatasetRepository
) -> None:
    """Invalid k, missing and mismatched files raise RagErrors."""
    with pytest.raises(InvalidInputError):
        datasets.evaluate("r.json", "in/dataset.json", k=0)
    with pytest.raises(NotFoundError):
        datasets.evaluate("missing.json", "in/dataset.json")
    with pytest.raises(CorruptedDataError):
        datasets.evaluate("in/dataset.json", "in/dataset.json")
