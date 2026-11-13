"""BEIR dataset adapter on a local archive (no network)."""

import json
import zipfile
from pathlib import Path

import pytest

from src.adapters.outbound.beir_repository import BeirRepository
from src.domain.errors import CorruptedDataError, NotFoundError


def make_archive(directory: Path, name: str = "tiny") -> None:
    """Write a three-document BEIR archive."""
    corpus = [
        {"_id": "D/1", "title": "Zinc", "text": "Zinc helps."},
        {"_id": "D2", "title": "", "text": "No title here."},
        {"_id": "D3", "text": "Unjudged."},
    ]
    queries = [
        {"_id": "q1", "text": "does zinc help"},
        {"_id": "q2", "text": "train only"},
    ]
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(directory / f"{name}.zip", "w") as archive:
        archive.writestr(
            f"{name}/corpus.jsonl",
            "\n".join(json.dumps(d) for d in corpus) + "\n",
        )
        archive.writestr(
            f"{name}/queries.jsonl",
            "\n".join(json.dumps(q) for q in queries),
        )
        archive.writestr(
            f"{name}/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nq1\tD/1\t1\nq1\tD2\t0\n",
        )


def test_load_writes_one_file_per_document(tmp_path: Path) -> None:
    """Documents become title + text files; only test queries are kept."""
    make_archive(tmp_path)
    data = BeirRepository(str(tmp_path)).load("tiny")
    files = Path(data.raw_dir)
    assert sorted(p.name for p in files.glob("*.txt")) == [
        "d0000000.txt", "d0000001.txt", "d0000002.txt",
    ]
    assert (files / "d0000000.txt").read_text() == "Zinc\n\nZinc helps."
    assert (files / "d0000001.txt").read_text() == "No title here."
    assert data.doc_ids == {
        "d0000000": "D/1", "d0000001": "D2", "d0000002": "D3",
    }
    assert data.queries == {"q1": "does zinc help"}
    assert data.qrels == {"q1": {"D/1": 1, "D2": 0}}


def test_load_twice_reuses_the_files(tmp_path: Path) -> None:
    """A second load does not rewrite the corpus."""
    make_archive(tmp_path)
    repository = BeirRepository(str(tmp_path))
    first = repository.load("tiny")
    marker = Path(first.raw_dir) / "d0000000.txt"
    marker.write_text("edited")
    assert repository.load("tiny").doc_ids == first.doc_ids
    assert marker.read_text() == "edited"


def test_invalid_names_and_archives_are_clean_errors(tmp_path: Path) -> None:
    """Path tricks and broken archives never escape as tracebacks."""
    repository = BeirRepository(str(tmp_path))
    with pytest.raises(NotFoundError):
        repository.load("../etc")
    (tmp_path / "broken.zip").write_text("not a zip")
    with pytest.raises(CorruptedDataError):
        repository.load("broken")
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
        archive.writestr("empty/readme.txt", "nothing")
    with pytest.raises(CorruptedDataError):
        repository.load("empty")


def test_coir_task_is_written_with_its_code_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python tasks become ``.py`` files; only test queries are kept."""
    from src.adapters.outbound.coir_repository import (
        BenchmarkCatalogue,
        CoirRepository,
    )

    tables = {
        "corpus": [
            {"_id": "d1", "title": "", "text": "def add(a, b):\n    return a"},
            {"_id": "d2", "title": "", "text": "def sub(a, b):\n    return b"},
        ],
        "queries": [
            {"_id": "q1", "text": "train question"},
            {"_id": "q2", "text": "add two numbers"},
        ],
        "qrels": [{"query-id": "q2", "corpus-id": "d1", "score": "1"}],
    }
    repository = CoirRepository(str(tmp_path / "coir"))
    monkeypatch.setattr(
        repository, "_rows", lambda name, member: tables[member]
    )
    data = repository.load("cosqa")
    assert data.name == "coir:cosqa"
    assert sorted(p.name for p in Path(data.raw_dir).glob("d*")) == [
        "d0000000.py", "d0000001.py",
    ]
    assert data.queries == {"q2": "add two numbers"}
    assert data.qrels == {"q2": {"d1": 1}}
    other = repository.load("stackoverflow-qa")
    assert list(Path(other.raw_dir).glob("*.txt"))
    with pytest.raises(NotFoundError):
        repository.load("../x")
    tables["qrels"] = [{"query": "q2"}]
    with pytest.raises(CorruptedDataError):
        repository.load("broken-layout")

    make_archive(tmp_path / "beir")
    catalogue = BenchmarkCatalogue(
        BeirRepository(str(tmp_path / "beir")), repository
    )
    tables["qrels"] = [{"query-id": "q2", "corpus-id": "d1", "score": "1"}]
    assert catalogue.load("coir:cosqa").name == "coir:cosqa"
    assert catalogue.load("tiny").name == "tiny"
