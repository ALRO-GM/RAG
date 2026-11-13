"""Incremental indexing end to end with the real adapters (no model)."""

from pathlib import Path

import pytest

from src.adapters.outbound.fs_corpus import FileSystemCorpus
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.application.search import SearchService
from src.domain.errors import NotFoundError
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder
from tests.integration.conftest import INDEX_DIR, real_indexing
from tests.samples import CORPUS, MD_PATH, PY_PATH, TXT_PATH


def search_service() -> SearchService:
    """Return a BM25 SearchService over the on-disk index."""
    return SearchService(FileSystemIndexRepository(INDEX_DIR), tokenize,
                         FakeEmbedder())


def test_full_then_incremental(project: Path) -> None:
    """A second run reuses every unchanged file."""
    first = real_indexing().run(max_chunk_size=300)
    assert first.incremental is False
    assert (first.files_total, first.files_reindexed) == (3, 3)
    for name in ("manifest.json", "chunks.jsonl", "bm25_weights.npz",
                 "bm25_vocabulary.json", "records.pkl"):
        assert (project / INDEX_DIR / name).is_file()
    assert not (project / INDEX_DIR / "embeddings.npy").exists()
    second = real_indexing().run(max_chunk_size=300)
    assert second.incremental is True
    assert (second.files_reindexed, second.files_reused) == (0, 3)
    assert second.chunks == first.chunks


def test_modified_added_and_removed_files(project: Path) -> None:
    """Edits, additions and deletions are reflected in the index."""
    real_indexing().run(max_chunk_size=300)
    fp_before = FileSystemIndexRepository(INDEX_DIR).load().manifest
    (project / TXT_PATH).write_text(
        CORPUS[TXT_PATH] + "\nChunked prefill splits long prompts.\n"
    )
    (project / MD_PATH).unlink()
    (project / "data/raw/proj/new.rst").write_text(
        "Pipeline parallelism\n====================\n\nStages.\n"
    )
    report = real_indexing().run(max_chunk_size=300)
    assert report.files_total == 3
    assert report.files_reindexed == 2
    assert report.files_reused == 1
    assert report.files_removed == 1
    snapshot = FileSystemIndexRepository(INDEX_DIR).load()
    assert snapshot.manifest.fingerprint != fp_before.fingerprint
    paths = {c.file_path for c in snapshot.chunks}
    assert paths == {PY_PATH, TXT_PATH, "data/raw/proj/new.rst"}
    hits = search_service().search("chunked prefill", 1)
    assert hits[0].chunk.file_path == TXT_PATH


def test_force_and_new_chunk_size(project: Path) -> None:
    """``force`` or another chunk size rebuild every file."""
    real_indexing().run(max_chunk_size=300)
    assert real_indexing().run(300, force=True).files_reindexed == 3
    report = real_indexing().run(max_chunk_size=100)
    assert report.files_reindexed == 3
    snapshot = FileSystemIndexRepository(INDEX_DIR).load()
    assert snapshot.manifest.max_chunk_size == 100
    assert all(
        c.last_character_index - c.first_character_index <= 100
        for c in snapshot.chunks
    )


def test_chunks_point_into_files(indexed_project: Path) -> None:
    """Every chunk is an exact slice of its source file."""
    snapshot = FileSystemIndexRepository(INDEX_DIR).load()
    for chunk in snapshot.chunks:
        text = (indexed_project / chunk.file_path).read_text()
        assert text[
            chunk.first_character_index:chunk.last_character_index
        ] == chunk.text


@pytest.mark.parametrize("query, path", [
    ("load_lora_adapter", PY_PATH),
    ("KV cache blocks allocate", PY_PATH),
    ("FP8 quantization memory", MD_PATH),
    ("speculative decoding draft", TXT_PATH),
])
def test_search_after_indexing(
    indexed_project: Path, query: str, path: str
) -> None:
    """BM25 retrieves the relevant file first."""
    assert search_service().search(query, 3)[0].chunk.file_path == path


def test_missing_and_empty_corpus(project: Path) -> None:
    """No corpus directory or no indexable file is a NotFoundError."""
    service = real_indexing()
    service.corpus = FileSystemCorpus("data/missing")
    with pytest.raises(NotFoundError, match="not found"):
        service.run()
    (project / "data/empty").mkdir()
    service.corpus = FileSystemCorpus("data/empty")
    with pytest.raises(NotFoundError, match="No indexable files"):
        service.run()
    assert not (project / INDEX_DIR).exists()
