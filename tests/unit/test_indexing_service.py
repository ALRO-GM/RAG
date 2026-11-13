"""IndexingService with in-memory fakes of every port."""

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pytest

from src.application.indexing import INDEX_FORMAT_VERSION, IndexingService
from src.domain.errors import InvalidInputError, NotFoundError
from tests.fakes import (
    FakeCorpus,
    FakeEmbedder,
    FakeLexicalBuilder,
    FakeProcessor,
    InMemoryIndexRepository,
    RecordingProgress,
)
from tests.samples import MD_PATH, PY_PATH, TXT_PATH

Wired = Tuple[IndexingService, FakeCorpus, FakeProcessor,
              InMemoryIndexRepository]


def wire(
    files: Dict[str, str],
    embedder: Optional[FakeEmbedder] = None,
    repository: Optional[InMemoryIndexRepository] = None,
    embed_style: str = "full",
    extra_params: Optional[Mapping[str, object]] = None,
) -> Wired:
    """Build a service over fakes and return it with its fakes."""
    corpus = FakeCorpus(files)
    processor = FakeProcessor(corpus)
    repo = repository or InMemoryIndexRepository()
    service = IndexingService(
        corpus=corpus,
        processor=processor,
        lexical_builder=FakeLexicalBuilder(),
        repository=repo,
        embedder=embedder,
        progress=RecordingProgress(),
        embed_style=embed_style,
        extra_params=extra_params,
    )
    return service, corpus, processor, repo


def test_full_run(corpus_files: Dict[str, str]) -> None:
    """A first run processes every file and saves a lexical index."""
    service, _, processor, repo = wire(corpus_files)
    report = service.run(max_chunk_size=500)
    assert report.files_total == 3
    assert report.files_reindexed == 3
    assert report.files_reused == 0
    assert report.files_removed == 0
    assert report.incremental is False
    assert processor.calls == [sorted(corpus_files)]
    snapshot = repo.snapshot
    assert snapshot is not None
    assert report.chunks == len(snapshot.chunks) > 0
    assert snapshot.embeddings is None
    manifest = snapshot.manifest
    assert manifest.version == INDEX_FORMAT_VERSION
    assert manifest.semantic is False
    assert manifest.embedding_model == ""
    assert manifest.n_files == 3
    assert manifest.n_chunks == len(snapshot.chunks)
    assert manifest.max_chunk_size == 500
    paths = [c.file_path for c in snapshot.chunks]
    assert paths == sorted(paths), "chunks follow the corpus order"


def test_lexical_index_aligned_with_chunks(
    corpus_files: Dict[str, str]
) -> None:
    """The lexical index scores exactly one row per chunk."""
    service, _, _, repo = wire(corpus_files)
    service.run(max_chunk_size=200)
    assert repo.snapshot is not None
    scores = repo.snapshot.lexical.scores(["lora"])
    assert scores.shape == (len(repo.snapshot.chunks),)
    best = repo.snapshot.chunks[int(np.argmax(scores))]
    assert "lora" in best.search_text().lower()


def test_second_run_reuses_unchanged_files(
    corpus_files: Dict[str, str]
) -> None:
    """Unchanged files are not reprocessed."""
    service, _, processor, repo = wire(corpus_files)
    first = service.run(max_chunk_size=500)
    assert repo.snapshot is not None
    fingerprint = repo.snapshot.manifest.fingerprint
    second = service.run(max_chunk_size=500)
    assert second.incremental is True
    assert second.files_reindexed == 0
    assert second.files_reused == 3
    assert second.chunks == first.chunks
    assert processor.calls[-1] == []
    assert repo.snapshot.manifest.fingerprint == fingerprint


def test_changed_file_is_reprocessed(corpus_files: Dict[str, str]) -> None:
    """Only a modified file is re-chunked; the fingerprint changes."""
    service, corpus, processor, repo = wire(corpus_files)
    service.run(max_chunk_size=500)
    assert repo.snapshot is not None
    old_fp = repo.snapshot.manifest.fingerprint
    corpus.files[TXT_PATH] += "\nPrefix caching reuses KV blocks.\n"
    report = service.run(max_chunk_size=500)
    assert report.files_reindexed == 1
    assert report.files_reused == 2
    assert processor.calls[-1] == [TXT_PATH]
    assert repo.snapshot.manifest.fingerprint != old_fp
    assert any("Prefix caching" in c.text for c in repo.snapshot.chunks)


def test_removed_and_added_files(corpus_files: Dict[str, str]) -> None:
    """Deleted files are dropped and new files are processed."""
    service, corpus, processor, repo = wire(corpus_files)
    service.run(max_chunk_size=500)
    del corpus.files[MD_PATH]
    corpus.files["data/raw/proj/new.md"] = "# New\n\nBrand new page.\n"
    report = service.run(max_chunk_size=500)
    assert report.files_total == 3
    assert report.files_removed == 1
    assert report.files_reindexed == 1
    assert processor.calls[-1] == ["data/raw/proj/new.md"]
    assert repo.snapshot is not None
    paths = {c.file_path for c in repo.snapshot.chunks}
    assert MD_PATH not in paths
    assert "data/raw/proj/new.md" in paths
    assert set(repo.records) == set(corpus.files)


def test_force_reprocesses_everything(corpus_files: Dict[str, str]) -> None:
    """``force`` ignores cached records."""
    service, _, processor, _ = wire(corpus_files)
    service.run(max_chunk_size=500)
    report = service.run(max_chunk_size=500, force=True)
    assert report.files_reindexed == 3
    assert report.incremental is False
    assert processor.calls[-1] == sorted(corpus_files)


def test_changed_parameters_invalidate_records(
    corpus_files: Dict[str, str]
) -> None:
    """A different chunk size rebuilds every file."""
    service, _, processor, _ = wire(corpus_files)
    service.run(max_chunk_size=500)
    report = service.run(max_chunk_size=300)
    assert report.files_reindexed == 3
    assert report.incremental is False
    report = service.run(max_chunk_size=300, target_chunk_size=100)
    assert report.files_reindexed == 3


def test_empty_corpus_raises() -> None:
    """An empty corpus is a user-facing error."""
    service, _, _, repo = wire({})
    with pytest.raises(NotFoundError, match="No indexable files"):
        service.run()
    assert repo.saves == 0


@pytest.mark.parametrize("size", [0, 49, 2001, 5000])
def test_invalid_chunk_size(corpus_files: Dict[str, str], size: int) -> None:
    """Chunk sizes outside [50, 2000] are rejected before any work."""
    service, _, processor, _ = wire(corpus_files)
    with pytest.raises(InvalidInputError):
        service.run(max_chunk_size=size)
    assert processor.calls == []


def test_with_embedder(corpus_files: Dict[str, str]) -> None:
    """A fake embedder yields one normalised row per chunk."""
    embedder = FakeEmbedder("emb-a")
    service, _, _, repo = wire(corpus_files, embedder)
    report = service.run(max_chunk_size=200)
    snapshot = repo.snapshot
    assert snapshot is not None and snapshot.embeddings is not None
    assert snapshot.embeddings.shape == (report.chunks, embedder.dim)
    assert snapshot.embeddings.dtype == np.float32
    assert snapshot.manifest.semantic is True
    assert snapshot.manifest.embedding_model == "emb-a"
    assert embedder.encoded == [c.search_text() for c in snapshot.chunks]
    expected = embedder.encode([c.search_text() for c in snapshot.chunks])
    np.testing.assert_allclose(snapshot.embeddings, expected)


def test_incremental_embeds_only_changed_chunks(
    corpus_files: Dict[str, str]
) -> None:
    """Reused files keep their embeddings; only new chunks are encoded."""
    embedder = FakeEmbedder()
    service, corpus, _, repo = wire(corpus_files, embedder)
    service.run(max_chunk_size=200)
    corpus.files[PY_PATH] += "\n\ndef added():\n    return 42\n"
    embedder.encoded.clear()
    service.run(max_chunk_size=200)
    snapshot = repo.snapshot
    assert snapshot is not None and snapshot.embeddings is not None
    py_chunks = [c for c in snapshot.chunks if c.file_path == PY_PATH]
    assert embedder.encoded == [c.search_text() for c in py_chunks]
    assert snapshot.embeddings.shape[0] == len(snapshot.chunks)
    expected = embedder.encode([c.search_text() for c in snapshot.chunks])
    np.testing.assert_allclose(snapshot.embeddings, expected)


def test_embedder_switch_invalidates_records(
    corpus_files: Dict[str, str]
) -> None:
    """Records built without (or with another) embedder are not reused."""
    repo = InMemoryIndexRepository()
    lexical, _, _, _ = wire(corpus_files, None, repo)
    lexical.run(max_chunk_size=500)
    assert repo.snapshot is not None
    fp_lexical = repo.snapshot.manifest.fingerprint
    semantic, _, _, _ = wire(corpus_files, FakeEmbedder("a"), repo)
    report = semantic.run(max_chunk_size=500)
    assert report.files_reindexed == 3
    assert repo.snapshot.embeddings is not None
    fp_a = repo.snapshot.manifest.fingerprint
    other, _, _, _ = wire(corpus_files, FakeEmbedder("b"), repo)
    assert other.run(max_chunk_size=500).files_reindexed == 3
    assert len({fp_lexical, fp_a, repo.snapshot.manifest.fingerprint}) == 3


def test_params() -> None:
    """Parameters include everything that changes per-file records."""
    service, _, _, _ = wire({}, FakeEmbedder("m"))
    assert service.params(100, 50) == {
        "version": INDEX_FORMAT_VERSION,
        "max_chunk_size": 100,
        "target_chunk_size": 50,
        "embedding_model": "m",
        "embed_style": "full",
    }
    lexical, _, _, _ = wire({})
    assert lexical.params(100, None)["embedding_model"] == ""


def test_fingerprint_is_deterministic(corpus_files: Dict[str, str]) -> None:
    """The same corpus and settings give the same fingerprint."""
    a, _, _, repo_a = wire(corpus_files)
    b, _, _, repo_b = wire(dict(reversed(list(corpus_files.items()))))
    a.run(max_chunk_size=500)
    b.run(max_chunk_size=500)
    assert repo_a.snapshot is not None and repo_b.snapshot is not None
    assert (repo_a.snapshot.manifest.fingerprint
            == repo_b.snapshot.manifest.fingerprint)


def test_progress_is_reported(corpus_files: Dict[str, str]) -> None:
    """Long loops go through the progress port."""
    service, _, _, _ = wire(corpus_files)
    service.run(max_chunk_size=500)
    assert isinstance(service.progress, RecordingProgress)
    assert ("Checking files", 3) in service.progress.loops
    assert ("Chunking files", 3) in service.progress.loops


def fingerprint(repo: InMemoryIndexRepository) -> str:
    """Return the fingerprint of the saved index."""
    assert repo.snapshot is not None
    return repo.snapshot.manifest.fingerprint


def test_params_include_embed_style_and_extra_params() -> None:
    """Adapter settings are listed next to the core parameters."""
    extra = {"bm25_k1": 0.9, "path_weight": 2}
    service, _, _, _ = wire({}, embed_style="body", extra_params=extra)
    params = service.params(100, None)
    assert params["embed_style"] == "body"
    assert params["bm25_k1"] == 0.9
    assert params["path_weight"] == 2
    assert params["max_chunk_size"] == 100
    extra["bm25_k1"] = 2.0
    assert service.params(100, None)["bm25_k1"] == 0.9, "copied at wiring"


def test_embed_style_body_embeds_context_and_text_only(
    corpus_files: Dict[str, str]
) -> None:
    """The body style leaves the file path out of the embedded text."""
    embedder = FakeEmbedder()
    service, _, _, repo = wire(corpus_files, embedder, embed_style="body")
    service.run(max_chunk_size=200)
    assert repo.snapshot is not None
    chunks = repo.snapshot.chunks
    assert embedder.encoded == [f"{c.context}\n{c.text}" for c in chunks]
    assert not any(t.startswith("data/raw") for t in embedder.encoded)
    assert repo.snapshot.embeddings is not None
    np.testing.assert_allclose(
        repo.snapshot.embeddings,
        embedder.encode([c.embedding_text("body") for c in chunks]),
    )


def test_changing_embed_style_reindexes(corpus_files: Dict[str, str]) -> None:
    """Records embedded with another style are not reused."""
    repo = InMemoryIndexRepository()
    full, _, _, _ = wire(corpus_files, FakeEmbedder(), repo)
    full.run(max_chunk_size=500)
    first = fingerprint(repo)
    body, _, processor, _ = wire(corpus_files, FakeEmbedder(), repo,
                                 embed_style="body")
    report = body.run(max_chunk_size=500)
    assert (report.files_reindexed, report.incremental) == (3, False)
    assert processor.calls == [sorted(corpus_files)]
    assert fingerprint(repo) != first
    assert body.run(max_chunk_size=500).files_reused == 3


@pytest.mark.parametrize("changed", [
    {"bm25_k1": 1.5}, {"bm25_k1": 1.2, "path_weight": 2}, {},
])
def test_changing_extra_params_reindexes(
    corpus_files: Dict[str, str], changed: Dict[str, object]
) -> None:
    """A changed, added or removed adapter setting rebuilds every file."""
    repo = InMemoryIndexRepository()
    first, _, _, _ = wire(corpus_files, None, repo,
                          extra_params={"bm25_k1": 1.2})
    first.run(max_chunk_size=500)
    before = fingerprint(repo)
    second, _, _, _ = wire(corpus_files, None, repo, extra_params=changed)
    report = second.run(max_chunk_size=500)
    assert report.files_reindexed == 3
    assert report.incremental is False
    assert fingerprint(repo) != before


def test_same_extra_params_reuse_records(corpus_files: Dict[str, str]) -> None:
    """Identical adapter settings keep the incremental behaviour."""
    repo = InMemoryIndexRepository()
    extra = {"bm25_k1": 1.2, "table_context": True}
    first, _, _, _ = wire(corpus_files, None, repo, extra_params=extra)
    first.run(max_chunk_size=500)
    before = fingerprint(repo)
    again, _, processor, _ = wire(corpus_files, None, repo,
                                  extra_params=dict(extra))
    report = again.run(max_chunk_size=500)
    assert (report.files_reused, report.incremental) == (3, True)
    assert processor.calls == [[]]
    assert fingerprint(repo) == before
