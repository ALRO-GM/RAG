"""FileSystemIndexRepository: round-trip, records and corruption."""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from src.adapters.outbound.bm25_index import BM25Index
from src.adapters.outbound.fs_index_repository import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    MANIFEST_FILE,
    RECORDS_FILE,
    VOCAB_FILE,
    FileSystemIndexRepository,
)
from src.application.indexing import INDEX_FORMAT_VERSION
from src.application.ports import FileRecord, IndexSnapshot
from src.domain.errors import CorruptedDataError, NotFoundError
from src.domain.models import Chunk, IndexManifest
from tests.fakes import FakeLexicalIndex

PARAMS: Dict[str, object] = {"version": INDEX_FORMAT_VERSION, "x": 1}


def snapshot(
    chunks: List[Chunk], semantic: bool, fingerprint: str
) -> IndexSnapshot:
    """Return a snapshot of ``chunks`` with a BM25 index."""
    counts = [{w: 1 for w in c.text.lower().split()} for c in chunks]
    embeddings = None
    if semantic:
        embeddings = np.eye(len(chunks), 8, dtype=np.float32)
    manifest = IndexManifest(
        version=INDEX_FORMAT_VERSION, max_chunk_size=2000,
        semantic=semantic, embedding_model="m" if semantic else "",
        fingerprint=fingerprint, n_chunks=len(chunks), n_files=2,
    )
    return IndexSnapshot(manifest, chunks, BM25Index.build(counts),
                         embeddings)


def records(chunks: List[Chunk]) -> Dict[str, FileRecord]:
    """Return one record per file."""
    return {
        "a.py": FileRecord("h1", chunks[:2], [{"x": 1}, {"y": 2}]),
        "b.md": FileRecord("h2", chunks[2:], [{"z": 1}] * 2,
                           np.ones((2, 8), dtype=np.float32)),
    }


@pytest.mark.parametrize("semantic", [False, True])
def test_round_trip(
    tmp_path: Path, chunks: List[Chunk], semantic: bool
) -> None:
    """A saved snapshot loads back identically."""
    repo = FileSystemIndexRepository(str(tmp_path / "idx"))
    original = snapshot(chunks, semantic, f"rt-{semantic}")
    repo.save(original, records(chunks), PARAMS)
    loaded = repo.load()
    assert loaded.manifest == original.manifest
    assert loaded.chunks == chunks
    query = ["kvcachemanager:", "fp8"]
    np.testing.assert_allclose(
        loaded.lexical.scores(query), original.lexical.scores(query)
    )
    if semantic:
        assert loaded.embeddings is not None
        assert original.embeddings is not None
        np.testing.assert_array_equal(loaded.embeddings, original.embeddings)
    else:
        assert loaded.embeddings is None
    assert repo.load() is loaded, "loads are cached per fingerprint"


def test_lexical_save_removes_stale_embeddings(
    tmp_path: Path, chunks: List[Chunk]
) -> None:
    """Saving a lexical index deletes embeddings from a previous run."""
    repo = FileSystemIndexRepository(str(tmp_path))
    repo.save(snapshot(chunks, True, "s1"), {}, PARAMS)
    assert (tmp_path / EMBEDDINGS_FILE).exists()
    repo.save(snapshot(chunks, False, "s2"), {}, PARAMS)
    assert not (tmp_path / EMBEDDINGS_FILE).exists()
    assert repo.load().embeddings is None


def test_records_round_trip(tmp_path: Path, chunks: List[Chunk]) -> None:
    """Records come back only for identical params."""
    repo = FileSystemIndexRepository(str(tmp_path))
    assert repo.load_records(PARAMS) == {}
    repo.save(snapshot(chunks, False, "r"), records(chunks), PARAMS)
    loaded = repo.load_records(dict(PARAMS))
    assert set(loaded) == {"a.py", "b.md"}
    assert loaded["a.py"].chunks == chunks[:2]
    assert loaded["a.py"].embeddings is None
    emb = loaded["b.md"].embeddings
    assert emb is not None and emb.shape == (2, 8)
    assert repo.load_records({**PARAMS, "x": 2}) == {}


@pytest.mark.parametrize("payload", [
    b"not a pickle",
    b"",
    pickle.dumps(["a", "list"]),
    pickle.dumps({"params": PARAMS, "records": "oops"}),
])
def test_corrupted_records_are_ignored(tmp_path: Path, payload: bytes) -> None:
    """Unreadable records only disable incremental indexing."""
    (tmp_path / RECORDS_FILE).write_bytes(payload)
    assert FileSystemIndexRepository(str(tmp_path)).load_records(PARAMS) == {}


def test_save_rejects_other_lexical_index(
    tmp_path: Path, chunks: List[Chunk]
) -> None:
    """Only BM25 indexes can be persisted."""
    snap = snapshot(chunks, False, "t")
    snap.lexical = FakeLexicalIndex([])
    with pytest.raises(TypeError):
        FileSystemIndexRepository(str(tmp_path)).save(snap, {}, PARAMS)


def test_missing_index(tmp_path: Path) -> None:
    """A missing directory or manifest is a NotFoundError."""
    with pytest.raises(NotFoundError, match="Run the 'index' command"):
        FileSystemIndexRepository(str(tmp_path / "nowhere")).load()
    with pytest.raises(NotFoundError):
        FileSystemIndexRepository(str(tmp_path)).load()


@pytest.mark.parametrize("content", [
    "{not json",
    '{"version": 2}',
    "",
])
def test_corrupted_manifest(tmp_path: Path, content: str) -> None:
    """An unparsable or incomplete manifest is a CorruptedDataError."""
    (tmp_path / MANIFEST_FILE).write_text(content)
    with pytest.raises(CorruptedDataError, match="manifest"):
        FileSystemIndexRepository(str(tmp_path)).load()


def test_manifest_from_another_version(
    tmp_path: Path, chunks: List[Chunk]
) -> None:
    """An index built by another format version must be rebuilt."""
    repo = FileSystemIndexRepository(str(tmp_path))
    repo.save(snapshot(chunks, False, "v"), {}, PARAMS)
    data = json.loads((tmp_path / MANIFEST_FILE).read_text())
    data["version"] = INDEX_FORMAT_VERSION + 1
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(data))
    with pytest.raises(CorruptedDataError, match="another version"):
        repo.load()


def test_inconsistent_chunks(tmp_path: Path, chunks: List[Chunk]) -> None:
    """Fewer chunks than BM25 rows is detected."""
    repo = FileSystemIndexRepository(str(tmp_path))
    repo.save(snapshot(chunks, False, "inc1"), {}, PARAMS)
    lines = (tmp_path / CHUNKS_FILE).read_text().splitlines()
    (tmp_path / CHUNKS_FILE).write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(CorruptedDataError, match="inconsistent"):
        repo.load()


def test_inconsistent_embeddings(
    tmp_path: Path, chunks: List[Chunk]
) -> None:
    """Embeddings with the wrong number of rows are detected."""
    repo = FileSystemIndexRepository(str(tmp_path))
    repo.save(snapshot(chunks, True, "inc2"), {}, PARAMS)
    np.save(tmp_path / EMBEDDINGS_FILE, np.zeros((1, 8), np.float32))
    with pytest.raises(CorruptedDataError, match="inconsistent"):
        repo.load()


@pytest.mark.parametrize("name, content", [
    (CHUNKS_FILE, "{broken json\n"),
    (VOCAB_FILE, "[["),
    ("bm25_weights.npz", "garbage"),
])
def test_corrupted_artefacts(
    tmp_path: Path, chunks: List[Chunk], name: str, content: str
) -> None:
    """Broken index files are reported, never raised raw."""
    repo = FileSystemIndexRepository(str(tmp_path))
    repo.save(snapshot(chunks, False, f"bad-{name}"), {}, PARAMS)
    (tmp_path / name).write_text(content)
    with pytest.raises(CorruptedDataError):
        repo.load()


@pytest.mark.parametrize("name", [CHUNKS_FILE, VOCAB_FILE, EMBEDDINGS_FILE])
def test_missing_artefact(
    tmp_path: Path, chunks: List[Chunk], name: str
) -> None:
    """A missing index file is reported as corruption."""
    repo = FileSystemIndexRepository(str(tmp_path))
    repo.save(snapshot(chunks, True, f"missing-{name}"), {}, PARAMS)
    (tmp_path / name).unlink()
    with pytest.raises(CorruptedDataError):
        repo.load()
