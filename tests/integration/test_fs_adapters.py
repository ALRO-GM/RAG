"""Filesystem corpus, parallel document processor and source reader."""

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from src.adapters.outbound.fs_corpus import FileSystemCorpus
from src.adapters.outbound.fs_source_reader import FileSystemSourceReader
from src.adapters.outbound.parallel_processor import (
    ParallelDocumentProcessor,
    field_term_counts,
    process_file,
)
from src.domain.chunking import default_registry
from src.domain.errors import NotFoundError
from src.domain.models import MinimalSource
from src.domain.text_analysis import tokenize
from tests.fakes import make_chunk
from tests.samples import CORPUS, MD_PATH, PY_PATH, TXT_PATH


def src(path: str, first: int, last: int) -> MinimalSource:
    """Return a source."""
    return MinimalSource(file_path=path, first_character_index=first,
                         last_character_index=last)


# ------------------------------------------------------------- corpus


def test_corpus_lists_relative_sorted_paths(project: Path) -> None:
    """Relative, deterministic (walk order); .json and hidden skipped."""
    corpus = FileSystemCorpus("data/raw")
    assert corpus.list_files() == [PY_PATH, TXT_PATH, MD_PATH]


def test_corpus_extension_filter(project: Path) -> None:
    """Only the requested extensions are listed (case-insensitive)."""
    (project / "data/raw/proj/UPPER.PY").write_text("x = 1\n")
    corpus = FileSystemCorpus("data/raw", (".PY",))
    assert corpus.list_files() == ["data/raw/proj/UPPER.PY", PY_PATH]
    json_only = FileSystemCorpus("data/raw", (".json",))
    assert json_only.list_files() == ["data/raw/proj/config.json"]


def test_corpus_hash(project: Path) -> None:
    """The hash is the SHA-256 of the file bytes."""
    expected = hashlib.sha256(CORPUS[PY_PATH].encode()).hexdigest()
    assert FileSystemCorpus("data/raw").file_hash(PY_PATH) == expected


def test_corpus_missing_directory(project: Path) -> None:
    """A missing corpus directory is a NotFoundError."""
    with pytest.raises(NotFoundError, match="Corpus directory not found"):
        FileSystemCorpus("data/nowhere").list_files()


def test_corpus_empty_directory(tmp_path: Path) -> None:
    """An empty directory lists nothing."""
    assert FileSystemCorpus(str(tmp_path)).list_files() == []


# ---------------------------------------------------------- processor


def test_process_file(project: Path) -> None:
    """One file is read, hashed, chunked and analysed."""
    doc = process_file(
        (PY_PATH, 300, None, default_registry, tokenize, 1, 1)
    )
    assert doc.file_path == PY_PATH
    assert doc.sha256 == hashlib.sha256(CORPUS[PY_PATH].encode()).hexdigest()
    assert len(doc.term_counts) == len(doc.chunks) > 0
    assert all(c.kind == "python" for c in doc.chunks)
    assert "lora" in doc.term_counts[0]
    for chunk in doc.chunks:
        assert CORPUS[PY_PATH][
            chunk.first_character_index:chunk.last_character_index
        ] == chunk.text


def test_process_file_invalid_utf8(project: Path) -> None:
    """Undecodable bytes are replaced instead of failing."""
    path = project / "data/raw/proj/latin1.txt"
    path.write_bytes("café crème\n".encode("latin-1"))
    doc = process_file(
        ("data/raw/proj/latin1.txt", 300, None, default_registry, tokenize,
         1, 1)
    )
    assert "�" in doc.chunks[0].text


@pytest.mark.parametrize("workers", [1, 2])
def test_processor_preserves_order(project: Path, workers: int) -> None:
    """Results follow the input order, in-process or with a pool."""
    extra = []
    for i in range(10):
        rel = f"data/raw/proj/many/file_{i:02d}.md"
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# File {i}\n\nunique_term_{i} content\n")
        extra.append(rel)
    paths = [TXT_PATH, *extra, PY_PATH, MD_PATH]
    processor = ParallelDocumentProcessor(default_registry, tokenize,
                                          workers)
    docs = list(processor.process(paths, 300, None))
    assert [d.file_path for d in docs] == paths
    assert "unique_term_3" in docs[4].term_counts[0]


def test_processor_empty_input() -> None:
    """No path yields no document."""
    processor = ParallelDocumentProcessor(default_registry, tokenize, 2)
    assert list(processor.process([], 300, None)) == []
    assert processor.workers == 2


FIELDS = make_chunk("pkg/engine.py", 0, "allocate blocks for the engine",
                    "class Scheduler")


def test_field_term_counts_default_weights_count_like_search_text() -> None:
    """With weights of 1 every field counts like the body."""
    assert field_term_counts(FIELDS, tokenize, 1, 1) == dict(
        Counter(tokenize(FIELDS.search_text()))
    )
    assert field_term_counts(FIELDS, tokenize, 1, 1)["engine"] == 2


def test_field_term_counts_weights_multiply_path_and_context() -> None:
    """A field weight counts each of its terms that many times."""
    counts = field_term_counts(FIELDS, tokenize, 3, 2)
    assert counts["pkg"] == 3
    assert counts["engine"] == 1 + 3
    assert counts["scheduler"] == 2
    assert counts["allocate"] == 1


def test_field_term_counts_zero_weight_ignores_the_field() -> None:
    """A weight of 0 leaves only the other fields searchable."""
    no_path = field_term_counts(FIELDS, tokenize, 0, 1)
    assert "pkg" not in no_path and no_path["engine"] == 1
    assert no_path["scheduler"] == 1
    body_only = field_term_counts(FIELDS, tokenize, 0, 0)
    assert body_only == dict(Counter(tokenize(FIELDS.text)))


def test_processor_applies_field_weights(project: Path) -> None:
    """The processor weights reach the term counts of every chunk."""
    plain = ParallelDocumentProcessor(default_registry, tokenize, 1)
    assert (plain.path_weight, plain.context_weight) == (1, 1)
    weighted = ParallelDocumentProcessor(
        default_registry, tokenize, 1, path_weight=3, context_weight=0
    )
    [base] = plain.process([MD_PATH], 300, None)
    [doc] = weighted.process([MD_PATH], 300, None)
    assert doc.chunks == base.chunks
    for chunk, counts in zip(doc.chunks, doc.term_counts):
        assert counts == field_term_counts(chunk, tokenize, 3, 0)
    assert doc.term_counts[0]["proj"] == 3 * base.term_counts[0]["proj"]


# ------------------------------------------------------- source reader


def test_source_reader_reads_ranges(project: Path) -> None:
    """Ranges are resolved against the file text."""
    chunks = FileSystemSourceReader().read([src(MD_PATH, 0, 9)])
    assert len(chunks) == 1
    assert chunks[0].text == "# Serving"
    assert chunks[0].kind == "source"


def test_source_reader_skips_missing_files(project: Path) -> None:
    """Missing files are skipped, the rest is kept in order."""
    chunks = FileSystemSourceReader().read([
        src("data/raw/proj/gone.md", 0, 10),
        src(TXT_PATH, 0, 11),
        src(PY_PATH, 0, 3),
    ])
    assert [c.file_path for c in chunks] == [TXT_PATH, PY_PATH]
    assert chunks[0].text == "Speculative"


def test_source_reader_clamps_ranges(project: Path) -> None:
    """Out-of-bound ranges are clamped; empty ranges are skipped."""
    size = len(CORPUS[TXT_PATH])
    chunks = FileSystemSourceReader().read([
        src(TXT_PATH, -5, 11),
        src(TXT_PATH, size - 7, size + 500),
        src(TXT_PATH, 50, 50),
        src(TXT_PATH, 60, 40),
        src(TXT_PATH, size + 10, size + 20),
    ])
    assert len(chunks) == 2
    assert chunks[0].first_character_index == 0
    assert chunks[0].text == "Speculative"
    assert chunks[1].last_character_index == size
    assert chunks[1].text == CORPUS[TXT_PATH][size - 7:]


def test_source_reader_directory_is_skipped(project: Path) -> None:
    """A path that is a directory is not readable and is skipped."""
    assert FileSystemSourceReader().read([src("data/raw", 0, 5)]) == []
