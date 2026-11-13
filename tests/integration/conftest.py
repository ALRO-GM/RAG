"""Fixtures wiring the real outbound adapters on the tiny corpus."""

from pathlib import Path

import pytest

from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.fs_corpus import FileSystemCorpus
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.parallel_processor import (
    ParallelDocumentProcessor,
)
from src.application.indexing import IndexingService
from src.domain.chunking import default_registry
from src.domain.text_analysis import tokenize

INDEX_DIR = "data/processed"


def real_indexing(index_dir: str = INDEX_DIR) -> IndexingService:
    """Return an IndexingService wired with real, lexical-only adapters."""
    return IndexingService(
        corpus=FileSystemCorpus("data/raw"),
        processor=ParallelDocumentProcessor(default_registry, tokenize, 1),
        lexical_builder=BM25Builder(),
        repository=FileSystemIndexRepository(index_dir),
        embedder=None,
    )


@pytest.fixture()
def indexed_project(project: Path) -> Path:
    """Index the tiny corpus (BM25 only) under ``data/processed``."""
    real_indexing().run(max_chunk_size=300)
    return project
