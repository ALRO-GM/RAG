"""Use case: build or incrementally update the index."""

import hashlib
import json
import time
from typing import Dict, List, Mapping, Optional

import numpy as np

from src.application.ports import (
    CorpusReader,
    DocumentProcessor,
    Embedder,
    FileRecord,
    IndexRepository,
    IndexSnapshot,
    LexicalIndexBuilder,
    NoProgress,
    Progress,
)
from src.domain.errors import NotFoundError
from src.domain.models import IndexManifest, IndexReport
from src.domain.validation import validate_chunk_size

# Bump whenever chunking or text analysis changes: cached file records
# and persisted indexes built by older code are then rebuilt.
INDEX_FORMAT_VERSION = 3


class IndexingService:
    """Index a corpus, reprocessing only new or modified files.

    Every file is hashed; files whose hash matches the previous run
    reuse their chunks, term counts and embeddings. Only new or changed
    files are re-chunked and re-embedded, and deleted files are dropped.
    Lexical statistics (IDF, average length) are corpus-wide, so the
    lexical index is rebuilt from the cached term counts, which is cheap.
    """

    def __init__(
        self,
        corpus: CorpusReader,
        processor: DocumentProcessor,
        lexical_builder: LexicalIndexBuilder,
        repository: IndexRepository,
        embedder: Optional[Embedder] = None,
        progress: Optional[Progress] = None,
        embed_style: str = "full",
        extra_params: Optional[Mapping[str, object]] = None,
    ):
        """Wire the ports; ``embedder=None`` builds a lexical-only index.

        ``extra_params`` describes adapter settings that change what is
        indexed (field weights, BM25 constants...). They are part of
        the index fingerprint, so changing one rebuilds the index.
        """
        self.corpus = corpus
        self.processor = processor
        self.lexical_builder = lexical_builder
        self.repository = repository
        self.embedder = embedder
        self.progress = progress or NoProgress()
        self.embed_style = embed_style
        self.extra_params = dict(extra_params or {})

    def params(
        self, max_chunk_size: int, target_chunk_size: Optional[int]
    ) -> Dict[str, object]:
        """Return the settings that invalidate cached file records."""
        return {
            "version": INDEX_FORMAT_VERSION,
            "max_chunk_size": max_chunk_size,
            "target_chunk_size": target_chunk_size,
            "embedding_model": (
                self.embedder.model_name if self.embedder else ""
            ),
            "embed_style": self.embed_style,
            **self.extra_params,
        }

    def run(
        self,
        max_chunk_size: int = 2000,
        target_chunk_size: Optional[int] = None,
        force: bool = False,
    ) -> IndexReport:
        """Index the corpus, reusing unchanged files unless ``force``."""
        validate_chunk_size(max_chunk_size)
        start = time.perf_counter()
        params = self.params(max_chunk_size, target_chunk_size)
        files = self.corpus.list_files()
        if not files:
            raise NotFoundError("No indexable files found in the corpus.")
        previous = {} if force else self.repository.load_records(params)
        records: Dict[str, FileRecord] = {}
        todo: List[str] = []
        for path in self.progress.track(
            files, len(files), "Checking files"
        ):
            old = previous.get(path)
            if old is not None and old.sha256 == self.corpus.file_hash(path):
                records[path] = old
            else:
                todo.append(path)
        fresh = self._process(todo, max_chunk_size, target_chunk_size)
        self._embed(fresh)
        records.update(fresh)
        ordered = [records[p] for p in files]
        snapshot = self._assemble(ordered, records, params, max_chunk_size)
        self.repository.save(snapshot, records, params)
        return IndexReport(
            files_total=len(files),
            files_reindexed=len(fresh),
            files_reused=len(files) - len(fresh),
            files_removed=len(set(previous) - set(files)),
            chunks=len(snapshot.chunks),
            seconds=round(time.perf_counter() - start, 2),
            incremental=bool(previous),
        )

    def _process(
        self, paths: List[str], max_size: int, target: Optional[int]
    ) -> Dict[str, FileRecord]:
        """Chunk and analyse the files that need (re)indexing."""
        fresh: Dict[str, FileRecord] = {}
        documents = self.processor.process(paths, max_size, target)
        for doc in self.progress.track(
            documents, len(paths), "Chunking files"
        ):
            fresh[doc.file_path] = FileRecord(
                sha256=doc.sha256,
                chunks=doc.chunks,
                term_counts=doc.term_counts,
            )
        return fresh

    def _embed(self, fresh: Dict[str, FileRecord]) -> None:
        """Embed the chunks of new or modified files."""
        if self.embedder is None:
            return
        texts = [
            c.embedding_text(self.embed_style)
            for r in fresh.values() for c in r.chunks
        ]
        if not texts:
            return
        vectors = self.embedder.encode(texts, progress=True)
        offset = 0
        for record in fresh.values():
            n = len(record.chunks)
            record.embeddings = vectors[offset:offset + n]
            offset += n

    def _assemble(
        self,
        ordered: List[FileRecord],
        records: Dict[str, FileRecord],
        params: Dict[str, object],
        max_chunk_size: int,
    ) -> IndexSnapshot:
        """Merge per-file records into one searchable snapshot."""
        chunks = [c for r in ordered for c in r.chunks]
        lexical = self.lexical_builder.build(
            [tc for r in ordered for tc in r.term_counts]
        )
        embeddings = None
        if self.embedder is not None:
            parts = [r.embeddings for r in ordered if r.embeddings is not None]
            if parts:
                embeddings = np.vstack(parts).astype(np.float32)
        hashes = {path: rec.sha256 for path, rec in records.items()}
        fingerprint = hashlib.sha256(
            json.dumps([params, hashes], sort_keys=True).encode()
        ).hexdigest()[:16]
        manifest = IndexManifest(
            version=INDEX_FORMAT_VERSION,
            max_chunk_size=max_chunk_size,
            semantic=embeddings is not None,
            embedding_model=self.embedder.model_name if self.embedder else "",
            fingerprint=fingerprint,
            n_chunks=len(chunks),
            n_files=len(ordered),
        )
        return IndexSnapshot(manifest, chunks, lexical, embeddings)
