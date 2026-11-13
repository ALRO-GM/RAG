"""Document processing adapter: chunk + analyse files in parallel."""

import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from src.application.ports import Analyzer
from src.domain.chunking import ChunkerRegistry
from src.domain.models import Chunk, ProcessedDocument

RegistryFactory = Callable[[], ChunkerRegistry]
_Job = Tuple[str, int, Optional[int], RegistryFactory, Analyzer, int, int]


def process_file(job: _Job) -> ProcessedDocument:
    """Read, hash, chunk and analyse one file (runs in a worker)."""
    path, max_size, target, registry_factory, analyzer, pw, cw = job
    with open(path, "rb") as handle:
        raw = handle.read()
    text = raw.decode("utf-8", errors="replace")
    chunks = registry_factory().chunk(path, text, max_size, target)
    return ProcessedDocument(
        file_path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        chunks=chunks,
        term_counts=[
            field_term_counts(c, analyzer, pw, cw) for c in chunks
        ],
    )


def field_term_counts(
    chunk: Chunk, analyzer: Analyzer, path_weight: int, context_weight: int
) -> Dict[str, int]:
    """Count the terms of a chunk, weighting its path and context.

    A weight of 1 counts a field like the body; 2 counts each of its
    terms twice (a match in the heading or the path is stronger
    evidence than a match in the body); 0 ignores the field.
    """
    counts: Counter[str] = Counter(analyzer(chunk.text))
    for weight, text in (
        (path_weight, chunk.file_path), (context_weight, chunk.context)
    ):
        if weight > 0:
            for term in analyzer(text):
                counts[term] += weight
    return dict(counts)


class ParallelDocumentProcessor:
    """Process files with a pool of worker processes.

    ``registry_factory`` and ``analyzer`` must be module-level callables
    so they can be sent to the worker processes. With ``workers=1`` the
    files are processed in-process (useful for tests and debugging).
    """

    def __init__(
        self,
        registry_factory: RegistryFactory,
        analyzer: Analyzer,
        workers: Optional[int] = None,
        path_weight: int = 1,
        context_weight: int = 1,
    ):
        """Configure the processor."""
        self.registry_factory = registry_factory
        self.analyzer = analyzer
        self.path_weight = path_weight
        self.context_weight = context_weight
        self.workers = workers or min(32, os.cpu_count() or 1)

    def process(
        self, paths: Sequence[str], max_size: int, target: Optional[int]
    ) -> Iterator[ProcessedDocument]:
        """Yield the processed documents in the order of ``paths``."""
        jobs: List[_Job] = [
            (p, max_size, target, self.registry_factory, self.analyzer,
             self.path_weight, self.context_weight)
            for p in paths
        ]
        if not jobs:
            return
        if self.workers == 1 or len(jobs) < 8:
            yield from map(process_file, jobs)
            return
        with ProcessPoolExecutor(self.workers) as pool:
            yield from pool.map(process_file, jobs, chunksize=16)
