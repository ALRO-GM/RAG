"""Use case: answer a ``knowledge.query.v1`` (PL-8)."""

import time
from typing import Callable

from src.application.knowledge.registry import DomainIndexRegistry
from src.domain.knowledge.contract import (
    KnowledgeQuery,
    KnowledgeResult,
    ResultTimings,
    result_chunk,
)
from src.domain.models import Chunk


class KnowledgeQueryService:
    """Search one domain with the query's mode, filters and ``top_k``."""

    def __init__(
        self,
        registry: DomainIndexRegistry,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Use ``registry``'s domains; ``clock`` times the query."""
        self.registry = registry
        self.clock = clock

    def run(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Return the best chunks of the domain that pass the filters.

        ``NotFoundError`` for an unknown domain or one without an index.
        """
        start = self.clock()
        searcher, version = self.registry.searcher(query.domain, query.mode)
        filters = query.filters

        def keep(chunk: Chunk) -> bool:
            """Tell whether ``chunk`` passes the filters."""
            return filters.matches(chunk.metadata)

        hits = searcher.search_where(query.text, query.top_k, keep)
        elapsed = (self.clock() - start) * 1000.0
        return KnowledgeResult(
            request_id=query.request_id,
            domain=query.domain,
            index_version=version,
            chunks=[result_chunk(query.domain, h.chunk, h.score)
                    for h in hits],
            timings=ResultTimings(total_ms=round(max(elapsed, 0.0), 3)),
        )
