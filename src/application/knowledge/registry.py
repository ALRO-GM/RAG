"""Use case: the registry of knowledge-domain indexes (PL-6).

Each domain has its own manifest, sources and index. The registry
indexes a domain on demand (incrementally, by file hash) and keeps one
loaded search service per domain and mode, replaced when the index on
disk changes, so several domains are searched side by side.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.application.knowledge.ports import (
    DomainCatalogue,
    DomainIndexes,
    DomainSearcher,
)
from src.domain.errors import InvalidInputError, NotFoundError, RagError
from src.domain.knowledge.domains import (
    DomainManifest,
    DomainSummary,
    check_domain_id,
    index_version,
)
from src.domain.models import IndexReport

LOG = logging.getLogger(__name__)

SEARCH_MODES = {"hybrid": "hybrid", "lexical": "bm25", "semantic": "semantic"}
"""``knowledge.query.v1`` mode → retrieval mode of the search service."""

DEFAULT_MAX_CHUNK_SIZE = 1000
"""Largest chunk of a domain, in characters."""


@dataclass(frozen=True)
class DomainIndexReport:
    """Outcome of indexing one domain."""

    domain: str
    index_version: Optional[str]
    report: IndexReport


class DomainIndexRegistry:
    """Index and search the knowledge domains.

    ``extra_documents`` counts a domain's documents that are not (yet)
    indexable files, e.g. uploads still queued; it is added to the
    sources found on disk in :meth:`summary`.
    """

    def __init__(
        self,
        catalogue: DomainCatalogue,
        indexes: DomainIndexes,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        extra_documents: Optional[Callable[[str], int]] = None,
    ) -> None:
        """Wire the catalogue and the per-domain services."""
        self.catalogue = catalogue
        self.indexes = indexes
        self.max_chunk_size = max_chunk_size
        self.extra_documents = extra_documents or (lambda _: 0)
        self._searchers: Dict[Tuple[str, str], Tuple[str, DomainSearcher]] \
            = {}
        self._lock = threading.RLock()

    def manifest(self, domain_id: str) -> DomainManifest:
        """Return the manifest of a domain (checks the id first)."""
        return self.catalogue.manifest(check_domain_id(domain_id))

    def summary(self, domain_id: str) -> DomainSummary:
        """Return one domain's row of the domain list."""
        manifest = self.manifest(domain_id)
        state = self.indexes.state(manifest.id)
        documents = len(self.indexes.sources(manifest.id)) \
            + self.extra_documents(manifest.id)
        return DomainSummary(
            id=manifest.id,
            title=manifest.title,
            language=manifest.language,
            documents=documents,
            chunks=state.chunks if state else 0,
            index_version=(
                index_version(manifest, state.fingerprint) if state else None
            ),
            updated_at=state.built_at if state else None,
        )

    def summaries(self) -> List[DomainSummary]:
        """Return every readable domain's row, sorted by id.

        A domain whose manifest or index cannot be read is left out
        (and logged) so that one broken folder does not hide the rest.
        """
        rows: List[DomainSummary] = []
        for domain_id in self.catalogue.ids():
            try:
                rows.append(self.summary(domain_id))
            except RagError as exc:
                LOG.warning("Skipping knowledge domain %s: %s", domain_id,
                            exc)
        return rows

    def index(self, domain_id: str, force: bool = False) -> DomainIndexReport:
        """Build or update a domain's index; drop it when it has no source.

        Only new or changed files are chunked and embedded again unless
        ``force``.
        """
        manifest = self.manifest(domain_id)
        with self._lock:
            self._forget(manifest.id)
            if not self.indexes.sources(manifest.id):
                self.indexes.drop(manifest.id)
                return DomainIndexReport(manifest.id, None, IndexReport(
                    files_total=0, files_reindexed=0, files_reused=0,
                    files_removed=0, chunks=0, seconds=0.0,
                    incremental=False,
                ))
            report = self.indexes.indexer(manifest).run(
                self.max_chunk_size, force=force
            )
            state = self.indexes.state(manifest.id)
        version = index_version(manifest, state.fingerprint) if state \
            else None
        return DomainIndexReport(manifest.id, version, report)

    def searcher(
        self, domain_id: str, mode: str = "hybrid"
    ) -> Tuple[DomainSearcher, str]:
        """Return a domain's search service and its index version.

        ``mode`` is a ``knowledge.query.v1`` mode (hybrid, lexical or
        semantic). ``NotFoundError`` when the domain has no index yet.
        """
        if mode not in SEARCH_MODES:
            raise InvalidInputError(
                f"Unknown mode {mode!r}; use one of "
                f"{', '.join(SEARCH_MODES)}."
            )
        manifest = self.manifest(domain_id)
        with self._lock:
            state = self.indexes.state(manifest.id)
            if state is None:
                raise NotFoundError(
                    f"The domain {manifest.id!r} has no index yet."
                )
            key = (manifest.id, mode)
            cached = self._searchers.get(key)
            if cached is None or cached[0] != state.fingerprint:
                cached = (
                    state.fingerprint,
                    self.indexes.searcher(manifest, SEARCH_MODES[mode]),
                )
                self._searchers[key] = cached
        return cached[1], index_version(manifest, state.fingerprint)

    def forget(self, domain_id: str) -> None:
        """Drop the loaded search services of a domain."""
        with self._lock:
            self._forget(domain_id)

    def _forget(self, domain_id: str) -> None:
        """Drop the cached searchers of ``domain_id`` (lock held)."""
        for key in [k for k in self._searchers if k[0] == domain_id]:
            del self._searchers[key]
