"""Ports of the knowledge-domain use cases.

The file-system catalogue lives in ``src/adapters/outbound``; the per-domain
indexing and search services are built by the composition root
(``src/bootstrap.py``). Unit tests use in-memory fakes.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from src.application.ports import IndexSnapshot
from src.domain.knowledge.domains import DomainManifest
from src.domain.models import Chunk, IndexReport, ScoredChunk


class DomainCatalogue(Protocol):
    """Where the domains' manifests live."""

    def ids(self) -> List[str]:
        """Return the ids of every domain, sorted."""
        ...

    def manifest(self, domain_id: str) -> DomainManifest:
        """Return a domain's manifest (``NotFoundError`` if none)."""
        ...

    def authored(self, domain_id: str) -> bool:
        """Tell whether the domain is authored in the repository."""
        ...

    def create(self, manifest: DomainManifest) -> None:
        """Create a domain (``BusyError`` if the id is taken)."""
        ...

    def remove(self, domain_id: str) -> None:
        """Delete a created domain's folder, uploads included."""
        ...


class DomainIndexer(Protocol):
    """Build or incrementally update one domain's index."""

    def run(
        self,
        max_chunk_size: int = 2000,
        target_chunk_size: Optional[int] = None,
        force: bool = False,
    ) -> IndexReport:
        """Index the domain's sources; reuse unchanged files."""
        ...


class DomainSearcher(Protocol):
    """Search one domain's index."""

    @property
    def snapshot(self) -> IndexSnapshot:
        """Return the loaded index."""
        ...

    def search_where(
        self, query: str, k: int, keep: Callable[[Chunk], bool]
    ) -> List[ScoredChunk]:
        """Return the top-``k`` chunks that ``keep`` accepts."""
        ...


@dataclass(frozen=True)
class IndexState:
    """What is on disk of a domain's index."""

    fingerprint: str
    chunks: int
    files: int
    built_at: str


class DomainIndexes(Protocol):
    """Build the indexing and search services of each domain."""

    def indexer(self, manifest: DomainManifest) -> DomainIndexer:
        """Return the indexing service of a domain."""
        ...

    def searcher(self, manifest: DomainManifest, mode: str) -> DomainSearcher:
        """Return a search service of a domain (``mode``: bm25,
        semantic or hybrid)."""
        ...

    def state(self, domain_id: str) -> Optional[IndexState]:
        """Return the state of a domain's index (``None``: not built)."""
        ...

    def sources(self, domain_id: str) -> List[str]:
        """Return the paths of the domain's indexable files."""
        ...

    def drop(self, domain_id: str) -> None:
        """Delete a domain's index."""
        ...
