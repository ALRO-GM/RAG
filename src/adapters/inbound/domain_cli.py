"""Inbound adapter: knowledge-domain commands (``python -m src domain``).

``domain list`` shows every domain with its documents, chunks and index
version; ``domain index <id>`` builds or updates one domain's index.
"""

from typing import Any, Callable, Protocol

from src.application.knowledge.registry import DomainIndexRegistry


class DomainApp(Protocol):
    """What the domain commands need from the composition root."""

    @property
    def knowledge_registry(self) -> DomainIndexRegistry:
        """The registry of domain indexes."""
        ...


class DomainCLI:
    """Knowledge domains: list them and build their indexes."""

    def __init__(
        self,
        app: Callable[[], DomainApp],
        guard: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> None:
        """Keep the container factory and the error guard of the CLI."""
        self._app = app
        self.list = guard(self._list)
        self.index = guard(self._index)

    def _list(self) -> None:
        """List the domains: id, documents, chunks, index version."""
        rows = self._app().knowledge_registry.summaries()
        if not rows:
            print("No knowledge domains.")
        for row in rows:
            print(
                f"{row.id}\t{row.title}\t{row.language}\t"
                f"{row.documents} documents\t{row.chunks} chunks\t"
                f"index {row.index_version or '-'}"
            )

    def _index(self, domain: str, force: bool = False) -> None:
        """Build or update one domain's index.

        Args:
            domain: The domain id (a folder with a manifest.yaml).
            force: Re-chunk and re-embed every file.
        """
        done = self._app().knowledge_registry.index(str(domain), bool(force))
        r = done.report
        print(
            f"Domain {done.domain}: {r.files_total} files "
            f"({r.files_reindexed} indexed, {r.files_reused} unchanged, "
            f"{r.files_removed} removed), {r.chunks} chunks in "
            f"{r.seconds}s; index {done.index_version or '-'}"
        )
