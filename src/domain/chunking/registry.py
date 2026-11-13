"""Selection of a chunking strategy by file extension."""

from typing import Callable, Dict, List, Optional, Protocol

from src.domain.models import Chunk


class Chunker(Protocol):
    """Split one document into chunks."""

    def chunk(self, file_path: str, text: str) -> List[Chunk]:
        """Return the chunks of ``text`` belonging to ``file_path``."""
        ...


# Builds a chunker from (max_size, target_size).
ChunkerFactory = Callable[[int, Optional[int]], Chunker]


class ChunkerRegistry:
    """Map file extensions to chunker factories, with a default.

    New document types (e.g. meeting transcripts) are supported by
    registering a factory, without touching the indexing use case.
    """

    def __init__(self, default: ChunkerFactory):
        """Create a registry that falls back to ``default``."""
        self._default = default
        self._by_extension: Dict[str, ChunkerFactory] = {}

    def register(self, extension: str, factory: ChunkerFactory) -> None:
        """Use ``factory`` for files ending with ``extension``."""
        self._by_extension[extension.lower()] = factory

    def for_path(
        self, file_path: str, max_size: int, target_size: Optional[int]
    ) -> Chunker:
        """Return the chunker suited to ``file_path``."""
        lower = file_path.lower()
        for extension, factory in self._by_extension.items():
            if lower.endswith(extension):
                return factory(max_size, target_size)
        return self._default(max_size, target_size)

    def chunk(
        self, file_path: str, text: str, max_size: int,
        target_size: Optional[int] = None,
    ) -> List[Chunk]:
        """Chunk ``text`` with the strategy registered for its path."""
        chunker = self.for_path(file_path, max_size, target_size)
        return chunker.chunk(file_path, text)


def default_registry(table_context: bool = False) -> ChunkerRegistry:
    """Return the registry used for source code repositories.

    ``table_context`` makes Markdown chunks cut inside a table keep the
    column names of the table as context.
    """
    from functools import partial

    from src.domain.chunking.python_chunker import PythonChunker
    from src.domain.chunking.text_chunker import TextChunker

    registry = ChunkerRegistry(
        partial(TextChunker, table_context=table_context)
    )
    registry.register(".py", PythonChunker)
    return registry
