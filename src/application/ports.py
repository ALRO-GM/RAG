"""Ports: the interfaces the use cases need from the outside world.

Outbound adapters (``src/adapters/outbound``) implement them; the
composition root (``src/bootstrap.py``) wires implementations in. Unit
tests replace them with in-memory fakes.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from src.domain.experiments import TrialResult
from src.domain.models import (
    Chunk,
    IndexManifest,
    IndexReport,
    MinimalSource,
    ProcessedDocument,
    ScoredChunk,
)
from src.domain.priors import RouterModel

FloatArray = npt.NDArray[np.float32]
Analyzer = Callable[[str], List[str]]
ModelT = TypeVar("ModelT", bound=BaseModel)
T = TypeVar("T")


class CorpusReader(Protocol):
    """Enumerate and fingerprint the documents to index."""

    def list_files(self) -> List[str]:
        """Return project-relative paths of the indexable files."""
        ...

    def file_hash(self, path: str) -> str:
        """Return a content hash of ``path``."""
        ...


class DocumentProcessor(Protocol):
    """Chunk and analyse documents (possibly in parallel)."""

    def process(
        self, paths: Sequence[str], max_size: int, target: Optional[int]
    ) -> Iterator[ProcessedDocument]:
        """Yield one processed document per path."""
        ...


class LexicalIndex(Protocol):
    """A lexical (term-based) scoring index."""

    def scores(self, terms: Sequence[str]) -> FloatArray:
        """Return one relevance score per indexed chunk."""
        ...


class LexicalIndexBuilder(Protocol):
    """Build a lexical index from per-chunk term counts."""

    def build(self, term_counts: Sequence[Mapping[str, int]]) -> LexicalIndex:
        """Return an index over the given documents."""
        ...


class Embedder(Protocol):
    """Turn texts into L2-normalised dense vectors."""

    @property
    def model_name(self) -> str:
        """Return the identifier of the embedding model."""
        ...

    def encode(
        self, texts: Sequence[str], progress: bool = False
    ) -> FloatArray:
        """Return one embedding row per text."""
        ...


@dataclass
class FileRecord:
    """Everything derived from one file, reusable while it is unchanged."""

    sha256: str
    chunks: List[Chunk]
    term_counts: List[Dict[str, int]]
    embeddings: Optional[FloatArray] = None


@dataclass
class IndexSnapshot:
    """A searchable index: chunks plus lexical and dense structures."""

    manifest: IndexManifest
    chunks: List[Chunk]
    lexical: LexicalIndex
    embeddings: Optional[FloatArray] = None
    positions: Dict[tuple[str, int, int], int] = field(default_factory=dict)

    def position_of(self, chunk: Chunk) -> int:
        """Return the index position of ``chunk``."""
        if not self.positions:
            self.positions = {
                chunk_location(c): i for i, c in enumerate(self.chunks)
            }
        return self.positions[chunk_location(chunk)]


def chunk_location(chunk: Chunk) -> tuple[str, int, int]:
    """Return the (path, first, last) tuple identifying a chunk."""
    return (
        chunk.file_path,
        chunk.first_character_index,
        chunk.last_character_index,
    )


class IndexRepository(Protocol):
    """Persist and load indexes and their per-file records."""

    def load_records(
        self, params: Mapping[str, object]
    ) -> Dict[str, FileRecord]:
        """Return cached file records built with ``params`` (or {})."""
        ...

    def save(
        self,
        snapshot: IndexSnapshot,
        records: Mapping[str, FileRecord],
        params: Mapping[str, object],
    ) -> None:
        """Persist the index and the records used to build it."""
        ...

    def load(self) -> IndexSnapshot:
        """Return the persisted index."""
        ...


class Generator(Protocol):
    """Produce an answer grounded in retrieved chunks."""

    @property
    def model_name(self) -> str:
        """Return the identifier of the language model."""
        ...

    def generate(self, question: str, context: Sequence[Chunk]) -> str:
        """Return an answer to ``question`` using only ``context``."""
        ...


class Reranker(Protocol):
    """Score (query, passage) pairs jointly, e.g. with a cross-encoder."""

    @property
    def model_name(self) -> str:
        """Return the identifier of the reranking model."""
        ...

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Return one relevance score per (query, passage) pair."""
        ...


class Cache(Protocol):
    """A persistent string key-value store."""

    def get(self, key: str) -> Optional[str]:
        """Return the value stored under ``key``, if any."""
        ...

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key``."""
        ...

    def clear(self) -> int:
        """Delete every entry; return how many were removed."""
        ...


class DatasetRepository(Protocol):
    """Read and write the JSON documents exchanged with the evaluator."""

    def read(self, path: str, model: Type[ModelT]) -> ModelT:
        """Load ``path`` and validate it as ``model``."""
        ...

    def write(self, directory: str, filename: str, data: BaseModel) -> str:
        """Write ``data`` and return the output path."""
        ...


class SourceReader(Protocol):
    """Resolve source locations to their text."""

    def read(self, sources: Sequence[MinimalSource]) -> List[Chunk]:
        """Return the readable sources as chunks, skipping broken ones."""
        ...


class Progress(Protocol):
    """Report progress of a long-running loop."""

    def track(
        self, items: Iterable[T], total: int, description: str
    ) -> Iterator[T]:
        """Yield ``items`` while displaying progress."""
        ...


class NoProgress:
    """A :class:`Progress` that displays nothing."""

    def track(
        self, items: Iterable[T], total: int, description: str
    ) -> Iterator[T]:
        """Yield ``items`` unchanged."""
        yield from items


class Pipeline(Protocol):
    """An indexing + search pipeline built for one parameter set."""

    def index(self) -> Tuple[IndexReport, bool]:
        """Build or reuse the index; the flag tells a cold build."""
        ...

    def search(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the top-``k`` chunks per query, without any cache."""
        ...


class PipelineFactory(Protocol):
    """Build pipelines from parameter overrides (used by experiments)."""

    def create(self, params: Mapping[str, Any]) -> Pipeline:
        """Return the pipeline configured by ``params``."""
        ...


class TrialStore(Protocol):
    """Persist experiment results."""

    def append(self, result: TrialResult) -> None:
        """Save one trial result."""
        ...

    def get(self, trial_id: str) -> Optional[TrialResult]:
        """Return the latest result of a trial in any suite, if any."""
        ...

    def all(self) -> List[TrialResult]:
        """Return the latest result of every (suite, trial)."""
        ...


class RouterRepository(Protocol):
    """Persist the query router model."""

    def load(self) -> Optional[RouterModel]:
        """Return the saved model, if any."""
        ...

    def save(self, model: RouterModel) -> str:
        """Save ``model`` and return where it was written."""
        ...
