"""``knowledge.query.v1`` and ``knowledge.result.v1`` as models.

They mirror ``contracts/knowledge.query.v1.schema.json`` and
``contracts/knowledge.result.v1.schema.json`` (CONTRACTS.md §2-3); the
contract tests check both against the schemas and their examples.
"""

from pathlib import PurePosixPath
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.domain.knowledge.domains import DOMAIN_ID_PATTERN
from src.domain.knowledge.filters import KnowledgeFilters
from src.domain.models import Chunk, MinimalSource

QUERY_SCHEMA = "knowledge.query.v1"
RESULT_SCHEMA = "knowledge.result.v1"


class KnowledgeQuery(BaseModel):
    """A query to one domain (body of ``POST /knowledge/query``).

    ``language`` and ``deadline_ms`` are accepted and not used yet: a
    domain has one analyzer and a query is never cut short.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["knowledge.query.v1"] = Field(alias="schema")
    request_id: str = Field(min_length=1, max_length=64)
    domain: str = Field(pattern=f"^{DOMAIN_ID_PATTERN}$")
    text: str = Field(min_length=1, max_length=4000)
    language: Optional[str] = Field(default=None, min_length=2)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "lexical", "semantic"] = "hybrid"
    filters: KnowledgeFilters = Field(default_factory=KnowledgeFilters)
    deadline_ms: Optional[int] = Field(default=None, ge=1)


class ResultMetadata(BaseModel):
    """The metadata of a returned chunk."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    intent: Optional[str] = None
    category: Optional[str] = None
    script_phase: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ResultChunk(BaseModel):
    """One retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    score: float
    source: MinimalSource
    metadata: ResultMetadata


class ResultTimings(BaseModel):
    """How long the query took."""

    model_config = ConfigDict(extra="forbid")

    total_ms: float = Field(ge=0)


class KnowledgeResult(BaseModel):
    """The answer to a :class:`KnowledgeQuery`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["knowledge.result.v1"] = Field(
        default="knowledge.result.v1", alias="schema"
    )
    request_id: str
    domain: str
    index_version: str
    chunks: List[ResultChunk]
    timings: Optional[ResultTimings] = None


def chunk_id(domain: str, chunk: Chunk) -> str:
    """Return the id of a chunk in results.

    ``<domain>:<record id>`` for a JSONL record, else
    ``<domain>:<file name>#<first character>``.
    """
    if chunk.metadata.ref:
        return f"{domain}:{chunk.metadata.ref}"
    name = PurePosixPath(chunk.file_path).name
    return f"{domain}:{name}#{chunk.first_character_index}"


def result_chunk(domain: str, chunk: Chunk, score: float) -> ResultChunk:
    """Return a retrieved chunk as the contract shows it."""
    m = chunk.metadata
    return ResultChunk(
        id=chunk_id(domain, chunk),
        text=chunk.text,
        score=score,
        source=chunk.to_source(),
        metadata=ResultMetadata(
            title=m.title, intent=m.intent, category=m.category,
            script_phase=m.script_phase, tags=list(m.tags),
        ),
    )
