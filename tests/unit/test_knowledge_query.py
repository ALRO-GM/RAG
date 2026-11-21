"""knowledge.query.v1 / result.v1 models and the query use case (PL-8)."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.domain.errors import NotFoundError
from src.domain.knowledge.contract import (
    KnowledgeQuery,
    KnowledgeResult,
    chunk_id,
)
from src.domain.knowledge.domains import DomainManifest
from src.domain.models import ChunkMetadata
from tests.fakes import make_chunk
from tests.knowledge_fakes import InMemoryCatalogue, InMemoryIndexes

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def schema(name: str) -> Draft202012Validator:
    """Return a validator of ``contracts/<name>.schema.json``."""
    data = json.loads((CONTRACTS / f"{name}.schema.json").read_text())
    Draft202012Validator.check_schema(data)
    return Draft202012Validator(data)


def example(name: str) -> Dict[str, Any]:
    """Return ``contracts/examples/<name>.json``."""
    data: Dict[str, Any] = json.loads(
        (CONTRACTS / "examples" / f"{name}.json").read_text()
    )
    return data


def test_examples_follow_their_schemas() -> None:
    """Both examples are valid documents of their contracts."""
    schema("knowledge.query.v1").validate(example("knowledge.query.v1"))
    schema("knowledge.result.v1").validate(example("knowledge.result.v1"))


def test_models_read_the_examples() -> None:
    """The models accept the examples and give them back unchanged."""
    query = KnowledgeQuery.model_validate(example("knowledge.query.v1"))
    assert query.domain == "interview.backend"
    assert query.filters.script_phase == ["technical"]
    assert query.mode == "hybrid"
    result = KnowledgeResult.model_validate(example("knowledge.result.v1"))
    dumped = result.model_dump(mode="json", by_alias=True)
    assert dumped == example("knowledge.result.v1")


def base_query(**changes: Any) -> Dict[str, Any]:
    """Return a valid query with ``changes``."""
    data: Dict[str, Any] = {
        "schema": "knowledge.query.v1", "request_id": "r-1",
        "domain": "support.acme", "text": "refund",
    }
    data.update(changes)
    return data


@pytest.mark.parametrize("changes", [
    {"schema": "knowledge.query.v2"},
    {"request_id": ""},
    {"request_id": "x" * 65},
    {"domain": "Support"},
    {"text": ""},
    {"text": "x" * 4001},
    {"top_k": 0},
    {"top_k": 51},
    {"mode": "bm25"},
    {"filters": {"category": ["chisme"]}},
    {"filters": {"speaker": ["S1"]}},
    {"deadline_ms": 0},
    {"language": "e"},
    {"extra": 1},
])
def test_model_and_schema_refuse_the_same_queries(
    changes: Dict[str, Any]
) -> None:
    """What the schema refuses the model refuses too."""
    document = base_query(**changes)
    assert list(schema("knowledge.query.v1").iter_errors(document))
    with pytest.raises(ValidationError):
        KnowledgeQuery.model_validate(document)


def test_defaults_of_a_minimal_query() -> None:
    """``top_k`` 5, hybrid, no filters."""
    query = KnowledgeQuery.model_validate(base_query())
    assert (query.top_k, query.mode) == (5, "hybrid")
    assert query.filters.is_empty()


def test_chunk_ids() -> None:
    """Records are named by their id, other chunks by file and offset."""
    chunk = make_chunk("data/knowledge/d/extracted/guide.md", 40, "x")
    assert chunk_id("d", chunk) == "d:guide.md#40"
    record = chunk.model_copy(update={"metadata": ChunkMetadata(ref="q-7")})
    assert chunk_id("d", record) == "d:q-7"


def ticking() -> Iterator[float]:
    """Yield 0.0, 0.012, 0.024... seconds."""
    now = 0.0
    while True:
        yield now
        now += 0.012


def service() -> KnowledgeQueryService:
    """Return the use case over one indexed in-memory domain."""
    indexes = InMemoryIndexes()
    indexes.files = {"support.acme": {}}
    catalogue = InMemoryCatalogue(
        DomainManifest(id="support.acme", title="Acme"),
        DomainManifest(id="empty", title="Empty"),
    )
    registry = DomainIndexRegistry(catalogue, indexes)
    indexes.built["support.acme"] = [
        make_chunk("kb.md", 0, "refund within 30 days").model_copy(
            update={"metadata": ChunkMetadata(intent="refund",
                                              tags=["faq"])}),
        make_chunk("kb.md", 30, "refund for vip customers").model_copy(
            update={"metadata": ChunkMetadata(intent="vip",
                                              title="VIP")}),
    ]
    clock = ticking()
    return KnowledgeQueryService(registry, lambda: next(clock))


def test_query_returns_contract_chunks() -> None:
    """Scores, sources, metadata, the index version and the time."""
    result = service().run(KnowledgeQuery.model_validate(base_query()))
    assert [c.id for c in result.chunks] == ["support.acme:kb.md#0",
                                             "support.acme:kb.md#30"]
    assert result.chunks[0].metadata.tags == ["faq"]
    assert result.chunks[1].metadata.title == "VIP"
    assert result.chunks[0].source.last_character_index == 21
    assert result.index_version and result.timings
    assert result.timings.total_ms == pytest.approx(12.0)
    schema("knowledge.result.v1").validate(
        result.model_dump(mode="json", by_alias=True)
    )


def test_query_applies_filters_and_top_k() -> None:
    """Only chunks passing the filters come back, at most ``top_k``."""
    svc = service()
    only = svc.run(KnowledgeQuery.model_validate(base_query(
        filters={"intent": ["vip"]}
    )))
    assert [c.metadata.intent for c in only.chunks] == ["vip"]
    one = svc.run(KnowledgeQuery.model_validate(base_query(top_k=1)))
    assert len(one.chunks) == 1


def test_query_unknown_or_unindexed_domain() -> None:
    """A missing domain or index is ``NotFoundError``."""
    svc = service()
    with pytest.raises(NotFoundError):
        svc.run(KnowledgeQuery.model_validate(base_query(domain="nope")))
    with pytest.raises(NotFoundError, match="no index"):
        svc.run(KnowledgeQuery.model_validate(base_query(domain="empty")))
