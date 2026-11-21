"""Chunk metadata and the knowledge filters (PL-5, CONTRACTS.md §2)."""

import pickle
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.application.search import SearchService
from src.domain.knowledge.filters import KnowledgeFilters
from src.domain.models import Chunk, ChunkMetadata
from src.domain.text_analysis import tokenize
from tests.fakes import (
    FakeEmbedder,
    InMemoryIndexRepository,
    make_chunk,
    make_snapshot,
)


def labelled(text: str, start: int, **fields: Any) -> Chunk:
    """Return a chunk of ``text`` with metadata ``fields``."""
    chunk = make_chunk("kb.md", start, text)
    return chunk.model_copy(update={"metadata": ChunkMetadata(**fields)})


def test_chunk_metadata_defaults_to_empty() -> None:
    """A chunk built as before has empty metadata."""
    chunk = make_chunk("a.md", 0, "text")
    assert chunk.metadata == ChunkMetadata()
    assert chunk.metadata.is_empty()
    assert not ChunkMetadata(tags=["x"]).is_empty()
    assert not ChunkMetadata(title="T").is_empty()


def test_chunk_metadata_rejects_unknown_fields() -> None:
    """Metadata keys are the contract's, nothing else."""
    with pytest.raises(ValidationError):
        ChunkMetadata.model_validate({"colour": "red"})


def test_chunk_pickled_before_metadata_loads_with_empty_one() -> None:
    """Records pickled by older code (no metadata) stay usable."""
    chunk = make_chunk("a.md", 0, "text")
    state = chunk.__getstate__()
    del state["__dict__"]["metadata"]
    old = Chunk.__new__(Chunk)
    old.__setstate__(state)
    assert old.metadata == ChunkMetadata()
    again = pickle.loads(pickle.dumps(old))
    assert again == chunk


def test_chunk_json_without_metadata_loads() -> None:
    """``chunks.jsonl`` lines written before metadata still validate."""
    line = ('{"file_path": "a.md", "first_character_index": 0, '
            '"last_character_index": 4, "kind": "text", "text": "text"}')
    assert Chunk.model_validate_json(line).metadata.is_empty()


@pytest.mark.parametrize("filters, expected", [
    ({}, True),
    ({"intent": [], "tags": [], "category": None}, True),
    ({"intent": ["pricing"]}, True),
    ({"intent": ["refund"]}, False),
    ({"intent": ["refund", "pricing"]}, True),
    ({"category": ["pregunta"]}, True),
    ({"category": ["respuesta"]}, False),
    ({"script_phase": ["opening"]}, False),
    ({"tags": ["vip"]}, True),
    ({"tags": ["vip", "other"]}, True),
    ({"tags": ["other"]}, False),
    ({"intent": ["pricing"], "tags": ["other"]}, False),
    ({"intent": ["pricing"], "category": ["pregunta"],
      "script_phase": ["closing"], "tags": ["b2b"]}, True),
])
def test_filters_follow_the_contract(
    filters: Dict[str, Any], expected: bool
) -> None:
    """Every filter given must match; tags match on any overlap."""
    metadata = ChunkMetadata(
        intent="pricing", category="pregunta", script_phase="closing",
        tags=["vip", "b2b"],
    )
    assert KnowledgeFilters.model_validate(filters).matches(metadata) \
        is expected


def test_a_chunk_without_a_value_fails_a_filter_on_it() -> None:
    """A filter on a field the chunk does not set excludes it."""
    assert not KnowledgeFilters(intent=["x"]).matches(ChunkMetadata())
    assert not KnowledgeFilters(tags=["x"]).matches(ChunkMetadata())
    assert KnowledgeFilters().is_empty()
    assert not KnowledgeFilters(tags=["x"]).is_empty()


def test_filters_reject_unknown_keys() -> None:
    """``additionalProperties: false`` as in the schema."""
    with pytest.raises(ValidationError):
        KnowledgeFilters.model_validate({"speaker": ["S1"]})


def corpus() -> List[Chunk]:
    """Return labelled chunks that all mention 'refund'."""
    return [
        labelled("refund policy for orders", 0, intent="refund"),
        labelled("refund refund refund in the opening", 100,
                 intent="greeting", script_phase="opening"),
        labelled("how to ask for a refund", 200, intent="refund",
                 tags=["faq"]),
        labelled("unrelated pricing text", 300, intent="pricing"),
    ]


def service(chunks: List[Chunk], mode: str = "bm25") -> SearchService:
    """Return a search service over ``chunks`` in memory."""
    embedder = FakeEmbedder()
    repo = InMemoryIndexRepository()
    repo.snapshot = make_snapshot(chunks, embedder)
    return SearchService(repo, tokenize, embedder, mode=mode)


@pytest.mark.parametrize("mode", ["bm25", "semantic", "hybrid"])
def test_search_where_keeps_only_matching_chunks(mode: str) -> None:
    """The filtered ranking is the unfiltered one minus excluded chunks."""
    svc = service(corpus(), mode)
    keep = KnowledgeFilters(intent=["refund"]).matches
    hits = svc.search_where("refund", 5, lambda c: keep(c.metadata))
    assert hits
    assert all(h.chunk.metadata.intent == "refund" for h in hits)
    full = [h.chunk for h in svc.search("refund", 4)
            if h.chunk.metadata.intent == "refund"]
    assert [h.chunk for h in hits] == full[:len(hits)]


def test_search_where_widens_to_the_whole_index() -> None:
    """When the first candidates are all excluded, every chunk is ranked."""
    chunks = [
        labelled(f"refund refund note {i}", i * 50, intent="noise")
        for i in range(60)
    ] + [labelled("refund once", 9000, intent="rare")]
    svc = service(chunks)
    hits = svc.search_where(
        "refund", 1, lambda c: c.metadata.intent == "rare"
    )
    assert [h.chunk.text for h in hits] == ["refund once"]


def test_search_where_blank_query_and_no_match() -> None:
    """A blank query or a filter nobody passes returns nothing."""
    svc = service(corpus())
    assert svc.search_where("  ", 3, lambda c: True) == []
    assert svc.search_where("refund", 3, lambda c: False) == []


def test_search_without_filters_is_unchanged() -> None:
    """``search`` and ``search_where`` agree when every chunk passes."""
    svc = service(corpus(), "hybrid")
    assert svc.search_where("refund", 3, lambda c: True) == \
        svc.search("refund", 3)
