"""Domain manifests, sources and the domain index registry (PL-6)."""

import pytest
from pydantic import ValidationError

from src.application.knowledge.registry import DomainIndexRegistry
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.knowledge.domains import (
    DomainManifest,
    Vocabularies,
    check_domain_id,
    index_version,
)
from src.domain.knowledge.sources import (
    chunk_text,
    fold_accents,
    knowledge_terms,
)
from src.domain.models import ChunkMetadata
from tests.knowledge_fakes import InMemoryCatalogue, InMemoryIndexes


def manifest(domain_id: str = "support.acme") -> DomainManifest:
    """Return a small manifest."""
    return DomainManifest(id=domain_id, title="Acme support")


@pytest.mark.parametrize("value", [
    "a", "support.acme", "interview_backend-2", "x" * 64, "0.1",
])
def test_valid_domain_ids(value: str) -> None:
    """Ids follow the contract's pattern."""
    assert check_domain_id(value) == value


@pytest.mark.parametrize("value", [
    "", "A", ".hidden", "-x", "x" * 65, "a/b", "../etc", "a b", "é",
])
def test_invalid_domain_ids(value: str) -> None:
    """Anything else is refused, path separators included."""
    with pytest.raises(InvalidInputError):
        check_domain_id(value)


def test_manifest_validation() -> None:
    """The manifest checks its id and refuses unknown keys."""
    assert manifest().language == "es"
    assert manifest().vocabularies == Vocabularies()
    with pytest.raises(ValidationError):
        DomainManifest(id="Bad", title="x")
    with pytest.raises(ValidationError):
        DomainManifest.model_validate({"id": "a", "title": "t", "x": 1})


def test_index_version_tracks_manifest_and_files() -> None:
    """The version changes with the manifest or the indexed files."""
    a = index_version(manifest(), "fp1")
    assert a == index_version(manifest(), "fp1")
    assert len(a) == 16
    assert a != index_version(manifest(), "fp2")
    other = manifest().model_copy(
        update={"vocabularies": Vocabularies(intent=["refund"])}
    )
    assert a != index_version(other, "fp1")


def test_fold_accents_keeps_spanish_words_whole() -> None:
    """Accented words become single ASCII terms."""
    assert fold_accents("Migración, año, pingüino") == \
        "Migracion, ano, pinguino"
    assert "migracion" in knowledge_terms("La migración del monolito")


def test_chunk_text_offsets_and_titles() -> None:
    """Chunks keep absolute offsets; headings become the title."""
    head = "---\nx: 1\n---\n"
    body = ("An introduction of thirty-five chars\n\n"
            "# Refunds\n\nAsk within 30 days.\n")
    text = head + body
    chunks = chunk_text("kb.md", text, 40, start=len(head),
                        base=ChunkMetadata(title="Doc", tags=["faq"]))
    assert chunks
    for c in chunks:
        assert text[c.first_character_index:c.last_character_index] == \
            c.text
        assert c.metadata.tags == ["faq"]
    assert chunks[0].metadata.title == "Doc"
    assert chunks[-1].metadata.title == "Refunds"


def registry() -> tuple[DomainIndexRegistry, InMemoryIndexes]:
    """Return a registry over two in-memory domains."""
    indexes = InMemoryIndexes()
    indexes.files = {
        "support.acme": {"a.md": "refund policy", "b.md": "opening hours"},
        "interview.backend": {"q.md": "monolith migration"},
    }
    catalogue = InMemoryCatalogue(
        manifest(), manifest("interview.backend"), manifest("empty")
    )
    return DomainIndexRegistry(
        catalogue, indexes, 800, extra_documents=lambda d: 2 * (d == "empty")
    ), indexes


def test_summaries_before_and_after_indexing() -> None:
    """Rows list documents, chunks and the index version once built."""
    reg, _ = registry()
    rows = {r.id: r for r in reg.summaries()}
    assert list(rows) == ["empty", "interview.backend", "support.acme"]
    assert rows["support.acme"].documents == 2
    assert rows["support.acme"].index_version is None
    assert rows["empty"].documents == 2
    done = reg.index("support.acme")
    assert done.report.chunks == 2
    row = reg.summary("support.acme")
    assert row.chunks == 2
    assert row.index_version == done.index_version
    assert row.updated_at == "2026-11-21T10:00:00+00:00"


def test_two_domains_are_searched_side_by_side() -> None:
    """Each domain answers from its own index only."""
    reg, indexes = registry()
    reg.index("support.acme")
    reg.index("interview.backend", force=True)
    assert indexes.runs[-1] == ("interview.backend", 800, True)
    acme, v1 = reg.searcher("support.acme")
    backend, v2 = reg.searcher("interview.backend", "lexical")
    assert v1 != v2
    assert [h.chunk.text for h in acme.search_where(
        "refund migration", 5, lambda c: True)] == ["refund policy"]
    assert [h.chunk.text for h in backend.search_where(
        "refund migration", 5, lambda c: True)] == ["monolith migration"]


def test_searchers_are_kept_until_the_index_changes() -> None:
    """A loaded searcher is reused; re-indexing replaces it."""
    reg, indexes = registry()
    reg.index("support.acme")
    first, version = reg.searcher("support.acme")
    again, same = reg.searcher("support.acme")
    assert first is again and version == same
    assert indexes.searchers_made == 1
    indexes.files["support.acme"]["c.md"] = "new text"
    reg.index("support.acme")
    third, changed = reg.searcher("support.acme")
    assert third is not first and changed != version
    reg.forget("support.acme")
    assert reg.searcher("support.acme")[0] is not third


def test_index_without_sources_drops_the_index() -> None:
    """A domain left without files has no index any more."""
    reg, indexes = registry()
    reg.index("support.acme")
    indexes.files["support.acme"] = {}
    done = reg.index("support.acme")
    assert done.index_version is None and done.report.files_total == 0
    with pytest.raises(NotFoundError, match="no index"):
        reg.searcher("support.acme")


def test_registry_refuses_bad_input() -> None:
    """Unknown domains, bad ids and modes are refused."""
    reg, _ = registry()
    with pytest.raises(NotFoundError):
        reg.index("nope")
    with pytest.raises(InvalidInputError):
        reg.summary("../x")
    with pytest.raises(InvalidInputError, match="mode"):
        reg.searcher("support.acme", "bm25")
