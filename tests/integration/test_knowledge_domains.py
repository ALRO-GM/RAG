"""Knowledge domains on disk: catalogue, index factory and CLI (PL-6)."""

from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import pytest

from src.adapters.inbound.domain_cli import DomainCLI
from src.adapters.outbound.fs_domain_catalogue import (
    FileSystemDomainCatalogue,
)
from src.adapters.outbound.fs_index_repository import (
    InMemoryIndexCopyRepository,
)
from src.application.knowledge.registry import DomainIndexRegistry
from src.bootstrap import KnowledgeIndexFactory
from src.config import KnowledgeSettings, Settings
from src.domain.errors import (
    BusyError,
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
)
from src.domain.knowledge.domains import DomainManifest
from tests.fakes import FakeEmbedder
from tests.integration.test_cli import run_cli

ACME = """id: support.acme
title: Acme support
language: es
"""
BACKEND = """id: interview.backend
title: Backend interviews
language: en
"""


def write(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, creating its folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def settings() -> KnowledgeSettings:
    """Return knowledge settings relative to the current folder."""
    return KnowledgeSettings(
        domains_dir="knowledge/domains", data_dir="data/knowledge",
        index_dir="data/processed/domains",
    )


@pytest.fixture()
def domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Tuple[DomainIndexRegistry, KnowledgeIndexFactory]:
    """Write two authored domains under ``tmp_path`` and chdir into it."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "knowledge/domains"
    write(root / "support.acme/manifest.yaml", ACME)
    write(root / "support.acme/refunds.md",
          "# Reembolsos\n\nLa devolución se pide en 30 días.\n")
    write(root / "support.acme/hours.txt", "Abrimos de 9 a 18.\n")
    write(root / "interview.backend/manifest.yaml", BACKEND)
    write(root / "interview.backend/arch.md",
          "# Architecture\n\nMigrating a monolith to services.\n")
    k = settings()
    factory = KnowledgeIndexFactory(k, Settings(), FakeEmbedder())
    catalogue = FileSystemDomainCatalogue(k.domains_dir, k.data_dir)
    return DomainIndexRegistry(catalogue, factory, k.max_chunk_size), factory


def test_two_domains_index_and_load_side_by_side(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory]
) -> None:
    """Each domain gets its own index folder and answers from it."""
    registry, factory = domains
    acme = registry.index("support.acme")
    backend = registry.index("interview.backend")
    assert acme.report.files_total == 2
    assert Path("data/processed/domains/support.acme/manifest.json").is_file()
    assert Path("data/processed/domains/interview.backend/chunks.jsonl") \
        .is_file()
    searcher, version = registry.searcher("support.acme")
    assert version == acme.index_version
    hits = searcher.search_where("devolucion", 3, lambda c: True)
    assert hits and hits[0].chunk.file_path.endswith("refunds.md")
    assert hits[0].chunk.metadata.title == "Reembolsos"
    other, _ = registry.searcher("interview.backend", "lexical")
    found = other.search_where("monolith", 3, lambda c: True)
    assert [h.chunk.file_path for h in found] == [
        "knowledge/domains/interview.backend/arch.md"
    ]
    assert backend.index_version != acme.index_version
    rows = {r.id: r for r in registry.summaries()}
    assert rows["support.acme"].documents == 2
    assert rows["support.acme"].chunks == acme.report.chunks
    assert rows["interview.backend"].language == "en"
    assert factory.state("nope") is None


def test_reindex_is_incremental_and_versioned(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory]
) -> None:
    """Unchanged files are reused; a change gives a new version."""
    registry, _ = domains
    first = registry.index("support.acme")
    again = registry.index("support.acme")
    assert again.report.files_reused == 2
    assert again.index_version == first.index_version
    write(Path("knowledge/domains/support.acme/hours.txt"),
          "Abrimos de 8 a 20.\n")
    changed = registry.index("support.acme")
    assert changed.report.files_reindexed == 1
    assert changed.index_version != first.index_version
    forced = registry.index("support.acme", force=True)
    assert forced.report.files_reindexed == 2


def test_loaded_index_survives_a_rewrite(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory]
) -> None:
    """Embeddings are copied out of the file: rewriting it is safe."""
    registry, _ = domains
    registry.index("support.acme")
    snapshot = InMemoryIndexCopyRepository(
        "data/processed/domains/support.acme"
    ).load()
    assert isinstance(snapshot.embeddings, np.ndarray)
    assert not isinstance(snapshot.embeddings, np.memmap)
    registry.index("support.acme", force=True)
    assert snapshot.embeddings.shape[0] == len(snapshot.chunks)


def test_catalogue_create_and_remove(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory]
) -> None:
    """Created domains live under the data root and can be removed."""
    registry, _ = domains
    catalogue = registry.catalogue
    catalogue.create(DomainManifest(id="hr", title="Políticas de RR. HH."))
    assert Path("data/knowledge/hr/manifest.yaml").is_file()
    assert catalogue.manifest("hr").title == "Políticas de RR. HH."
    assert not catalogue.authored("hr")
    assert catalogue.authored("support.acme")
    assert "hr" in catalogue.ids()
    with pytest.raises(BusyError):
        catalogue.create(DomainManifest(id="support.acme", title="x"))
    with pytest.raises(InvalidInputError, match="authored"):
        catalogue.remove("support.acme")
    catalogue.remove("hr")
    assert not Path("data/knowledge/hr").exists()
    with pytest.raises(NotFoundError):
        catalogue.manifest("hr")


@pytest.mark.parametrize("text, error", [
    ("id: other\ntitle: x\n", "differs"),
    ("id: broken\n", "title"),
    ("id: [unclosed\n", "unreadable"),
])
def test_catalogue_refuses_bad_manifests(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory],
    text: str, error: str,
) -> None:
    """A wrong id, a missing title or bad YAML is reported."""
    write(Path("knowledge/domains/broken/manifest.yaml"), text)
    with pytest.raises(CorruptedDataError, match=error):
        domains[0].catalogue.manifest("broken")
    listed = [row.id for row in domains[0].summaries()]
    assert listed == ["interview.backend", "support.acme"]


class App:
    """The container seen by :class:`DomainCLI`."""

    def __init__(self, registry: DomainIndexRegistry) -> None:
        """Hold the registry."""
        self.knowledge_registry = registry


def identity(func: Callable[..., Any]) -> Callable[..., Any]:
    """A guard that lets errors through."""
    return func


def test_domain_cli_lists_and_indexes(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``domain index`` and ``domain list`` print one line per domain."""
    app = App(domains[0])
    cli = DomainCLI(lambda: app, identity)
    cli.index("support.acme")
    out = capsys.readouterr().out
    assert out.startswith("Domain support.acme: 2 files (2 indexed")
    cli.list()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("interview.backend\tBackend interviews\ten")
    assert "index -" in lines[0]
    assert "2 documents" in lines[1] and "index -" not in lines[1]


def test_domain_cli_in_a_subprocess(tmp_path: Path) -> None:
    """``python -m src domain list`` works and errors are one line."""
    empty = run_cli(["domain", "list"], tmp_path)
    assert empty.returncode == 0, empty.stderr
    assert "No knowledge domains." in empty.stdout
    write(tmp_path / "knowledge/domains/support.acme/manifest.yaml", ACME)
    listed = run_cli(["domain", "list"], tmp_path)
    assert listed.stdout.startswith("support.acme\tAcme support\tes\t0 ")
    missing = run_cli(["domain", "index", "nope"], tmp_path)
    assert missing.returncode == 1
    assert "No knowledge domain 'nope'" in missing.stderr


LABELLED = """id: support.labelled
title: Labelled support
vocabularies:
  intent: [refund, pricing]
  script_phase: [opening, closing]
"""


def test_front_matter_and_jsonl_are_indexed_with_metadata(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory]
) -> None:
    """Metadata from both formats reaches the index and the filters."""
    registry, _ = domains
    root = Path("knowledge/domains/support.labelled")
    write(root / "manifest.yaml", LABELLED)
    write(root / "refunds.md",
          "---\nintent: refund\nscript_phase: closing\n---\n"
          "# Devoluciones\n\nEl reembolso tarda 5 días.\n")
    write(root / "faq.jsonl",
          '{"id": "p1", "text": "El reembolso no aplica a ofertas.", '
          '"intent": "pricing", "tags": ["oferta"]}\n')
    done = registry.index("support.labelled")
    assert done.report.files_total == 2
    searcher, _ = registry.searcher("support.labelled")
    every = searcher.search_where("reembolso", 5, lambda c: True)
    assert {h.chunk.metadata.intent for h in every} == {"refund", "pricing"}
    only = searcher.search_where(
        "reembolso", 5, lambda c: c.metadata.intent == "pricing"
    )
    assert [h.chunk.metadata.ref for h in only] == ["p1"]
    write(root / "refunds.md",
          "---\nintent: refund\nscript_phase: opening\n---\n"
          "# Devoluciones\n\nEl reembolso tarda 5 días.\n")
    again = registry.index("support.labelled")
    assert (again.report.files_reindexed, again.report.files_reused) == (1, 1)


def test_out_of_vocabulary_values_stop_indexing_with_file_line(
    domains: Tuple[DomainIndexRegistry, KnowledgeIndexFactory]
) -> None:
    """The index is not built and the error points at the value."""
    registry, _ = domains
    root = Path("knowledge/domains/support.labelled")
    write(root / "manifest.yaml", LABELLED)
    write(root / "bad.jsonl",
          '{"text": "ok", "intent": "refund"}\n'
          '{"text": "no", "intent": "shipping"}\n')
    with pytest.raises(InvalidInputError,
                       match=r"support.labelled/bad.jsonl:2: intent "
                             r"'shipping'"):
        registry.index("support.labelled")
    with pytest.raises(NotFoundError, match="no index"):
        registry.searcher("support.labelled")


def test_container_wires_two_knowledge_embedders() -> None:
    """Queries and the indexing worker never share a tokenizer."""
    from src.bootstrap import Container

    container = Container(Settings())
    factory = container.knowledge_indexes
    assert factory.embedder is container.knowledge_embedder
    assert factory.index_embedder is container.knowledge_index_embedder
    assert factory.embedder is not factory.index_embedder
    assert factory.embedder.model_name == factory.index_embedder.model_name
    assert container.knowledge_library.registry is \
        container.knowledge_registry
