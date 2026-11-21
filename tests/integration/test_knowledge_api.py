"""``POST /knowledge/query`` over a real domain index (PL-8)."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from src.adapters.inbound.http_api import (
    KnowledgeRoutes,
    TranscriptRoutes,
    create_app,
)
from src.adapters.outbound.fs_domain_catalogue import (
    FileSystemDomainCatalogue,
)
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.application.answering import AnswerService
from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.bootstrap import KnowledgeIndexFactory
from src.config import KnowledgeSettings, Settings
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator
from tests.integration.test_knowledge_domains import LABELLED, write
from tests.integration.test_transcript_ingest import AUTH, TOKEN

RESULT_SCHEMA = Path(__file__).resolve().parents[2] / \
    "contracts/knowledge.result.v1.schema.json"


def knowledge_client(
    tmp: Path, registry: DomainIndexRegistry, routes: Any = None
) -> TestClient:
    """Return a client whose knowledge routes use ``registry``."""
    store = FileSystemTranscriptStore(str(tmp / "transcripts"))
    search = SearchService(FileSystemIndexRepository(str(tmp / "none")),
                           tokenize, FakeEmbedder())
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()),
        TranscriptRoutes(
            IngestTranscript(store), ShowIngestedTranscript(store), TOKEN,
            knowledge=routes or KnowledgeRoutes(
                KnowledgeQueryService(registry)
            ),
        ),
    ))


def labelled_registry() -> DomainIndexRegistry:
    """Index a labelled domain under the current folder."""
    root = Path("knowledge/domains/support.labelled")
    write(root / "manifest.yaml", LABELLED)
    write(root / "refunds.md",
          "---\nintent: refund\nscript_phase: closing\n---\n"
          "# Devoluciones\n\nEl reembolso tarda 5 días.\n")
    write(root / "faq.jsonl",
          '{"id": "p1", "text": "El reembolso no aplica a ofertas.", '
          '"intent": "pricing", "category": "informacion"}\n')
    k = KnowledgeSettings()
    registry = DomainIndexRegistry(
        FileSystemDomainCatalogue(k.domains_dir, k.data_dir),
        KnowledgeIndexFactory(k, Settings(), FakeEmbedder()),
    )
    registry.index("support.labelled")
    return registry


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a client over an indexed labelled domain."""
    monkeypatch.chdir(tmp_path)
    return knowledge_client(tmp_path, labelled_registry())


def query(**changes: Any) -> Dict[str, Any]:
    """Return a valid query body with ``changes``."""
    body: Dict[str, Any] = {
        "schema": "knowledge.query.v1", "request_id": "r-1",
        "domain": "support.labelled", "text": "reembolso",
    }
    body.update(changes)
    return body


def test_query_answers_a_valid_result(client: TestClient) -> None:
    """The answer follows knowledge.result.v1."""
    response = client.post("/knowledge/query", json=query(), headers=AUTH)
    assert response.status_code == 200, response.text
    body = response.json()
    validator = Draft202012Validator(json.loads(RESULT_SCHEMA.read_text()))
    validator.validate(body)
    assert body["request_id"] == "r-1"
    assert {c["id"] for c in body["chunks"]} == {
        "support.labelled:p1", "support.labelled:refunds.md#45",
    }
    assert body["timings"]["total_ms"] >= 0


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_query_filters_in_every_mode(client: TestClient, mode: str) -> None:
    """Filters keep only the matching chunks."""
    response = client.post("/knowledge/query", headers=AUTH, json=query(
        mode=mode, filters={"category": ["informacion"]}
    ))
    assert [c["id"] for c in response.json()["chunks"]] == \
        ["support.labelled:p1"]


def test_query_needs_the_token(client: TestClient) -> None:
    """Without the bearer token the route refuses (401)."""
    assert client.post("/knowledge/query", json=query()).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/knowledge/query", json=query(),
                       headers=wrong).status_code == 401


@pytest.mark.parametrize("body, status", [
    (b"not json", 422),
    (json.dumps(query(top_k=0)).encode(), 422),
    (json.dumps(query(filters={"x": []})).encode(), 422),
    (json.dumps(query(domain="nope")).encode(), 404),
    (b"x" * (64 * 1024 + 1), 413),
])
def test_query_errors(client: TestClient, body: bytes, status: int) -> None:
    """Contract violations 422, unknown domains 404, huge bodies 413."""
    response = client.post("/knowledge/query", content=body, headers=AUTH)
    assert response.status_code == status


def test_routes_are_off_without_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no token nothing under /knowledge is mounted."""
    monkeypatch.chdir(tmp_path)
    store = FileSystemTranscriptStore(str(tmp_path / "t"))
    search = SearchService(FileSystemIndexRepository(str(tmp_path / "i")),
                           tokenize, FakeEmbedder())
    app = create_app(search, AnswerService(search, FakeGenerator()),
                     TranscriptRoutes(IngestTranscript(store),
                                      ShowIngestedTranscript(store), "",
                                      knowledge=KnowledgeRoutes(
                                          KnowledgeQueryService(
                                              labelled_registry()))))
    response = TestClient(app).post("/knowledge/query", json=query(),
                                    headers=AUTH)
    assert response.status_code == 404
