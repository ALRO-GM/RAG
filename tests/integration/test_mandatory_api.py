"""``/knowledge/.../mandatory`` over a domain folder (LV-16)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import KnowledgeRoutes
from src.adapters.outbound.fs_domain_catalogue import (
    FileSystemDomainCatalogue,
)
from src.application.knowledge.mandatory import MandatoryItems
from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.bootstrap import KnowledgeIndexFactory
from src.config import KnowledgeSettings, Settings
from tests.fakes import FakeEmbedder
from tests.integration.test_knowledge_api import knowledge_client
from tests.integration.test_knowledge_domains import write
from tests.integration.test_transcript_ingest import AUTH

MANIFEST = """id: legal.demo
title: Legal
mandatory:
  version: 1
  threshold: 0.6
  items:
    - id: desistimiento
      text: "14 días para desistir"
      phase: closing
      must_say:
        - "tiene catorce días para desistir del contrato"
"""


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose knowledge routes include the mandatory items."""
    monkeypatch.chdir(tmp_path)
    write(Path("knowledge/domains/legal.demo/manifest.yaml"), MANIFEST)
    write(Path("knowledge/domains/plain/manifest.yaml"),
          "id: plain\ntitle: Plain\n")
    k = KnowledgeSettings()
    registry = DomainIndexRegistry(
        FileSystemDomainCatalogue(k.domains_dir, k.data_dir),
        KnowledgeIndexFactory(k, Settings(), FakeEmbedder()),
    )
    return knowledge_client(tmp_path, registry, KnowledgeRoutes(
        KnowledgeQueryService(registry), None,
        MandatoryItems(registry.manifest, FakeEmbedder()),
    ))


def test_the_list_and_a_check(client: TestClient) -> None:
    listed = client.get("/knowledge/domains/legal.demo/mandatory",
                        headers=AUTH)
    assert listed.status_code == 200
    assert listed.json()["items"][0]["id"] == "desistimiento"
    plain = client.get("/knowledge/domains/plain/mandatory", headers=AUTH)
    assert plain.json() == {"domain": "plain", "version": 0, "items": []}
    said = client.post("/knowledge/mandatory", headers=AUTH, json={
        "domain": "legal.demo",
        "text": "Tiene catorce días para desistir del contrato.",
    })
    assert said.status_code == 200
    assert said.json()["items"][0]["said"] is True


@pytest.mark.parametrize("body, status", [
    (b"not json", 422), (b'{"domain": "legal.demo"}', 422),
    (b'{"domain": "legal.demo", "text": ""}', 400),
    (b'{"domain": "plain", "text": "hola"}', 404),
    (b'{"domain": "nope", "text": "hola"}', 404),
])
def test_bad_checks(client: TestClient, body: bytes, status: int) -> None:
    answer = client.post("/knowledge/mandatory", headers=AUTH,
                         content=body)
    assert answer.status_code == status


def test_the_routes_need_the_token(client: TestClient) -> None:
    assert client.get(
        "/knowledge/domains/legal.demo/mandatory").status_code == 401
    assert client.get("/knowledge/domains/nope/mandatory",
                      headers=AUTH).status_code == 404
