"""``/knowledge/interview/avoid`` over KN-4's rules file (LV-17, #232).

The rules module of #234 (``src.domain.knowledge.interview_rules``) is
loaded by name; these tests give the adapter a stand-in module, and the
last ones use the real module with the fixture file when it exists.
"""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import KnowledgeRoutes
from src.adapters.outbound.fs_domain_catalogue import (
    FileSystemDomainCatalogue,
)
from src.adapters.outbound.fs_forbidden_rules import (
    RULES_MODULE,
    FileForbiddenRules,
)
from src.application.knowledge.avoid import AvoidQuestions
from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.bootstrap import KnowledgeIndexFactory
from src.config import KnowledgeSettings, Settings
from src.domain.errors import NotFoundError
from tests.fakes import FakeEmbedder
from tests.integration.test_knowledge_api import knowledge_client
from tests.integration.test_transcript_ingest import AUTH

FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures" / "interview"
           / "forbidden-questions.min.json")


def text(es: str, en: str) -> SimpleNamespace:
    """A two-language text as the rules module gives it."""
    return SimpleNamespace(es=es, en=en)


class StandInRules:
    """Parsed rules: one category, matched on "edad" or "old"."""

    def __init__(self, data: Any) -> None:
        self.data = data
        self.categories = [SimpleNamespace(
            id="age", title=text("Edad", "Age"),
            why=text("No influye.", "It does not count."),
            lawful_alternative=text("Experiencia.", "Experience."),
        ), SimpleNamespace(
            id="health", title=text("Salud", "Health"),
            why=text("No.", "No."), lawful_alternative=text("Tarea.", "Task."),
        )]

    def hits(self, question: str) -> List[str]:
        return ["age"] if "edad" in question or "old" in question else []


def stand_in(parsed: List[Any]) -> Any:
    """A loader that returns a module whose parser records each parse."""

    def parse(data: Any) -> StandInRules:
        if data.get("schema") != "interview.forbidden.v1":
            raise ValueError("not the schema")
        rules = StandInRules(data)
        parsed.append(rules)
        return rules

    def load(name: str) -> ModuleType:
        assert name == RULES_MODULE
        module = ModuleType(name)
        setattr(module, "parse_forbidden", parse)
        return module

    return load


def write_rules(path: Path, schema: str = "interview.forbidden.v1") -> Path:
    path.write_text(json.dumps({"schema": schema}), encoding="utf-8")
    return path


def test_hits_become_hints_and_the_file_is_read_once(tmp_path: Path) -> None:
    parsed: List[Any] = []
    rules = FileForbiddenRules(write_rules(tmp_path / "r.json"),
                               stand_in(parsed))
    found = rules.hints("¿Qué edad tienes?", "es")
    assert [h.model_dump() for h in found] == [{
        "category": "age", "title": "Edad", "why": "No influye.",
        "alternative": "Experiencia."}]
    assert rules.hints("How old are you?", "en")[0].title == "Age"
    assert rules.hints("¿Cuántos años llevas con Python?", "es") == []
    assert len(parsed) == 1


def test_missing_file_module_or_bad_json_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        FileForbiddenRules(tmp_path / "none.json", stand_in([])).hints(
            "x", "es")

    def no_module(name: str) -> ModuleType:
        raise ImportError(name)

    present = write_rules(tmp_path / "r.json")
    with pytest.raises(NotFoundError, match="module"):
        FileForbiddenRules(present, no_module).hints("x", "es")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(NotFoundError, match="Unreadable"):
        FileForbiddenRules(bad, stand_in([])).hints("x", "es")
    wrong = write_rules(tmp_path / "wrong.json", "other.v1")
    with pytest.raises(NotFoundError):
        FileForbiddenRules(wrong, stand_in([])).hints("x", "es")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose knowledge routes include the avoid check."""
    monkeypatch.chdir(tmp_path)
    k = KnowledgeSettings()
    registry = DomainIndexRegistry(
        FileSystemDomainCatalogue(k.domains_dir, k.data_dir),
        KnowledgeIndexFactory(k, Settings(), FakeEmbedder()),
    )
    rules = FileForbiddenRules(write_rules(tmp_path / "r.json"),
                               stand_in([]))
    return knowledge_client(tmp_path, registry, KnowledgeRoutes(
        KnowledgeQueryService(registry), avoid=AvoidQuestions(rules),
    ))


def test_the_route_answers_the_hints(client: TestClient) -> None:
    answer = client.post("/knowledge/interview/avoid", headers=AUTH,
                         json={"text": "¿Qué edad tienes?"})
    assert answer.status_code == 200
    assert answer.json()["language"] == "es"
    assert answer.json()["hints"][0]["category"] == "age"
    clean = client.post("/knowledge/interview/avoid", headers=AUTH,
                        json={"text": "How many years?", "language": "en"})
    assert clean.json() == {"language": "en", "hints": []}


@pytest.mark.parametrize("body, status", [
    (b"not json", 422), (b"{}", 422), (b'{"text": 3}', 422),
    (b'{"text": "a", "language": 1}', 422), (b'{"text": " "}', 400),
])
def test_bad_bodies(client: TestClient, body: bytes, status: int) -> None:
    answer = client.post("/knowledge/interview/avoid", headers=AUTH,
                         content=body)
    assert answer.status_code == status


def test_without_rules_the_route_says_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    k = KnowledgeSettings()
    registry = DomainIndexRegistry(
        FileSystemDomainCatalogue(k.domains_dir, k.data_dir),
        KnowledgeIndexFactory(k, Settings(), FakeEmbedder()),
    )
    client = knowledge_client(tmp_path, registry, KnowledgeRoutes(
        KnowledgeQueryService(registry),
        avoid=AvoidQuestions(FileForbiddenRules(tmp_path / "none.json")),
    ))
    answer = client.post("/knowledge/interview/avoid", headers=AUTH,
                         json={"text": "¿Qué edad tienes?"})
    assert answer.status_code == 404


def test_the_real_module_reads_the_fixture() -> None:
    """Runs once KN-4's module is in the tree (#234)."""
    pytest.importorskip(RULES_MODULE)
    rules = FileForbiddenRules(FIXTURE)
    assert [h.category for h in rules.hints(
        "Muy bien. Por cierto, ¿qué edad tienes?", "es")] == ["age"]
    assert rules.hints("¿Cuántos años llevas con Python?", "es") == []
    assert rules.hints("Any sick leave last year?", "en")[0].title == "Health"


def test_the_demo_flags_only_its_age_question() -> None:
    """With KN-4's real file, only the demo's age question is flagged."""
    pytest.importorskip(RULES_MODULE)
    root = Path(__file__).resolve().parents[2]
    path = root / KnowledgeSettings().interview_rules
    if not path.is_file():
        pytest.skip("KN-4's rules file is not in this tree")
    rules = FileForbiddenRules(path)
    demo = json.loads((root / "examples" / "demo"
                       / "entrevista-lucia.demo.json").read_text("utf-8"))
    flagged = {a["utterance"] for a in demo["avoid"]}
    for line in demo["transcript"]["utterances"]:
        if line["speaker"] != "entrevistador":
            continue
        found = [h.category for h in rules.hints(line["text"], "es")]
        assert found == (["age"] if line["id"] in flagged else []), line


def test_the_route_needs_the_token(client: TestClient) -> None:
    answer = client.post("/knowledge/interview/avoid",
                         json={"text": "¿Qué edad tienes?"})
    assert answer.status_code == 401
    wrong = client.post("/knowledge/interview/avoid",
                        headers={"Authorization": "Bearer nope"},
                        json={"text": "¿Qué edad tienes?"})
    assert wrong.status_code == 401
