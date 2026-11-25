"""Candidate profiles on disk, the core-questions file, the routes (#235)."""

import argparse
import json
import stat
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.cli import (
    build_interview,
    load_products,
    retention_days,
)
from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.adapters.outbound.fs_candidates import JsonCandidateStore
from orchestrator.adapters.outbound.fs_core_questions import FileCoreQuestions
from orchestrator.application.candidates import CandidateBook
from orchestrator.domain.candidate import (
    Captured,
    Value,
    new_profile,
    with_capture,
)
from tests.unit.orchestrator.fixtures import PRODUCTS, REPO
from tests.unit.orchestrator.test_candidate import NOW
from tests.unit.orchestrator.test_interview_sessions import live
from tests.unit.orchestrator.test_interview_usecases import OFFER

CORE = REPO / "knowledge" / "interview" / "core-questions.json"


def test_profiles_are_kept_private_and_deleted(tmp_path: Path) -> None:
    store = JsonCandidateStore(tmp_path)
    assert store.plans() == [] and store.list("dev") == []
    profile = new_profile("dev", "Lucía", [], NOW)
    store.save(profile)
    path = tmp_path / "candidates" / "dev" / "lucia.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "candidates").stat().st_mode) == 0o700
    assert store.load("dev", "lucia") == profile
    assert json.loads(path.read_text())["schema"] == "interview.candidate.v1"
    (path.parent / "broken.json").write_text("{", encoding="utf-8")
    assert [p.id for p in store.list("dev")] == ["lucia"]
    assert store.plans() == ["dev"]
    with pytest.raises(KeyError):
        store.load("dev", "broken")
    for bad in ("../x", "Dev", ""):
        with pytest.raises(KeyError):
            store.load(bad, "lucia")
        with pytest.raises(KeyError):
            store.load("dev", bad)
    assert store.delete("dev", "lucia") is True
    assert store.delete("dev", "lucia") is False
    (path.parent / "broken.json").unlink()
    assert store.delete("dev", "nobody") is False
    store.save(profile)
    assert store.delete("dev", "lucia") is True
    assert not path.parent.exists()


def test_the_core_questions_file_is_read_and_followed(tmp_path: Path) -> None:
    source = FileCoreQuestions(CORE)
    assert len(source.questions()) == 5 and source.error is None
    assert source.questions()[0].id == "work_mode_location"  # cached
    copy = tmp_path / "core.json"
    data = json.loads(CORE.read_text(encoding="utf-8"))
    data["questions"] = data["questions"][:2]
    copy.write_text(json.dumps(data), encoding="utf-8")
    assert len(FileCoreQuestions(copy).questions()) == 2
    copy.write_text("{", encoding="utf-8")
    broken = FileCoreQuestions(copy)
    assert broken.questions() == [] and broken.error
    assert FileCoreQuestions(tmp_path / "none.json").questions() == []


def wired(tmp_path: Path) -> TestClient:
    sessions = live([])
    wiring = sessions.interview
    assert wiring is not None
    wiring.candidates = CandidateBook(
        JsonCandidateStore(tmp_path), FileCoreQuestions(CORE),
        lambda p: p in wiring.store.plans,  # type: ignore[attr-defined]
        30,
    )
    return TestClient(create_app(sessions))


def test_the_candidate_routes(tmp_path: Path) -> None:
    web = wired(tmp_path)
    plan = web.post("/interview/plans", json={"offer": OFFER}).json()
    base = f"/interview/plans/{plan['id']}/candidates"
    assert web.get(base).json()["candidates"] == []
    made = web.post(base, json={"name": "Ana Ruiz"})
    assert made.status_code == 201 and made.json()["id"] == "ana-ruiz"
    assert web.post(base, json={"name": " "}).status_code == 422
    assert web.post("/interview/plans/plan-x/candidates",
                    json={"name": "Ana"}).status_code == 404
    one = f"{base}/ana-ruiz"
    assert web.get(one).json()["answered"] == 0
    put = web.put(f"{one}/answers/work_authorization", json={"text": "Sí"})
    assert put.status_code == 200 and put.json()["answered"] == 1
    assert web.put(f"{one}/answers/shoe_size",
                   json={"text": "42"}).status_code == 404
    assert web.delete(f"{one}/answers/work_authorization").json()[
        "answered"] == 0
    started = web.post("/sessions", json={
        "product": "voicebot.interview", "audio": "mic",
        "plan": plan["id"], "candidate_id": "ana-ruiz"})
    assert started.status_code == 201
    assert started.json()["candidate_id"] == "ana-ruiz"
    found = web.get("/interview/candidates/by-session/vb-1")
    assert found.status_code == 200 and found.json()["id"] == "ana-ruiz"
    assert web.get("/interview/candidates/by-session/x").status_code == 404
    assert web.delete(one).json()["deleted"] is True
    assert web.delete(one).status_code == 404
    assert web.get(one).status_code == 404


def test_without_candidate_profiles_the_routes_say_unavailable() -> None:
    sessions = live([], wired=False)
    sessions.tts = FakeTts()
    web = TestClient(create_app(sessions))
    assert web.get("/interview/plans/p/candidates").status_code == 503
    assert web.get("/interview/candidates/by-session/s").status_code == 503


def test_the_cli_keeps_profiles_for_the_interview_retention(
    tmp_path: Path,
) -> None:
    products = load_products(PRODUCTS)
    assert retention_days(products) == 30
    assert retention_days({}) is None
    args = argparse.Namespace(
        interview_dir=tmp_path, knowledge="fake", reasoner="fake",
        knowledge_url="http://x", reasoner_url="http://x", token=None,
        fake_delay_ms=0.0,
    )
    wiring = build_interview(args, products, [])
    assert wiring is not None and wiring.candidates is not None
    book = wiring.candidates
    assert book.retention_days == 30
    assert len(book.questions()) == 5
    assert book.plan_exists("nope") is False


def test_proposals_are_confirmed_or_discarded_through_the_routes(
    tmp_path: Path,
) -> None:
    web = wired(tmp_path)
    plan = web.post("/interview/plans", json={"offer": OFFER}).json()
    base = f"/interview/plans/{plan['id']}/candidates"
    web.post(base, json={"name": "Ana Ruiz"})
    store = JsonCandidateStore(tmp_path)
    profile = store.load(plan["id"], "ana-ruiz")
    answers: List[Tuple[str, Dict[str, Value]]] = [
        ("work_authorization", {"eligibility.work_authorization": True}),
        ("salary_expectation", {"salary.expectation_min": 30000,
                                "salary.expectation_max": 30000}),
    ]
    for qid, fields in answers:
        changed = with_capture(profile, qid, Captured(fields), "s1", NOW)
        assert changed is not None
        profile = changed
    store.save(profile)
    one = f"{base}/ana-ruiz"
    assert web.get(one).json()["proposed"] == 2
    gone = web.delete(f"{one}/proposals/salary_expectation")
    assert gone.status_code == 200 and gone.json()["proposed"] == 1
    assert web.delete(f"{one}/proposals/salary_expectation"
                      ).status_code == 404
    bad = web.post(f"{one}/confirm", json={"questions": "all"})
    assert bad.status_code == 422
    done = web.post(f"{one}/confirm", json={})
    assert done.status_code == 200
    assert done.json()["answered"] == 1 and done.json()["proposed"] == 0
    assert web.post(f"{one}/confirm", json={
        "questions": ["motivation"]}).status_code == 404


def test_the_retention_applies_when_the_orchestrator_starts(
    tmp_path: Path,
) -> None:
    store = JsonCandidateStore(tmp_path)
    old = new_profile("plan-1", "Ana", [], NOW)
    store.save(old.model_copy(update={"updated_at": "2020-01-01T00:00:00"}))
    args = argparse.Namespace(
        interview_dir=tmp_path, knowledge="fake", reasoner="fake",
        knowledge_url="http://x", reasoner_url="http://x", token=None,
        fake_delay_ms=0.0,
    )
    build_interview(args, load_products(PRODUCTS), [])
    assert store.list("plan-1") == []


def test_every_stored_file_has_an_id_even_unreadable(tmp_path: Path) -> None:
    """#262: a file that cannot be read still owns its reports."""
    store = JsonCandidateStore(tmp_path)
    store.save(new_profile("dev", "Ana Ruiz", [], NOW))
    (tmp_path / "candidates" / "dev" / "broken.json").write_text("{", "utf-8")
    assert [p.id for p in store.list("dev")] == ["ana-ruiz"]
    assert store.ids("dev") == ["ana-ruiz", "broken"]
    assert store.ids("other") == []
