"""#294 adapters: the owner's lists on disk, the sent list and the review
over HTTP and with the word check, the orchestrator's routes, the CLI
wiring and the offer page's table."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.cli import build_lists
from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.fs_mandatory import FileMandatory
from orchestrator.adapters.outbound.fs_mandatory_lists import (
    JsonMandatoryLists,
)
from orchestrator.adapters.outbound.http_mandatory import HttpMandatory
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.mandatory_lists import MandatoryLists
from orchestrator.application.ports import MandatoryUnavailable
from orchestrator.application.sessions import LiveSessions
from orchestrator.domain.mandatory_lists import OwnerList, with_item_ids
from orchestrator.domain.offer_board import interview_view
from tests.unit.orchestrator.fixtures import PRODUCTS, REPO, product
from tests.unit.orchestrator.test_mandatory_lists import ITEMS
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    Transcripts,
)

DOMAINS = REPO / "knowledge" / "domains"
NOW = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)


def owner(version: int = 1, name: str = "Verano") -> OwnerList:
    return OwnerList.model_validate({
        "id": "verano", "name": name, "kind": "support",
        "version": version, "items": with_item_ids(ITEMS)})


def test_lists_are_kept_with_every_version(tmp_path: Path) -> None:
    store = JsonMandatoryLists(tmp_path / "m")
    assert store.list() == [] and store.offers() == {}
    store.save(owner())
    store.save(owner(2, "Verano 2026"))
    assert store.load("verano").name == "Verano 2026"
    assert store.load("verano", 1).name == "Verano"
    current = tmp_path / "m" / "lists" / "verano.json"
    assert current.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "m" / "lists" / "verano" / "v1.json").exists()
    (tmp_path / "m" / "lists" / "broken.json").write_text("{")
    assert [o.id for o in store.list()] == ["verano"]
    for bad in ("../x", "Bad", ""):
        with pytest.raises(KeyError):
            store.load(bad)
    with pytest.raises(KeyError):
        store.load("verano", 9)


def test_offers_name_a_list_and_lose_it_with_the_list(tmp_path: Path
                                                      ) -> None:
    store = JsonMandatoryLists(tmp_path)
    store.save(owner())
    store.assign("plan-1", "verano")
    store.assign("plan-2", "verano")
    assert store.offer_list("plan-1") == "verano"
    assert store.offers() == {"plan-1": "verano", "plan-2": "verano"}
    store.assign("plan-2", None)
    assert store.offer_list("plan-2") is None
    (tmp_path / "offers" / "plan-3.json").write_text('{"list": "../x"}')
    assert store.offer_list("plan-3") is None
    with pytest.raises(KeyError):
        store.assign("../p", "verano")
    with pytest.raises(KeyError):
        store.assign("plan-1", "Bad Id")
    assert store.delete("verano") is True
    assert store.offer_list("plan-1") is None
    assert store.delete("verano") is False
    assert not (tmp_path / "lists" / "verano").exists()


def test_the_word_check_takes_a_sent_list_and_reviews_it() -> None:
    port = FileMandatory(DOMAINS)
    listed = {"version": 3, "items": with_item_ids(ITEMS)}
    said = port.check_list(listed, ITEMS[0]["must_say"][0])
    assert [i["id"] for i in said["items"] if i["said"]] == [
        "precio-de-la-tarifa-verano"]
    assert said["version"] == 3
    close = port.review({"items": [
        {"id": "a", "must_say": ["la tarifa cuesta veinte euros"]},
        {"id": "b", "must_say": ["la tarifa cuesta veinte euros al mes"]},
        {"id": "c", "must_say": ["algo muy distinto de lo anterior"]},
    ]})
    assert {(c["item"], c["other"]) for c in close["close"]} == {
        ("a", "b"), ("b", "a")}
    assert port.review({"items": [{"id": "a", "must_say": ["x"]}]}
                       )["close"] == []


def test_a_sent_list_and_its_review_go_to_the_rag() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"items": [], "close": []})

    port = HttpMandatory("http://rag:8000", "tok", client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    port.check_list({"items": []}, "hola")
    port.review({"items": []})
    assert [r.url.path for r in seen] == ["/knowledge/mandatory",
                                          "/knowledge/mandatory/review"]
    assert json.loads(seen[0].content) == {"list": {"items": []},
                                           "text": "hola"}
    assert json.loads(seen[1].content) == {"list": {"items": []}}
    down = HttpMandatory("http://x", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(MandatoryUnavailable):
        down.review({})


def app(tmp_path: Path, with_lists: bool = True) -> TestClient:
    port = FileMandatory(DOMAINS)
    lists = MandatoryLists(
        JsonMandatoryLists(tmp_path), port,
        {"support": "support.demo", "interview": "interview.generic"},
        lambda: NOW, plan_exists=lambda plan: plan.startswith("plan"))
    sessions = LiveSessions(
        {"live_assist.support": product()}, FakeKnowledge(), FakeReasoner(),
        Transcripts({}), Clock(), InlineRunner(), InlineExecutor,
        mandatory=port, lists=lists if with_lists else None)
    return TestClient(create_app(sessions, None))


def test_the_list_routes(tmp_path: Path) -> None:
    web = app(tmp_path)
    seen = web.get("/mandatory/lists").json()
    assert [x["id"] for x in seen["lists"]] == ["general.support",
                                                "general.interview"]
    made = web.post("/mandatory/lists", json={
        "name": "Verano", "kind": "support", "items": ITEMS})
    assert made.status_code == 201 and made.json()["id"] == "verano"
    assert "warnings" in made.json()
    one = web.get("/mandatory/lists/verano").json()
    assert one["version"] == 1 and one["offers"] == []
    edit = web.put("/mandatory/lists/verano",
                   json={**one, "name": "Verano 2026"})
    assert edit.status_code == 200 and edit.json()["version"] == 2
    stale = web.put("/mandatory/lists/verano", json=one)
    assert stale.status_code == 409 and stale.json()["code"] == "conflict"
    general = web.get("/mandatory/lists/general.support").json()
    assert general["general"] is True and len(general["items"]) == 4
    copy = web.post("/mandatory/lists", json={"from": "general.interview",
                                              "name": "Panadería"})
    assert copy.json()["mode"] == "replace"
    review = web.post("/mandatory/review", json={
        "kind": "support", "items": ITEMS})
    assert review.status_code == 200 and review.json()["short"]
    assert web.put("/interview/plans/plan-1/mandatory",
                   json={"list": "panaderia"}).json()["list"] == "panaderia"
    assert web.get("/interview/plans/plan-1/mandatory").json()[
        "name"] == "Panadería"
    assert web.put("/interview/plans/plan-1/mandatory",
                   json={"list": 3}).status_code == 422
    assert web.put("/interview/plans/x/mandatory",
                   json={"list": None}).status_code == 404
    assert web.post("/mandatory/lists", json={
        "name": "x", "kind": "x", "items": ITEMS}).status_code == 422
    assert web.post("/mandatory/lists",
                    content=b"x" * (97 * 1024)).status_code == 422
    assert web.delete("/mandatory/lists/verano").json()["deleted"] is True
    assert web.get("/mandatory/lists/verano").status_code == 404
    assert web.delete("/mandatory/lists/general.support").status_code == 422


def test_without_lists_the_routes_say_so(tmp_path: Path) -> None:
    web = app(tmp_path, with_lists=False)
    assert web.get("/mandatory/lists").status_code == 503


def test_the_cli_wires_the_general_list_of_each_kind(tmp_path: Path
                                                     ) -> None:
    from orchestrator.adapters.inbound.cli import load_products
    args = argparse.Namespace(mandatory_dir=tmp_path / "m",
                              interview_dir=tmp_path / "i")
    lists = build_lists(args, load_products(PRODUCTS),
                        FileMandatory(DOMAINS))
    assert lists.domains == {"support": "support.demo",
                             "interview": "interview.generic"}
    assert lists.plan_exists("plan-1") is False
    with pytest.raises(Exception):
        lists.offer("plan-1")


def test_the_offer_table_names_the_list_of_a_round() -> None:
    from tests.unit.orchestrator.test_offer_board import live, offer, slot
    plan = offer()
    sheet: Dict[str, Any] = live("l-1", [slot("s1")])
    view = interview_view(plan, {**sheet, "mandatory_list": {
        "id": "verano", "name": "Verano", "version": 2, "general": False}})
    assert view.mandatory_list == {"id": "verano", "name": "Verano",
                                   "version": 2, "general": False}
    assert interview_view(plan, {**sheet, "mandatory_list": "junk"}
                          ).mandatory_list is None
