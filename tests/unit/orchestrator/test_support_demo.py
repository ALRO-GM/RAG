"""#250: the scripted support call, its report and its voice example."""

import copy
import json
import re
from typing import Any, Dict, List

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.fs_demos import DirDemos
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.sessions import LiveSessions, SessionError
from orchestrator.domain.demo import DemoScript
from tests.unit.orchestrator.fixtures import REPO, product
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    Transcripts,
)

DEMOS = REPO / "examples" / "demo"
DOMAIN = REPO / "knowledge" / "domains" / "support.demo"
DEMO_ID = "llamada-lumbre"


def raw() -> Dict[str, Any]:
    """The demo file as plain data."""
    data: Dict[str, Any] = json.loads(
        (DEMOS / f"{DEMO_ID}.demo.json").read_text("utf-8"))
    return data


def script() -> DemoScript:
    return DirDemos(DEMOS).load(DEMO_ID)


def sessions() -> LiveSessions:
    return LiveSessions(
        {"live_assist.support": product("live_assist.support"),
         "live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([]), FakeReasoner(), Transcripts({}), Clock(),
        InlineRunner(), InlineExecutor, demos=DirDemos(DEMOS),
    )


# ---- the file --------------------------------------------------------------

def test_the_call_is_fictitious_and_has_a_natural_length() -> None:
    demo = script()
    assert demo.demo is True and "ficticia" in demo.note
    assert demo.product == "live_assist.support"
    assert 12 <= len(demo.transcript.utterances) <= 20
    agent, customer = demo.transcript.speakers
    said = " ".join(u.text for u in demo.transcript.utterances
                    if u.speaker == agent.id)
    assert re.search(r"\busted\b|\ble\b", said)
    assert not re.search(r"\b(tú|tienes|puedes)\b", said.lower())
    assert customer.name == "Andrés Ferrer"


def test_every_suggestion_cites_a_chunk_copied_from_the_domain() -> None:
    demo = script()
    fragments = {
        d["id"]: d for d in (
            json.loads(line) for line in (
                DOMAIN / "fragmentos.jsonl").read_text("utf-8").splitlines())
    }
    assert demo.sources
    for source in demo.sources:
        found = fragments[source.id]
        assert source.text == found["text"]
        assert source.metadata["intent"] == found["intent"]
    assert all(s.cites for s in demo.suggestions)
    assert {c for s in demo.suggestions for c in s.cites} <= {
        s.id for s in demo.sources}


def test_the_mandatory_items_are_those_of_the_domain() -> None:
    manifest = yaml.safe_load((DOMAIN / "manifest.yaml").read_text("utf-8"))
    listed = manifest["mandatory"]
    demo = script().mandatory
    assert demo is not None and demo.domain == "support.demo"
    assert [(i.id, i.text, i.phase) for i in demo.items] == [
        (i["id"], i["text"], i["phase"]) for i in listed["items"]]
    assert demo.version == listed["version"]
    assert demo.remind_seconds == listed["remind_seconds"]
    said = {i.id: i.said_by for i in demo.items}
    assert said["aceptacion-expresa"] is None
    assert all(said[k] for k in ("aviso-datos", "condiciones-contrato",
                                 "desistimiento"))
    spoken = {u.id: u for u in script().transcript.utterances}
    assert spoken[said["aviso-datos"] or ""].start > demo.remind_seconds


def test_the_report_has_every_part_in_both_languages() -> None:
    demo = script()
    report = demo.report
    assert report is not None
    assert report.missing_english() == []
    assert report.top_scorers.illustrative is True
    assert "ilustrativo" in report.top_scorers.note
    assert {r.effect for r in report.lines_review} == {
        "helped", "neutral", "hurt"}
    assert {s.status for s in report.script.steps} == {
        "followed", "reordered", "skipped"}
    assert {m.outcome for m in report.mandatory} == {
        "said", "said_late", "never"}
    assert report.script.verdict.improvable is True
    assert report.unanswered and len(report.summary.strengths) == 2
    assert demo.voicebot is not None
    assert all(e.en and e.en.get("bot") and e.en.get("user")
               for e in demo.voicebot.exchanges)


def test_the_report_judges_the_agent_by_the_reaction() -> None:
    demo = script()
    report = demo.report
    assert report is not None
    who = {u.id: u for u in demo.transcript.utterances}
    agent = demo.transcript.speakers[0].id
    for review in report.lines_review:
        assert who[review.utterance].speaker == agent
        assert who[review.reaction].speaker != agent
        assert who[review.reaction].start > who[review.utterance].start
    assert any(r.effect == "hurt" and r.better for r in report.lines_review)


def test_the_texts_are_coaching_not_ranking() -> None:
    text = json.dumps(raw(), ensure_ascii=False).lower()
    for word in ("ranking", "despid", "contratar a", "puntuaci", "scored"):
        assert word not in text, word


# ---- validation ------------------------------------------------------------

def test_the_demo_refuses_what_does_not_add_up() -> None:
    base = raw()

    def broken(change: Any) -> None:
        data = copy.deepcopy(base)
        change(data)
        with pytest.raises(ValidationError):
            DemoScript.model_validate(data)

    broken(lambda d: d["suggestions"][0]["cites"].append("zz"))
    broken(lambda d: d["report"]["lines_review"][0].update(utterance="t99.s0"))
    broken(lambda d: d["mandatory"]["items"][0].update(said_by="t99.s0"))
    broken(lambda d: d["report"]["mandatory"][0].update(utterance="t0.s0"))
    broken(lambda d: d["report"]["mandatory"][0].update(item="zz"))
    broken(lambda d: d["report"]["top_scorers"].update(illustrative=False))
    broken(lambda d: d["voicebot"]["exchanges"][0].update(cites=["zz"]))
    hurt = next(r for r in base["report"]["lines_review"]
                if r["effect"] == "hurt")

    def drop_better(d: Dict[str, Any]) -> None:
        for r in d["report"]["lines_review"]:
            if r["effect"] == "hurt":
                r.pop("better")

    assert hurt["better"]
    broken(drop_better)

    def skipped_with_lines(d: Dict[str, Any]) -> None:
        for s in d["report"]["script"]["steps"]:
            if s["status"] == "skipped":
                s["about"] = ["t0.s0"]

    broken(skipped_with_lines)


def test_a_demo_without_a_report_still_lists_and_plays() -> None:
    data = raw()
    for key in ("report", "voicebot", "mandatory", "sources"):
        data.pop(key)
    data["suggestions"] = []
    plain = DemoScript.model_validate(data)
    assert plain.summary()["report"] is False
    assert plain.summary()["voicebot"] is False


# ---- the replay ------------------------------------------------------------

def test_the_listing_marks_the_report_and_the_voice_example() -> None:
    listed = {d["id"]: d for d in sessions().list_demos()}
    assert set(listed) == {"entrevista-lucia", DEMO_ID}
    assert listed[DEMO_ID]["report"] is True
    assert listed[DEMO_ID]["voicebot"] is True
    assert listed["entrevista-lucia"]["report"] is False


def play() -> List[Dict[str, Any]]:
    live = sessions()
    live.start_replay("demo-s", "", "", speed=None, demo=DEMO_ID)
    found, _ = live.feed("demo-s").read(0, 0.01)
    return [m for _, m in found]


def test_the_support_demo_publishes_signals_and_sources() -> None:
    messages = play()
    records = [m["record"] for m in messages if m["type"] == "suggestion"]
    assert len(records) == len(script().suggestions)
    assert {r["backend"] for r in records} == {"demo-script"}
    first = records[0]
    suggestion = first["suggestions"][0]
    assert suggestion["cites"] == ["s18"]
    assert first["retrieved"][0]["id"] == "s18"
    assert "cancelada antes de su renovación" in first["retrieved"][0]["text"]
    quoted = suggestion["expressed"]
    assert {e["signal"] for e in quoted} == {"complaint", "churn_risk"}
    assert all(e["quote"].startswith("Hola, buenas") for e in quoted)
    assert all(e["utterance"] == "t1.s0" for e in quoted)
    # A suggestion that expresses no signal carries none.
    assert "expressed" not in records[3]["suggestions"][0]


def test_the_checklist_goes_to_said_and_one_is_never_said() -> None:
    messages = play()
    lists = [m["checklist"] for m in messages if m["type"] == "mandatory"]
    assert [sum(i["state"] == "said" for i in c["items"])
            for c in lists] == [0, 1, 2, 3]
    assert {i["id"]: i["state"] for i in lists[0]["items"]} == {
        "aviso-datos": "pending", "condiciones-contrato": "pending",
        "desistimiento": "pending", "aceptacion-expresa": "pending"}
    assert all(i["score"] is None for c in lists for i in c["items"])
    order = [next(i["id"] for i in b["items"]
                  if i["state"] == "said"
                  and i["id"] not in [j["id"] for j in a["items"]
                                      if j["state"] == "said"])
             for a, b in zip(lists, lists[1:])]
    assert order == ["aviso-datos", "condiciones-contrato", "desistimiento"]
    handoff = [m["handoff"] for m in messages if m["type"] == "handoff"]
    assert len(handoff) == 1 and handoff[0]["demo"] is True
    summary = handoff[0]["mandatory"]
    assert summary["said"] == ["aviso-datos", "condiciones-contrato",
                               "desistimiento"]
    assert [m["id"] for m in summary["missing"]] == ["aceptacion-expresa"]
    assert messages[-1]["type"] == "status"


def test_the_interview_demo_is_unchanged_by_the_support_one() -> None:
    live = sessions()
    live.start_replay("demo-i", "", "", speed=None, demo="entrevista-lucia")
    found, _ = live.feed("demo-i").read(0, 0.01)
    kinds = {m["type"] for _, m in found}
    assert "mandatory" not in kinds and "handoff" not in kinds


# ---- the report and the voice example --------------------------------------

def test_the_report_and_the_voice_example_are_served() -> None:
    live = sessions()
    report = live.demo_part(DEMO_ID, "report")
    assert report["demo"] is True and report["id"] == DEMO_ID
    assert report["top_scorers"]["illustrative"] is True
    assert len(report["utterances"]) == 19
    assert report["speakers"][0]["name"] == "Mario"
    assert [i["id"] for i in report["mandatory_items"]][-1] \
        == "aceptacion-expresa"
    assert all("said_by" not in i for i in report["mandatory_items"])
    assert report["sources"][0]["id"]
    voice = live.demo_part(DEMO_ID, "voicebot")
    assert len(voice["exchanges"]) == 4
    assert voice["sources"] == report["sources"]
    # The voice example keeps its own title and English texts, not the call's.
    assert voice["title"] == "Hablar con el asistente de voz de Lumbre Hogar"
    assert voice["en"]["title"].startswith("Talking to the")
    assert voice["en"]["intro"].startswith("A customer talks")
    assert report["title"].startswith("Llamada")
    assert report["en"]["title"].startswith("Call")


@pytest.mark.parametrize("demo_id,part", [
    ("nope", "report"), (DEMO_ID, "other"), ("entrevista-lucia", "report"),
    ("entrevista-lucia", "voicebot"),
])
def test_a_missing_demo_or_part_is_not_found(demo_id: str, part: str) -> None:
    with pytest.raises(SessionError) as failed:
        sessions().demo_part(demo_id, part)
    assert failed.value.code == "not_found"


def test_no_demos_no_parts() -> None:
    bare = LiveSessions(
        {"live_assist.support": product("live_assist.support")},
        FakeKnowledge([]), FakeReasoner(), Transcripts({}), Clock(),
        InlineRunner(), InlineExecutor,
    )
    with pytest.raises(SessionError):
        bare.demo_part(DEMO_ID, "report")


def test_the_http_routes_serve_the_parts() -> None:
    web = TestClient(create_app(sessions(), None, keepalive=0.01))
    assert web.get(f"/demos/{DEMO_ID}/report").json()["demo"] is True
    assert web.get(f"/demos/{DEMO_ID}/voicebot").status_code == 200
    assert web.get("/demos/nope/report").status_code == 404
    assert web.get(f"/demos/{DEMO_ID}/other").status_code == 404


def test_the_demo_is_a_demo_session_a_real_one_can_replace() -> None:
    live = sessions()
    live.start_replay("demo-s", "", "", speed=None, demo=DEMO_ID)
    assert live.get("demo-s")["mode"] == "demo"


def test_the_call_matches_what_the_knowledge_base_says() -> None:
    demo = script()
    text = " ".join(u.text for u in demo.transcript.utterances).lower()
    # Lumbre Plus renews monthly (s17): no 12-month contract is offered.
    assert "doce meses" not in text and "12 meses" not in text
    sources = {s.id: s.text for s in demo.sources}
    assert "se renueva cada mes" in sources["s17"].lower()
    assert "no ha usado el envío gratuito" in text


def test_sources_have_readable_titles_and_items_an_english_name() -> None:
    demo = script()
    assert all(s.title and s.en and s.en.get("title") for s in demo.sources)
    assert all(not s.title.startswith("s") or " " in s.title
               for s in demo.sources)
    assert demo.mandatory is not None
    assert all(i.en and i.en.get("text") for i in demo.mandatory.items)
    assert demo.report is not None
    late = next(m for m in demo.report.mandatory
                if m.outcome == "said_late")
    assert "derechos" in late.note and late.en
    assert "rights" in late.en["note"]


def test_the_cited_chunks_carry_their_title_to_the_page() -> None:
    first = play()
    cites = [m["record"]["retrieved"][0]["metadata"]["title"]
             for m in first if m["type"] == "suggestion"]
    assert cites[0] == "Baja antes de la renovación y devolución del cobro"
