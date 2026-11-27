"""LV-17 (#232): lines better avoided and the scripted interview demo."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.fs_demos import DirDemos
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.avoid import (
    MAX_FAILURES,
    NOTICES,
    PAUSE_S,
    AvoidTracker,
)
from orchestrator.application.ports import AvoidUnavailable, Document
from orchestrator.application.replay import transcript_to_events
from orchestrator.application.sessions import LiveSessions, SessionError
from orchestrator.domain.avoid import (
    REPHRASE_WINDOW_S,
    AvoidLedger,
    is_question,
    parse_hints,
)
from orchestrator.domain.demo import DemoScript, with_partials
from orchestrator.domain.events import LiveEvent
from tests.unit.orchestrator.fixtures import REPO, product, transcript
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    Transcripts,
)

DEMOS = REPO / "examples" / "demo"
AGE = {"category": "age", "title": "Edad", "why": "No influye.",
       "alternative": "Pregunta por la experiencia."}


def line(uid: str, speaker: str, text: str, start: float = 0.0,
         kind: str = "final") -> LiveEvent:
    """A final (or speaker update) event."""
    return LiveEvent.model_validate({
        "schema": "live.event.v1", "session_id": "s", "seq": 1,
        "type": kind, "at": "2026-11-24T10:00:00Z",
        "stream_time": start + 1.0,
        "utterance": {"id": uid, "turn": 0, "index": 0, "speaker": speaker,
                      "start": start, "end": start + 1.0, "text": text,
                      "revision": 0},
    })


# ---- domain -----------------------------------------------------------------

def test_a_flag_is_rephrased_by_a_clean_question_soon_after() -> None:
    ledger = AvoidLedger()
    assert ledger.record("u1", 10.0, "¿Qué edad tienes?", [AGE])
    assert ledger.flags[0].state == "flagged"
    assert not ledger.record("u2", 12.0, "Vale.", [])
    assert ledger.record("u3", 14.0, "¿Cuántos años llevas con Python?", [])
    assert ledger.view()[0]["state"] == "rephrased"
    assert ledger.view()[0]["rephrased_by"] == "u3"
    assert not ledger.record("u1", 10.0, "¿Qué edad tienes?", [AGE])


def test_results_out_of_order_give_the_same_flags() -> None:
    ledger = AvoidLedger()
    assert not ledger.record("u3", 14.0, "¿Y en Python?", [])
    assert ledger.record("u1", 10.0, "¿Qué edad tienes?", [AGE])
    assert ledger.flags[0].state == "rephrased"


def test_no_rephrase_outside_the_window_or_by_the_same_topic() -> None:
    ledger = AvoidLedger()
    ledger.record("u1", 0.0, "¿Qué edad tienes?", [AGE])
    ledger.record("u2", 5.0, "¿Pero cuántos años tienes?", [AGE])
    assert [f.state for f in ledger.flags] == ["flagged", "flagged"]
    ledger.record("u9", 5.0 + REPHRASE_WINDOW_S + 1, "¿Y Python?", [])
    assert [f.state for f in ledger.flags] == ["flagged", "flagged"]
    ledger.record("u3", 7.0, "¿Cuántos años con Python?", [])
    assert [f.state for f in ledger.flags] == ["rephrased", "rephrased"]


def test_hints_are_read_defensively() -> None:
    assert parse_hints({"hints": [AGE, {"category": "x"}, "junk", None]}) \
        == [AGE]
    assert parse_hints({}) == []
    ledger = AvoidLedger()
    assert not ledger.record("u1", 0.0, "Hola", [{"category": ""} | {
        "title": "t", "why": "w", "alternative": "a"}])
    assert is_question("¿Sí") and is_question("No?") and not is_question("No")


# ---- use case ---------------------------------------------------------------

class FakeAvoid:
    """Flags "edad"; fails on demand."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.asked: List[str] = []

    def check(self, text: str, language: str) -> Document:
        if self.fail:
            raise AvoidUnavailable("down")
        self.asked.append(text)
        return {"hints": [AGE] if "edad" in text else []}


def test_only_the_interviewer_is_checked_and_flags_are_published() -> None:
    port = FakeAvoid()
    published: List[Document] = []
    roles = {"S1": "interviewer", "S2": "candidate"}
    track = AvoidTracker(port, roles, published.append, InlineExecutor())
    track.on_event(line("u0", "S2", "Tengo treinta años de edad."))
    track.on_event(line("u1", "S1", "¿Qué edad tienes?", 10.0))
    track.on_event(line("u1", "S1", "¿Qué edad tienes?", 10.0))
    track.on_event(line("u2", "S1", "¿Cuántos años con Python?", 12.0))
    track.finish()
    assert port.asked == ["¿Qué edad tienes?", "¿Cuántos años con Python?"]
    assert [m["flags"][0]["state"] for m in published] == [
        "flagged", "rephrased"]
    assert track.view()[0]["category"] == "age" and track.checks == 2


def test_a_line_given_to_the_interviewer_later_is_checked() -> None:
    port = FakeAvoid()
    roles = {"S1": "interviewer", "S2": "candidate"}
    sent: List[Document] = []
    track = AvoidTracker(port, roles, sent.append, InlineExecutor())
    track.on_event(line("u1", "S2", "¿Qué edad tienes?"))
    track.on_event(line("u1", "S1", "", kind="speaker_update"))
    track.on_event(line("u5", "S1", "x", kind="partial"))
    assert port.asked == ["¿Qué edad tienes?"]


def test_failures_rest_then_retry_and_each_cause_is_said_once() -> None:
    port = FakeAvoid(fail=True)
    published: List[Document] = []
    track = AvoidTracker(port, {"S1": "interviewer"}, published.append,
                         InlineExecutor(), None)
    assert track.language == "es"
    for i in range(MAX_FAILURES):
        track.on_event(line(f"u{i}", "S1", "¿Qué edad tienes?", float(i)))
    assert not track.available
    assert [m["code"] for m in published] == ["avoid_unreachable"]
    track.on_event(line("u8", "S1", "¿Qué edad tienes?", 10.0))
    assert "u8" not in track._checked
    port.fail = False
    track.on_event(line("u9", "S1", "¿Qué edad tienes?", PAUSE_S + 5))
    assert track.available and port.asked == ["¿Qué edad tienes?"]
    assert published[-1]["type"] == "avoid"


def test_missing_rules_say_so_and_one_failure_does_not_stop() -> None:
    class NoRules(FakeAvoid):
        def check(self, text: str, language: str) -> Document:
            raise AvoidUnavailable("404", "no_rules")

    published: List[Document] = []
    track = AvoidTracker(NoRules(), {"S1": "interviewer"}, published.append,
                         InlineExecutor())
    track.on_event(line("u1", "S1", "¿Qué edad tienes?"))
    assert track.available
    assert published == [{"type": "notice", "code": "avoid_no_rules",
                          "message": NOTICES["no_rules"]}]


def test_a_line_given_to_the_candidate_later_loses_its_flag() -> None:
    published: List[Document] = []
    track = AvoidTracker(FakeAvoid(), {"S1": "interviewer",
                                       "S2": "candidate"},
                         published.append, InlineExecutor())
    track.on_event(line("u1", "S1", "¿Qué edad tienes?"))
    assert track.view()[0]["utterance"] == "u1"
    track.on_event(line("u1", "S2", "", kind="speaker_update"))
    assert track.view() == [] and published[-1]["flags"] == []
    ledger = AvoidLedger()
    ledger.record("u1", 0.0, "¿Qué edad tienes?", [AGE])
    ledger.record("u2", 3.0, "¿Y en Python?", [])
    assert ledger.drop("u2") is False
    assert ledger.flags[0].state == "flagged"


# ---- the demo file and its replay -------------------------------------------

def demo() -> DemoScript:
    return DirDemos(DEMOS).load("entrevista-lucia")


def test_the_demo_is_marked_and_lasts_about_a_minute() -> None:
    script = demo()
    assert script.demo is True and "ficticia" in script.note
    summary = script.summary()
    assert 60 <= float(str(summary["seconds"])) <= 70
    assert script.result_session == "demo-entrevista-lucia-marin"
    texts = " ".join(u.text for u in script.transcript.utterances)
    assert "edad" in texts and "reformulo" in texts


def test_partials_grow_before_each_final() -> None:
    script = demo()
    roles = {"entrevistador": "interviewer", "candidata": "candidate"}
    events = list(with_partials(transcript_to_events(script.transcript,
                                                     roles)))
    assert [e.seq for e in events] == list(range(len(events)))
    times = [e.stream_time for e in events]
    assert times == sorted(times)
    first = [e for e in events if e.utterance and e.utterance.id == "t0.s0"]
    assert [e.type for e in first][-1] == "final"
    texts = [e.utterance.text for e in first if e.utterance]
    assert all(b.startswith(a.rstrip(".,")) or len(b) >= len(a)
               for a, b in zip(texts, texts[1:]))
    assert all(e.timings is None for e in first if e.type == "partial")


def sessions(demos: Optional[DirDemos] = None,
             runner: Optional[InlineRunner] = None) -> LiveSessions:
    return LiveSessions(
        {"live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([]), FakeReasoner(), Transcripts({}), Clock(),
        runner or InlineRunner(), InlineExecutor,
        demos=demos if demos is not None else DirDemos(DEMOS),
    )


def test_the_demo_replays_its_script_without_any_model_call() -> None:
    live = sessions()
    assert [d["id"] for d in live.list_demos()] == ["entrevista-lucia"]
    view = live.start_replay("demo-1", "", "", speed=None,
                             demo="entrevista-lucia")
    assert view["mode"] == "demo" and view["demo"]["demo"] is True
    assert view["demo"]["result_session"] == "demo-entrevista-lucia-marin"
    found, _ = live.feed("demo-1").read(0, 0.01)
    messages = [m for _, m in found]
    records = [m["record"] for m in messages if m["type"] == "suggestion"]
    assert len(records) == 4 and {r["backend"] for r in records} == {
        "demo-script"}
    assert records[0]["suggestions"][0]["example"].startswith("¿Cómo supiste")
    avoid = [m["flags"] for m in messages if m["type"] == "avoid"]
    assert [f[0]["state"] for f in avoid] == ["flagged", "rephrased"]
    assert avoid[-1][0]["utterance"] == "t4.s0"
    topics = [m["topics"] for m in messages if m["type"] == "topics"]
    done = [sum(t["done"] for t in ts) for ts in topics]
    assert done == sorted(done) and sorted(set(done)) == [0, 1, 2, 3, 4, 5]
    # The concepts of each topic are ticked by their own lines (#258);
    # integration tests are never mentioned by Lucía, so they stay pending.
    last = {t["id"]: t for t in topics[-1]}
    assert [p["status"] for p in last["testing"]["points"]] == [
        "treated", "treated", "pending"]
    assert all(p["status"] == "treated" for p in last["python_apis"]["points"])
    assert [i for i, t in last.items() if t.get("current")] == ["teamwork"]
    # #286: levels by the live rule, concepts asked apart from treated,
    # and the next concept to ask.
    levels = {i: t["coverage"]["level"] for i, t in last.items()}
    assert levels == {"python_apis": "well_treated",
                      "databases": "well_treated", "experience": "treated",
                      "testing": "treated", "teamwork": "treated"}
    integration = last["testing"]["points"][2]
    assert integration["status"] == "pending"
    assert integration["asked"]["by"] == "rules"  # asked, no answer
    assert last["databases"]["points"][1]["asked"] is None  # not asked
    assert topics[0][0]["coverage"]["level"] == "none"
    order = ["none", "named", "treated", "well_treated"]
    for before, after in zip(topics, topics[1:]):
        for a, b in zip(before, after):
            assert order.index(b["coverage"]["level"]) >= order.index(
                a["coverage"]["level"])
    proposals = [m["next"] for m in messages if m["type"] == "topics"]
    assert proposals[0]["topic"] == "python_apis"
    assert proposals[-1] == {
        "topic": "testing", "topic_name": "Pruebas automáticas",
        "point": "integration", "text": "Pruebas de integración",
        "question": "Volviendo a pruebas de integración: ¿me lo puedes "
                    "concretar un poco más, con un ejemplo?",
        "again": True}
    asked = [next(t["id"] for t in ts if t.get("current"))
             for ts in topics[1:]]
    assert asked[0] == "python_apis" and "testing" in asked
    kinds = [m["event"]["type"] for m in messages if m["type"] == "event"]
    assert kinds.count("partial") > kinds.count("final") > 0
    assert messages[-1]["type"] == "status"
    assert live.get("demo-1")["state"] == "ended"
    assert live.get("demo-1")["handoff"] is None


def test_bad_demo_requests_are_refused(tmp_path: Path) -> None:
    live = sessions()
    with pytest.raises(SessionError) as missing:
        live.start_replay("d-2", "", "", demo="nope")
    assert missing.value.code == "not_found"
    with pytest.raises(SessionError):
        live.start_replay("d-2", "", "", speed=99, demo="entrevista-lucia")
    none = LiveSessions({}, FakeKnowledge([]), FakeReasoner(),
                        Transcripts({}), Clock(), InlineRunner(),
                        InlineExecutor)
    assert none.list_demos() == []
    with pytest.raises(SessionError):
        none.start_replay("d-3", "", "", demo="entrevista-lucia")


def test_the_demo_folder_skips_what_it_cannot_read(tmp_path: Path) -> None:
    data: Dict[str, Any] = json.loads(
        (DEMOS / "entrevista-lucia.demo.json").read_text("utf-8"))
    (tmp_path / "other.demo.json").write_text(json.dumps(data), "utf-8")
    (tmp_path / "broken.demo.json").write_text("{", "utf-8")
    wrong = {**data, "id": "lost",
             "topics": [{"id": "x", "name": "X", "covered_by": "t99.s0"}]}
    (tmp_path / "lost.demo.json").write_text(json.dumps(wrong), "utf-8")
    folder = DirDemos(tmp_path)
    assert folder.list() == []
    with pytest.raises(KeyError):
        folder.load("../etc")
    assert DirDemos(tmp_path / "none").list() == []
    empty = {**data, "id": "empty",
             "transcript": {**data["transcript"], "utterances": []},
             "topics": [], "suggestions": [], "avoid": []}
    with pytest.raises(ValueError):
        DemoScript.model_validate(empty)


def test_a_live_interview_gets_the_avoid_check() -> None:
    port = FakeAvoid()
    live = LiveSessions(
        {"live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([]), FakeReasoner(),
        Transcripts({"t": transcript([
            ("a", 0.5, 2.0, "Hola, ¿qué edad tienes?"),
            ("c", 2.5, 6.0, "Prefiero no decirlo."),
            ("a", 6.5, 9.0, "Claro, ¿cuántos años con Python?"),
        ])}), Clock(), InlineRunner(), InlineExecutor, avoid=port,
    )
    live.start_replay("i-1", "live_assist.interview", "t", speed=None)
    assert port.asked == ["Hola, ¿qué edad tienes?",
                          "Claro, ¿cuántos años con Python?"]
    assert live.get("i-1")["avoid"][0]["state"] == "rephrased"
    support = LiveSessions(
        {"live_assist.support": product()}, FakeKnowledge([]),
        FakeReasoner(), Transcripts({"t": transcript()}), Clock(),
        InlineRunner(), InlineExecutor, avoid=port,
    )
    support.start_replay("s-1", "live_assist.support", "t", speed=None)
    assert support.get("s-1")["avoid"] is None


def test_a_new_demo_stops_the_one_still_playing() -> None:
    runner = InlineRunner(hold=True)

    class RunningClock(Clock):
        """Waiting lets the held sessions run (they were told to stop)."""

        def sleep(self, seconds: float) -> None:
            super().sleep(seconds)
            while runner.jobs:
                runner.jobs.pop(0)()

    live = LiveSessions(
        {"live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([]), FakeReasoner(), Transcripts({}), RunningClock(),
        runner, InlineExecutor, demos=DirDemos(DEMOS),
    )
    live.start_replay("demo-a", "", "", demo="entrevista-lucia")
    assert live.get("demo-a")["state"] == "running"
    live.start_replay("demo-b", "", "", demo="entrevista-lucia")
    assert live.get("demo-a")["state"] == "ended"
    assert live.get("demo-b")["state"] == "running"


def test_a_real_session_stops_a_playing_demo_but_not_the_reverse() -> None:
    runner = InlineRunner(hold=True)

    class RunningClock(Clock):
        def sleep(self, seconds: float) -> None:
            super().sleep(seconds)
            while runner.jobs:
                runner.jobs.pop(0)()

    live = LiveSessions(
        {"live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([]), FakeReasoner(),
        Transcripts({"t": transcript()}), RunningClock(), runner,
        InlineExecutor, demos=DirDemos(DEMOS),
    )
    live.start_replay("demo-a", "", "", demo="entrevista-lucia")
    live.start_replay("real-1", "live_assist.interview", "t")
    assert live.get("demo-a")["state"] == "ended"
    assert live.get("real-1")["state"] == "running"
    with pytest.raises(SessionError) as busy:
        live.start_replay("demo-b", "", "", demo="entrevista-lucia")
    assert busy.value.code == "busy"
    assert live.get("real-1")["state"] == "running"
    with pytest.raises(SessionError) as taken:
        live.start_replay("real-1", "", "", demo="entrevista-lucia")
    assert taken.value.code == "wrong_state"
