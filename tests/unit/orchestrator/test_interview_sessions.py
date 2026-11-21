"""Interview sessions of the voice bot, and the voice bot's topic (#208)."""

from typing import Any, List

import pytest

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.ports import Document
from orchestrator.application.sessions import LiveSessions, SessionError
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.live_choices import apply_live_choices
from tests.unit.orchestrator.fixtures import product, transcript
from tests.unit.orchestrator.test_interview_usecases import (
    OFFER,
    Handoff,
    Store,
)
from tests.unit.orchestrator.test_sessions import (
    Clock,
    FakeEvents,
    FakeTranscriber,
    InlineRunner,
    Transcripts,
    messages,
)
from tests.unit.orchestrator.test_voicebot_sessions import user_final


def live(events: List[Document], wired: bool = True,
         handoff: Any = None) -> LiveSessions:
    store = Store()
    reasoner = FakeReasoner()
    knowledge = FakeKnowledge([{"id": "m1", "text": "Pregunta STAR"}],
                              everything=True)
    planner = InterviewPlanner(store, knowledge, reasoner, None,
                               lambda: InterviewSection(), new_id=lambda:
                               "plan-1")
    return LiveSessions(
        {"voicebot": product("voicebot"),
         "voicebot.interview": product("voicebot.interview")},
        knowledge, reasoner, Transcripts({"t": transcript()}), Clock(),
        InlineRunner(), InlineExecutor, FakeTranscriber(),
        FakeEvents(events), None, lambda: "vb-1", tts=FakeTts(),
        interview=InterviewWiring(planner, store, reasoner,
                                  handoff or Handoff()) if wired else None,
    )


def test_the_interviewer_is_offered_only_with_its_wiring() -> None:
    names = {p["product"] for p in live([], wired=False).list_products()}
    assert names == {"voicebot"}
    shown = {p["product"]: p for p in live([]).list_products()}
    assert shown["voicebot.interview"]["interview"] is True
    assert shown["voicebot.interview"]["domain"] == "interview.generic"
    assert shown["voicebot"]["interview"] is False


def test_an_interview_session_speaks_first_and_hands_over() -> None:
    handoff = Handoff()
    sessions = live([
        user_final("t0.s0", "Estuve dos años en caja en un supermercado "
                   "del barrio atendiendo a clientes", 1),
        user_final("t1.s0", "No lo conozco.", 2),
        user_final("t2.s0", "No lo conozco.", 3),
        user_final("t3.s0", "No lo conozco.", 4),
        user_final("t4.s0", "No lo conozco.", 5),
        user_final("t5.s0", "Vale, perfecto entonces", 6),
    ], handoff=handoff)
    plan = sessions.interview.planner.create(  # type: ignore[union-attr]
        {"offer": OFFER})
    view = sessions.start_live("voicebot.interview", "mic", plan=plan["id"],
                               candidate="  Ana   García ")
    assert view["plan"] == "plan-1" and view["candidate"] == "Ana García"
    body = sessions.transcriber.bodies[0]  # type: ignore[union-attr]
    assert body["handoff"] == {"transcript_v1": False,
                               "run_post_call": False}
    assert body["speakers"] == [{"channel": 0, "id": "S1",
                                 "role": "candidate"}]
    found = messages(sessions, "vb-1")
    kinds = [m["type"] for m in found]
    turns = [m["record"] for m in found if m["type"] == "turn"]
    assert turns[0]["trigger"]["role"] == "bot"  # the opening
    assert kinds.count("coverage") == len(turns)
    assert "closed" in [m.get("state") for m in found
                        if m["type"] == "interview"]
    assert sessions.get("vb-1")["skipped"].get("closed", 0) >= 1
    result = [m for m in found if m["type"] == "interview_result"][0]
    assert result["sheet"]["candidate"] == "Ana García"
    assert result["sheet"]["closed_by"] == "covered"
    assert handoff.transcripts[0]["utterances"][0]["speaker"] == "BOT"
    assert all(r["stages"]["director_ms"] is not None for r in turns)
    rows = sessions.interview.planner.results(  # type: ignore[union-attr]
        "plan-1")["candidates"]
    assert rows[0]["candidate"] == "Ana García"


def test_interview_sessions_need_a_known_plan() -> None:
    sessions = live([])
    with pytest.raises(SessionError) as caught:
        sessions.start_live("voicebot.interview", "mic")
    assert caught.value.code == "invalid_request"
    with pytest.raises(SessionError) as caught:
        sessions.start_live("voicebot.interview", "mic", plan="plan-9")
    assert caught.value.code == "not_found"
    with pytest.raises(SessionError) as caught:
        sessions.start_live("voicebot", "mic", plan="plan-1")
    assert caught.value.code == "invalid_request"


def test_a_voicebot_session_takes_a_topic_domain() -> None:
    sessions = live([user_final("t0.s0", "¿Qué hago en la entrevista?", 1)])
    sessions.start_live("voicebot", "mic", domain="interview.generic")
    knowledge: FakeKnowledge = sessions.knowledge  # type: ignore
    assert knowledge.queries[-1]["domain"] == "interview.generic"
    for bad in ("Bad Domain", "../x"):
        with pytest.raises(SessionError):
            live([]).start_live("voicebot", "mic", domain=bad)
    with pytest.raises(SessionError):
        live([]).start_live("voicebot.interview", "mic", plan="plan-1",
                            domain="support.demo")


def test_saved_settings_tune_the_interviewer() -> None:
    config = product("voicebot.interview")
    tuned = apply_live_choices(config, {
        "interview.slot_filler": "heuristic", "interview.max_probes": "2",
        "interview.minutes": "30", "interview.post_call": "on",
        "voicebot.reasoner_model": "sonnet",
    })
    section = tuned.interview
    assert section is not None
    assert (section.filler, section.max_probes, section.minutes,
            section.post_call) == ("heuristic", 2, 30, True)
    assert tuned.reasoner is not None and tuned.reasoner.model == "sonnet"
    same = apply_live_choices(config, {"interview.max_probes": "9"})
    assert same.interview == config.interview
