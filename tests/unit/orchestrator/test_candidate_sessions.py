"""The candidate across rounds in live sessions and the AI interviewer (#235).

Two rounds of one synthetic candidate: the first captures answers to
the core questions, the second shows them as answered, does not capture
them again and drops suggestions that would ask them. The AI interviewer
asks the open core questions (not the one with blanks) and never the
answered ones.
"""

from typing import Any, Dict, List, Optional

import pytest

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.candidates import CandidateBook
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.ports import CancelToken, Document
from orchestrator.application.sessions import LiveSessions, SessionError
from orchestrator.domain.config import InterviewSection
from tests.unit.orchestrator.fixtures import product, transcript
from tests.unit.orchestrator.test_candidate import Core, MemoryCandidates
from tests.unit.orchestrator.test_interview_sessions import PacedEvents
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


class AskingReasoner(FakeReasoner):
    """Suggests asking for the work permit, and a follow-up."""

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        if request.get("role") != "suggester":
            return super().reason(request, cancel)
        return {
            "schema": "reasoner.response.v1",
            "request_id": request["request_id"], "role": "suggester",
            "status": "ok", "model": "fake", "backend": "local",
            "latency": {"total_ms": 1.0},
            "suggestions": [
                {"kind": "question", "text": "Pregunta por el permiso",
                 "example": "¿Tienes autorización para trabajar en España?"},
                {"kind": "follow_up", "text": "Pide un ejemplo concreto",
                 "example": "¿Qué hiciste tú exactamente?"},
            ],
        }


def event(seq: int, kind: str, speaker: str = "S1", text: str = "",
          session: str = "r1") -> Document:
    document: Document = {
        "schema": "live.event.v1", "session_id": session, "seq": seq,
        "type": kind, "at": "2026-11-24T10:00:00Z",
        "stream_time": float(seq),
    }
    if kind == "final":
        document["utterance"] = {
            "id": f"t{seq}.s0", "turn": seq, "index": 0, "speaker": speaker,
            "start": float(seq), "end": seq + 0.9, "text": text,
            "revision": 0,
        }
    return document


def conversation(session: str, lines: List[List[str]]) -> List[Document]:
    found = [event(n, "final", who, text, session)
             for n, (who, text) in enumerate(lines)]
    return found + [event(len(lines), "session_ended", session=session)]


def wired(documents: List[Document], reasoner: Any = None,
          store: Optional[MemoryCandidates] = None
          ) -> LiveSessions:
    plans = Store()
    reasoner = reasoner or FakeReasoner()
    knowledge = FakeKnowledge([{"id": "m1", "text": "STAR"}],
                              everything=True)
    planner = InterviewPlanner(plans, knowledge, reasoner, None,
                               lambda: InterviewSection(),
                               new_id=lambda: "plan-1")
    planner.create({"offer": OFFER})
    book = CandidateBook(store or MemoryCandidates(), Core(),
                         lambda p: p in plans.plans, 30)
    return LiveSessions(
        {"live_assist.interview": product("live_assist.interview"),
         "live_assist.support": product()},
        knowledge, reasoner, Transcripts({"t": transcript()}), Clock(),
        InlineRunner(), InlineExecutor, FakeTranscriber(),
        FakeEvents(documents),
        interview=InterviewWiring(planner, plans, reasoner, Handoff(),
                                  book),
    )


ROUND_ONE = [
    ["S1", "Buenos días, empezamos. ¿Tienes autorización para trabajar "
           "en España?"],
    ["S2", "Sí, la tengo."],
    ["S1", "¿Qué te ha llevado a presentarte a este puesto?"],
    ["S2", "Me gusta mucho el producto y quiero trabajar con un equipo "
           "que cuida la calidad del código."],
    ["S1", "¿Qué expectativa salarial tienes?"],
    ["S2", "Prefiero que me digas primero la banda."],
]


def test_answers_are_kept_and_the_next_round_shows_them() -> None:
    store = MemoryCandidates()
    first = wired(conversation("r1", ROUND_ONE), AskingReasoner(), store)
    view = first.start_live("live_assist.interview", "stereo",
                            session_id="r1", plan="plan-1",
                            candidate="Lucía Marín")
    assert view["candidate"] == "Lucía Marín"
    assert view["candidate_id"] == "lucia-marin"
    kept = store.kept[("plan-1", "lucia-marin")]
    assert set(kept.answers) == {"work_authorization", "motivation"}
    assert kept.answers["work_authorization"].quote == ""
    assert [r.session_id for r in kept.rounds] == ["r1"]
    core = [m for m in messages(first, "r1") if m["type"] == "core"]
    assert core[0]["profile"]["answered"] == 0
    assert [m.get("captured") for m in core[1:]] == [
        "work_authorization", "motivation"]

    second = wired(conversation("r2", [
        ["S1", "¿Tienes autorización para trabajar en España?"],
        ["S2", "No."],
        ["S1", "Cuéntame un proyecto difícil."],
        ["S2", "Migré una base de datos grande sin parar el servicio."],
    ]), AskingReasoner(), store)
    second.start_live("live_assist.interview", "stereo", session_id="r2",
                      plan="plan-1", candidate_id="lucia-marin")
    kept = store.kept[("plan-1", "lucia-marin")]
    # Answered in round one: not captured again (the «No» is not taken).
    assert kept.answers["work_authorization"].fields == {
        "eligibility.work_authorization": True}
    assert [r.session_id for r in kept.rounds] == ["r1", "r2"]
    shown = [m for m in messages(second, "r2") if m["type"] == "core"]
    assert shown[0]["profile"]["answered"] == 2
    records = [m["record"] for m in messages(second, "r2")
               if m["type"] == "suggestion" and m["record"]["suggestions"]]
    examples = [s["example"] for r in records for s in r["suggestions"]]
    assert examples and "¿Tienes autorización para trabajar en España?" \
        not in examples


def test_a_live_interview_without_a_role_keeps_nothing() -> None:
    store = MemoryCandidates()
    live = wired(conversation("r1", ROUND_ONE), store=store)
    view = live.start_live("live_assist.interview", "stereo",
                           session_id="r1")
    assert view["candidate_id"] is None and store.kept == {}
    assert not [m for m in messages(live, "r1") if m["type"] == "core"]


def test_wrong_roles_and_candidates_are_refused() -> None:
    live = wired([])
    with pytest.raises(SessionError) as caught:
        live.start_live("live_assist.support", "stereo", plan="plan-1")
    assert caught.value.code == "invalid_request"
    with pytest.raises(SessionError) as caught:
        live.start_live("live_assist.interview", "stereo", plan="plan-9",
                        candidate="Ana")
    assert caught.value.code == "not_found"
    with pytest.raises(SessionError) as caught:
        live.start_live("live_assist.interview", "stereo", plan="plan-1",
                        candidate_id="nobody")
    assert caught.value.code == "not_found"
    with pytest.raises(SessionError) as caught:
        live.start_live("live_assist.interview", "stereo", plan="plan-1",
                        candidate="   ")
    assert caught.value.code in ("invalid_request", "not_found")


def bot(events: List[Document], store: MemoryCandidates) -> LiveSessions:
    plans = Store()
    reasoner = FakeReasoner()
    knowledge = FakeKnowledge([{"id": "m1", "text": "STAR"}],
                              everything=True)
    planner = InterviewPlanner(plans, knowledge, reasoner, None,
                               lambda: InterviewSection(),
                               new_id=lambda: "plan-1")
    book = CandidateBook(store, Core(), lambda p: p in plans.plans, 30)
    paced = PacedEvents(events)
    sessions = LiveSessions(
        {"voicebot.interview": product("voicebot.interview")},
        knowledge, reasoner, Transcripts({}), Clock(), InlineRunner(),
        InlineExecutor, FakeTranscriber(), paced, None, lambda: "vb-1",
        tts=FakeTts(),
        interview=InterviewWiring(planner, plans, reasoner, Handoff(),
                                  book),
    )
    paced.sessions = sessions
    planner.create({"offer": OFFER})
    return sessions


def asked(sessions: LiveSessions) -> List[Dict[str, Any]]:
    """The payload of every interviewer turn the reasoner was asked for."""
    reasoner: FakeReasoner = sessions.reasoner  # type: ignore[assignment]
    return [r["payload"] for r in reasoner.requests
            if r.get("template") == "interviewer@1"]


def test_the_ai_interviewer_asks_the_open_core_questions_first() -> None:
    store = MemoryCandidates()
    sessions = bot([
        user_final("t0.s0", "Sí, tengo permiso de trabajo.", 1),
        user_final("t1.s0", "Podría empezar en dos semanas.", 2),
        user_final("t2.s0", "Me gusta el deporte y quiero trabajar "
                   "atendiendo a clientes en tienda.", 3),
        user_final("t3.s0", "Estuve dos años en caja en un supermercado "
                   "del barrio atendiendo a clientes", 4),
        *[user_final(f"t{n}.s0", "No lo conozco.", n + 1)
          for n in range(4, 10)],
        user_final("t10.s0", "Unos 22.000 euros al año.", 11),
        user_final("t11.s0", "Vale, gracias.", 12),
    ], store)
    sessions.start_live("voicebot.interview", "mic", plan="plan-1",
                        candidate="Ana García")
    payloads = asked(sessions)
    topics = [p.get("topic", "") for p in payloads]
    assert payloads[0]["action"] == "open"
    assert topics[0].startswith("¿Tienes autorización")
    assert topics[1].startswith("¿Cuándo podrías incorporarte")
    assert topics[2].startswith("¿Qué te ha llevado")
    assert payloads[3]["action"] == "ask"  # the plan's first slot, not «open»
    assert not any("[" in t for t in topics)  # work mode needs the job's data
    kept = store.kept[("plan-1", "ana-garcia")]
    assert {"work_authorization", "start_date_notice",
            "motivation"} <= set(kept.answers)
    assert any(t.startswith("¿Qué expectativa salarial") for t in topics)
    assert kept.answers["salary_expectation"].fields[
        "salary.expectation_min"] == 22000
    assert kept.rounds[0].kind == "bot"


def test_the_ai_interviewer_skips_what_is_answered() -> None:
    store = MemoryCandidates()
    sessions = bot([
        user_final("t0.s0", "Estuve dos años en caja en un supermercado "
                   "del barrio atendiendo a clientes", 1),
        user_final("t1.s0", "Vale, perfecto entonces", 2),
    ], store)
    assert sessions.interview is not None
    book = sessions.interview.candidates
    assert book is not None
    made = book.create("plan-1", "Ana")
    for question, text in (("work_authorization", "Sí"),
                           ("start_date_notice", "De inmediato"),
                           ("motivation", "Le gusta el sector"),
                           ("salary_expectation", "22.000")):
        book.correct("plan-1", made["id"], question, text)
    sessions.start_live("voicebot.interview", "mic", plan="plan-1",
                        candidate_id=made["id"])
    topics = [p.get("topic", "") for p in asked(sessions)]
    assert not any(t.startswith(("¿Tienes autorización", "¿Cuándo podrías",
                                 "¿Qué te ha llevado", "¿Qué expectativa"))
                   for t in topics)
    assert asked(sessions)[0]["action"] == "open"
