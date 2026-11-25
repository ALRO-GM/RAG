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
        self.requests.append(request)
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
    # Captured as proposals (the motivation never is), nothing answered.
    assert set(kept.proposals) == {"work_authorization"}
    assert kept.answers == {}
    assert [r.session_id for r in kept.rounds] == ["r1"]
    core = [m for m in messages(first, "r1") if m["type"] == "core"]
    assert core[0]["profile"]["answered"] == 0
    assert [m.get("captured") for m in core[1:]] == ["work_authorization"]
    # Unconfirmed, nothing is screened yet; the interviewer confirms it
    # and types the motivation.
    assert first.interview is not None
    book = first.interview.candidates
    assert book is not None
    book.confirm("plan-1", "lucia-marin")
    book.correct("plan-1", "lucia-marin", "motivation", "Calidad del código")

    second = wired(conversation("r2", [
        ["S1", "¿Tienes autorización para trabajar en España?"],
        ["S2", "No."],
        ["S1", "Cuéntame un proyecto difícil."],
        ["S2", "Migré una base de datos grande sin parar el servicio."],
    ]), AskingReasoner(), store)
    second.start_live("live_assist.interview", "stereo", session_id="r2",
                      plan="plan-1", candidate_id="lucia-marin")
    kept = store.kept[("plan-1", "lucia-marin")]
    # Confirmed after round one: not captured again (the «No» is not
    # even proposed).
    assert kept.answers["work_authorization"].fields == {
        "eligibility.work_authorization": True}
    assert kept.proposals == {}
    assert [r.session_id for r in kept.rounds] == ["r1", "r2"]
    shown = [m for m in messages(second, "r2") if m["type"] == "core"]
    assert shown[0]["profile"]["answered"] == 2
    records = [m["record"] for m in messages(second, "r2")
               if m["type"] == "suggestion" and m["record"]["suggestions"]]
    examples = [s["example"] for r in records for s in r["suggestions"]]
    assert examples and "¿Tienes autorización para trabajar en España?" \
        not in examples


def test_an_unconfirmed_proposal_does_not_count_in_the_next_round() -> None:
    store = MemoryCandidates()
    first = wired(conversation("r1", ROUND_ONE), AskingReasoner(), store)
    first.start_live("live_assist.interview", "stereo", session_id="r1",
                     plan="plan-1", candidate="Lucía Marín")
    second = wired(conversation("r2", [
        ["S1", "¿Tienes autorización para trabajar en España?"],
        ["S2", "No, no tengo autorización de momento."],
    ]), AskingReasoner(), store)
    second.start_live("live_assist.interview", "stereo", session_id="r2",
                      plan="plan-1", candidate_id="lucia-marin")
    kept = store.kept[("plan-1", "lucia-marin")]
    assert kept.answers == {}
    assert kept.proposals["work_authorization"].fields == {
        "eligibility.work_authorization": False}
    assert kept.proposals["work_authorization"].session_id == "r2"
    records = [m["record"] for m in messages(second, "r2")
               if m["type"] == "suggestion" and m["record"]["suggestions"]]
    examples = [s["example"] for r in records for s in r["suggestions"]]
    assert "¿Tienes autorización para trabajar en España?" in examples


def test_a_failed_start_leaves_no_empty_candidate() -> None:
    store = MemoryCandidates()
    live = wired([], store=store)
    live.transcriber.fail = True  # type: ignore[union-attr]
    with pytest.raises(SessionError) as caught:
        live.start_live("live_assist.interview", "stereo", session_id="x1",
                        plan="plan-1", candidate="Ana")
    assert caught.value.code == "transcriber"
    assert store.kept == {}
    live.transcriber.fail = False  # type: ignore[union-attr]
    live.start_live("live_assist.interview", "stereo", session_id="x2",
                    plan="plan-1", candidate="Ana")
    with pytest.raises(SessionError) as taken:
        live.start_live("live_assist.interview", "stereo", session_id="x2",
                        plan="plan-1", candidate="Bea")
    assert taken.value.code == "wrong_state"
    assert ("plan-1", "bea") not in store.kept


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
            if r.get("template") == "interviewer@2"]


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
          for n in range(4, 8)],
        user_final("t8.s0", "Unos 22.000 euros al año.", 9),
        user_final("t9.s0", "Vale, gracias.", 10),
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
    # Proposals for the interviewer to confirm; the motivation is never
    # captured from the candidate's words.
    assert set(kept.proposals) == {"work_authorization", "start_date_notice",
                                   "salary_expectation"}
    assert kept.answers == {}
    assert any(t.startswith("¿Qué expectativa salarial") for t in topics)
    assert kept.proposals["salary_expectation"].fields[
        "salary.expectation_min"] == 22000
    assert kept.rounds[0].kind == "bot"
    # The salary is asked with the interview still open, then it closes
    # normally (review #235: it used to stay open).
    found = messages(sessions, "vb-1")
    assert [m.get("state") for m in found if m["type"] == "interview"
            ][0] == "closed"
    assert asked(sessions)[-1]["action"] == "close"
    result = [m for m in found if m["type"] == "interview_result"][0]
    assert result["sheet"]["closed_by"] == "covered"


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


def test_a_later_round_starts_from_what_earlier_rounds_left() -> None:
    store = MemoryCandidates()
    first = wired(conversation("r1", ROUND_ONE), AskingReasoner(), store)
    first.start_live("live_assist.interview", "stereo", session_id="r1",
                     plan="plan-1", candidate="Lucía Marín")
    assert first.interview is not None
    plan = first.interview.store.load("plan-1")
    slots = plan.ordered()
    kept = store.kept[("plan-1", "lucia-marin")]
    talk = kept.rounds[0].talk
    assert set(talk) == {"interviewer", "candidate"}
    assert abs(sum(talk.values()) - 1.0) < 0.01
    topics = [m for m in messages(first, "r1") if m["type"] == "topics"]
    assert [t["earlier"] for t in topics[0]["topics"]] == [
        "open"] * len(slots)
    assert first.get("r1")["goal"] is None
    # Round two: the first slot was asked and answered at length, the
    # second asked and left short.
    one, two = slots[0], slots[1]
    second = wired(conversation("r2", [
        ["S1", f"Hablemos de {one.label}. {one.questions[0]}"],
        ["S2", "Estuve dos años en caja en un supermercado del barrio "
               "atendiendo a clientes cada día y resolviendo dudas."],
        ["S1", f"Ahora {two.label}. {two.questions[0]}"],
        ["S2", "Pues no mucho."],
    ]), AskingReasoner(), store)
    second.start_live("live_assist.interview", "stereo", session_id="r2",
                      plan="plan-1", candidate_id="lucia-marin")
    kept = store.kept[("plan-1", "lucia-marin")]
    assert kept.rounds[-1].covered == [one.id]
    assert kept.rounds[-1].revisit == [two.id]
    assert kept.rounds[-1].weak == []
    done = [m for m in messages(second, "r2") if m["type"] == "topics"][-1]
    assert [t["id"] for t in done["topics"] if t["done"]] == [one.id]
    # Round three: the weak slot leads, the covered one goes last, and
    # the suggestions are told so.
    third = wired(conversation("r3", [
        ["S1", "Cuéntame algo más."],
        ["S2", "Trabajé en equipo en muchos proyectos de tienda."],
    ]), AskingReasoner(), store)
    third.start_live("live_assist.interview", "stereo", session_id="r3",
                     plan="plan-1", candidate_id="lucia-marin")
    shown = [m for m in messages(third, "r3") if m["type"] == "topics"][0]
    order = [(t["id"], t["earlier"]) for t in shown["topics"]]
    assert order[0] == (two.id, "revisit")
    assert order[-1] == (one.id, "covered")
    goal = third.get("r3")["goal"]
    assert goal.startswith("Ronda 3") and two.label in goal
    assert "volver a tratar" in goal and "nivel" not in goal
    reasoner: AskingReasoner = third.reasoner  # type: ignore[assignment]
    sent = [r["context"]["session"] for r in reasoner.requests
            if r.get("role") == "suggester"]
    assert sent and sent[0]["goal"] == goal


def test_the_ai_interviewer_asks_the_weak_slots_first() -> None:
    store = MemoryCandidates()
    sessions = bot([user_final("t0.s0", "Vale.", 1)], store)
    assert sessions.interview is not None
    plan = sessions.interview.store.load("plan-1")
    minimos = [s for s in plan.ordered() if s.level == "minimo"]
    assert len(minimos) >= 2
    book = sessions.interview.candidates
    assert book is not None
    made = book.create("plan-1", "Ana")
    for question, text in (("work_authorization", "Sí"),
                           ("start_date_notice", "De inmediato"),
                           ("motivation", "Le gusta el sector"),
                           ("salary_expectation", "22.000")):
        book.correct("plan-1", made["id"], question, text)
    from orchestrator.domain.candidate import Round
    book.keep_round("plan-1", made["id"], Round(
        session_id="old", at="x", covered=[minimos[0].id],
        weak=[minimos[-1].id]))
    sessions.start_live("voicebot.interview", "mic", plan="plan-1",
                        candidate_id=made["id"])
    first = asked(sessions)[0]
    assert first["action"] == "open"
    assert first["topic"] == minimos[-1].label
    kept = store.kept[("plan-1", made["id"])]
    assert kept.rounds[-1].kind == "bot"
    assert kept.rounds[-1].talk.get("interviewer", 0) > 0


def test_a_reask_after_a_core_question_rewords_that_question() -> None:
    import json

    from orchestrator.domain.candidate import core_questions
    from tests.unit.orchestrator.fixtures import REPO
    from tests.unit.orchestrator.test_interview_usecases import director

    core = core_questions(json.loads(
        (REPO / "knowledge/interview/core-questions.json").read_text()))
    told: List[Optional[str]] = []
    boss = director()
    boss.core_open = lambda: list(core)
    boss.on_core = told.append
    first = boss.plan_turn("", None, CancelToken())
    assert first.payload["topic"].startswith("¿Tienes autorización")
    again = boss.plan_turn("", None, CancelToken(), kind="reask")
    assert again.payload["action"] == "reask"
    assert again.payload["questions"] == [first.payload["topic"]]
    hint = boss.plan_turn("Pues mira", None, CancelToken(), kind="hint")
    assert hint.payload["answer_so_far"] == "Pues mira"
    assert told == ["work_authorization"]  # re-asks keep it waiting
    assert not boss.closed


def test_a_stop_while_the_salary_waits_keeps_the_plans_close() -> None:
    from orchestrator.application.interview import (
        HeuristicFiller,
        InterviewDirector,
    )
    from orchestrator.domain.interview import InterviewPlan, Slot
    from tests.unit.orchestrator.test_interview_usecases import (
        NOW,
        Clock,
        Output,
    )

    core = [q for q in Core().questions() if q.id == "salary_expectation"]
    plan = InterviewPlan(id="p", title="T", minutes=20, slots=[
        Slot(id="s1", label="Caja", level="minimo", questions=["¿Caja?"])])
    boss = InterviewDirector(
        plan, "vb-1", InterviewSection(max_probes=1), HeuristicFiller(),
        FakeReasoner(), Output(), Clock(), Store(), Handoff(), False, "Ana",
        now=lambda: NOW, core_open=lambda: list(core),
    )
    boss.plan_turn("", None, CancelToken())  # opens with the slot
    salary = boss.plan_turn("No lo conozco.", "t0.s0", CancelToken())
    assert salary.payload["topic"].startswith("¿Qué expectativa salarial")
    assert boss.deferred_close == "covered" and not boss.closed
    result = boss.finish()  # stopped before the answer
    assert result is not None
    assert result["sheet"]["closed_by"] == "covered"
    assert result["sheet"]["state"] == "closed"


def test_a_live_round_follows_the_concepts_of_the_topic_asked() -> None:
    """#258: the topic's concepts are ticked and the suggestions told."""
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    assert [p.text for p in one.points] == [
        "Dónde y en qué puesto", "Cuánto tiempo", "Tareas que hacía"]
    live = wired(conversation("r1", [
        ["S1", f"Hablemos de {one.label}. {one.questions[0]}"],
        ["S2", "Llevo dos años trabajando de cara al público."],
        ["S1", "¿Y dónde fue, en qué puesto?"],
        ["S2", "Trabajé de dependienta en una tienda de deportes."],
    ]), AskingReasoner(), store)
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    topics = [m["topics"] for m in messages(live, "r1")
              if m["type"] == "topics"]
    current = [next((t for t in ts if t.get("current")), None)
               for ts in topics]
    last = current[-1]
    assert current[0] is None and last is not None and last["id"] == one.id
    first = next(t for ts in topics for t in ts
                 if t["id"] == one.id and any(
                     p["status"] == "treated" for p in t["points"]))
    assert [p["status"] for p in first["points"]] == [
        "pending", "treated", "treated"]
    assert all(p["status"] == "treated" for p in last["points"])
    assert last["done"] is True
    reasoner: AskingReasoner = live.reasoner  # type: ignore[assignment]
    goals = [r["context"]["session"].get("goal") for r in reasoner.requests
             if r.get("role") == "suggester"]
    assert goals[0] == (f"Tema en curso: «{one.label}»; conceptos aún sin "
                        "tratar: Dónde y en qué puesto.")
    assert goals[-1] is None  # nothing left of it: no goal (first round)
    assert live.get("r1")["goal"] is None
