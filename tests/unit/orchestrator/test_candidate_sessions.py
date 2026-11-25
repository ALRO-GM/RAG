"""The candidate across rounds in live sessions and the AI interviewer (#235).

Two rounds of one synthetic candidate: the first captures answers to
the core questions, the second shows them as answered, does not capture
them again and drops suggestions that would ask them. The AI interviewer
asks the open core questions (not the one with blanks) and never the
answered ones.
"""

import json
from typing import Any, Dict, List, Optional

import pytest

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.jsonl_sink import MemoryHandoff
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.candidates import CandidateBook
from orchestrator.application.next_round import TopicTracker
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.ports import (
    CancelToken,
    Document,
    ReasonerUnavailable,
)
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
          store: Optional[MemoryCandidates] = None,
          runner: Any = None, handoff: Any = None,
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
                         lambda p: p in plans.plans, 30, plans=plans)
    return LiveSessions(
        {"live_assist.interview": product("live_assist.interview"),
         "live_assist.support": product()},
        knowledge, reasoner, Transcripts({"t": transcript()}), Clock(),
        runner or InlineRunner(), InlineExecutor, FakeTranscriber(),
        FakeEvents(documents),
        interview=InterviewWiring(planner, plans, reasoner, Handoff(),
                                  book),
        handoff=handoff,
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
        ["S2", "Trabajé de dependienta en una tienda de deportes, "
               "atendiendo clientes."],
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
        "pending", "treated", "pending"]
    assert all(p["status"] == "treated" for p in last["points"])
    assert last["done"] is True
    reasoner: AskingReasoner = live.reasoner  # type: ignore[assignment]
    goals = [r["context"]["session"].get("goal") for r in reasoner.requests
             if r.get("role") == "suggester"]
    assert goals[0] == (f"Tema en curso: «{one.label}»; conceptos aún sin "
                        "tratar: Dónde y en qué puesto y Tareas que hacía.")
    assert goals[-1] is None  # nothing left of it: no goal (first round)
    assert live.get("r1")["goal"] is None


def test_a_failing_topic_tracker_never_stops_the_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of #258: the topics run first, so they must not break."""
    def broken(self: TopicTracker, event: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(TopicTracker, "on_event", broken)
    live = wired(conversation("r1", [
        ["S1", "¿Tienes autorización para trabajar en España?"],
        ["S2", "Sí, la tengo desde hace años."],
    ]), AskingReasoner(), MemoryCandidates())
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    reasoner: AskingReasoner = live.reasoner  # type: ignore[assignment]
    assert any(r.get("role") == "suggester" for r in reasoner.requests)
    assert live.get("r1")["state"] == "ended"


class NoJudge(AskingReasoner):
    """Answers everything but the report of a live round (while down)."""

    down = True

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        if request.get("template") == "interview.report@2" and self.down:
            raise ReasonerUnavailable("down")
        return super().reason(request, cancel)


def round_lines(one: Any) -> List[List[str]]:
    return [
        ["S1", f"Hablemos de {one.label}. {one.questions[0]}"],
        ["S2", "Trabajé de dependienta en una tienda de deportes, "
               "atendiendo clientes durante dos años."],
    ]


def test_a_live_round_ends_with_a_draft_judgement_of_its_answers() -> None:
    """#259: the live page's end result judges the answers, once."""
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    live = wired(conversation("r1", round_lines(one)), AskingReasoner(),
                 store)
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    found = [m for m in messages(live, "r1") if m["type"] == "accuracy"]
    assert [m["state"] for m in found] == ["reporting", "ready"]
    ready = found[-1]
    assert [s["id"] for s in ready["slots"]] == [one.id]  # only answered
    assert ready["slots"][0]["judgement"]["label"] == "correct_basic"
    assert "score" not in ready["slots"][0]  # the live page never scores
    reasoner: AskingReasoner = live.reasoner  # type: ignore[assignment]
    judged = [r for r in reasoner.requests
              if r.get("template") == "interview.report@2"]
    assert len(judged) == 1
    handoff = live.get("r1")["handoff"]
    assert handoff["accuracy"]["state"] == "ready"
    assert live.get("r1")["accuracy"]["slots"][0]["id"] == one.id


def test_a_failed_or_empty_live_judgement_never_breaks_the_end() -> None:
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    down = wired(conversation("r1", round_lines(one)), NoJudge(), store)
    down.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    assert [m["state"] for m in messages(down, "r1")
            if m["type"] == "accuracy"] == ["reporting", "failed"]
    assert down.get("r1")["state"] == "ended"
    quiet = wired(conversation("r2", [["S1", "Buenos días."]]),
                  AskingReasoner(), store)
    quiet.start_live("live_assist.interview", "stereo", session_id="r2",
                     plan="plan-1", candidate="Lucía Marín")
    assert not [m for m in messages(quiet, "r2") if m["type"] == "accuracy"]
    assert quiet.get("r2")["accuracy"] is None


class FirstInline:
    """Runs the session's own job inline and holds the later ones."""

    def __init__(self) -> None:
        self.ran = False
        self.held: List[Any] = []

    def start(self, job: Any) -> None:
        if self.ran:
            self.held.append(job)
            return
        self.ran = True
        job()


def test_a_round_is_judged_after_it_ends_and_the_file_has_no_words() -> None:
    """#259 review: judging never keeps the session running."""
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    runner, kept = FirstInline(), MemoryHandoff()
    live = wired(conversation("r1", round_lines(one)), AskingReasoner(),
                 store, runner, lambda session, product: kept)
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    assert live.get("r1")["state"] == "ended"  # not running while judged
    assert live.get("r1")["accuracy"] == {"state": "reporting"}
    assert len(runner.held) == 1
    early, finished = live.feed("r1").read(0, 0.01)
    assert not finished  # the feed waits for the judgement
    assert not [m for _, m in early if m["type"] == "accuracy"]
    assert "accuracy" not in kept.records[-1]
    runner.held.pop()()
    found = [m for m in messages(live, "r1") if m["type"] == "accuracy"]
    assert [m["state"] for m in found] == ["reporting", "ready"]
    assert live.get("r1")["handoff"]["accuracy"]["state"] == "ready"
    stored = kept.records[-1]["accuracy"]
    judged = stored["slots"][0]["judgement"]
    assert judged["label"] == "correct_basic" and judged["utterance"]
    assert "quote" not in judged and "reason" not in judged
    assert "Trabajé" not in json.dumps(kept.records[-1], ensure_ascii=False)


def test_a_closed_position_starts_no_interview() -> None:
    """#262: closing refuses new rounds, AI or live; reopening allows."""
    store = MemoryCandidates()
    live = wired(conversation("r1", []), None, store)
    assert live.interview is not None
    live.interview.positions.close("plan-1")
    with pytest.raises(SessionError) as refused:
        live.start_live("live_assist.interview", "stereo", session_id="r1",
                        plan="plan-1", candidate="Lucía Marín")
    assert refused.value.code == "plan_closed"
    live.interview.positions.reopen("plan-1")
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    assert live.get("r1")["state"] == "ended"


def test_a_failed_round_is_judged_again_and_kept() -> None:
    """#262: «Reintentar» judges the round again and stores the report."""
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    reasoner = NoJudge()
    live = wired(conversation("r1", round_lines(one)), reasoner, store)
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    assert live.get("r1")["accuracy"]["state"] == "failed"
    assert live.interview is not None
    assert live.interview.store.results("plan-1") == []
    with pytest.raises(SessionError):
        live.rejudge("nope")
    reasoner.down = False
    view = live.rejudge("r1")
    assert view["accuracy"]["state"] == "ready"
    assert live.get("r1")["accuracy"]["saved"] is True
    kept = live.interview.store.results("plan-1")
    assert [k["kind"] for k in kept] == ["live"]
    assert kept[0]["candidate_id"] == live.get("r1")["candidate_id"]
    with pytest.raises(SessionError) as again:
        live.rejudge("r1")
    assert again.value.code == "wrong_state"


def test_only_one_judgement_of_a_round_runs_at_a_time() -> None:
    """Review of #262: «Reintentar» never starts a second judgement."""
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    runner = FirstInline()
    live = wired(conversation("r1", round_lines(one)), NoJudge(), store,
                 runner)
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    assert live.get("r1")["accuracy"] == {"state": "reporting"}
    with pytest.raises(SessionError):  # the first one is still running
        live.rejudge("r1")
    runner.held.pop()()
    assert live.get("r1")["accuracy"]["state"] == "failed"
    live.rejudge("r1")
    with pytest.raises(SessionError):  # a second quick POST
        live.rejudge("r1")
    assert len(runner.held) == 1


def test_retry_after_a_save_that_failed_and_never_for_a_deleted_file() -> None:
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    live = wired(conversation("r1", round_lines(one)), AskingReasoner(),
                 store)
    assert live.interview is not None
    plans = live.interview.store
    real = plans.save_result

    def broken(plan_id: str, sheet: Document) -> None:
        raise OSError("disk full")

    plans.save_result = broken  # type: ignore[method-assign]
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    found = live.get("r1")["accuracy"]
    assert found["state"] == "ready" and found["saved"] is False
    plans.save_result = real  # type: ignore[method-assign]
    live.rejudge("r1")  # ready but not saved: accepted
    assert live.get("r1")["accuracy"]["saved"] is True
    with pytest.raises(SessionError):  # saved: nothing to retry
        live.rejudge("r1")
    # A file deleted before «Reintentar» never gets its report back.
    book = live.interview.candidates
    assert book is not None
    candidate = live.get("r1")["candidate_id"]
    book.delete("plan-1", candidate)
    assert plans.results("plan-1") == []
    live.get("r1")["accuracy"]  # view still has the old result in memory
    session = live._find("r1")
    session.accuracy = {"state": "failed"}
    live.rejudge("r1")
    assert live.get("r1")["accuracy"]["saved"] is False
    assert plans.results("plan-1") == []


def test_a_live_round_and_its_replayed_transcript_attribute_alike() -> None:
    """Review of #262: what `interview rejudge` rebuilds is what the live
    round saw, for the same finals (no revisions)."""
    from orchestrator.application.live_judgement import replayed_tracker
    from orchestrator.domain.events import Transcript

    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    plan = probe.interview.store.load("plan-1")
    one, two = plan.ordered()[:2]
    lines = [
        ["S1", f"Hablemos de {one.label}. {one.questions[0]}"],
        ["S2", "Trabajé de dependienta en una tienda de deportes, "
               "atendiendo clientes durante dos años."],
        ["S2", "Sobre todo en caja."],
        ["S1", f"Ahora {two.label}. {two.questions[0]}"],
        ["S2", "Pues no mucho."],
    ]
    live = wired(conversation("r1", lines), AskingReasoner(), store)
    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    seen = live._find("r1").topics
    assert seen is not None
    transcript = Transcript.model_validate({
        "session": {"id": "r1", "language": "es",
                    "started_at": "2026-11-24T10:00:00Z"},
        "speakers": [{"id": "S1", "role": "interviewer"},
                     {"id": "S2", "role": "candidate"}],
        "utterances": [
            {"id": f"t{n}.s0", "turn": n, "index": 0, "speaker": who,
             "start": float(n), "end": n + 0.9, "text": text}
            for n, (who, text) in enumerate(lines)],
    })
    again = replayed_tracker(transcript, plan)

    def shape(tracker: Any) -> Any:
        slots, texts = tracker.sheet_slots()
        return ([(s.id, s.status, [a.utterance for a in s.answers],
                  [p.status for p in s.points]) for s in slots], texts)

    assert shape(again) == shape(seen)


def test_a_judgement_that_raises_ends_as_failed_not_stuck() -> None:
    """Re-review of #262: never «Preparando…» forever."""
    store = MemoryCandidates()
    probe = wired(conversation("r0", []), None, store)
    assert probe.interview is not None
    one = probe.interview.store.load("plan-1").ordered()[0]
    live = wired(conversation("r1", round_lines(one)), AskingReasoner(),
                 store)

    def boom(topics: Any) -> Any:
        raise RuntimeError("odd")

    live.start_live("live_assist.interview", "stereo", session_id="r1",
                    plan="plan-1", candidate="Lucía Marín")
    session = live._find("r1")
    assert session.judge is not None
    session.judge.judge = boom  # type: ignore[method-assign]
    session.accuracy = {"state": "failed"}
    live.rejudge("r1")
    assert live.get("r1")["accuracy"] == {"state": "failed", "saved": False}
    # The end of a round: the same, and the page is told.
    runner = FirstInline()
    held = wired(conversation("r2", round_lines(one)), AskingReasoner(),
                 store, runner)
    held.start_live("live_assist.interview", "stereo", session_id="r2",
                    plan="plan-1", candidate="Lucía Marín")
    target = held._find("r2")
    assert target.judge is not None
    target.judge.judge = boom  # type: ignore[method-assign]
    runner.held.pop()()
    assert held.get("r2")["accuracy"] == {"state": "failed", "saved": False}
    found, finished = held.feed("r2").read(0, 0.01)
    assert finished
    assert {"type": "accuracy", "state": "failed", "saved": False} in [
        m for _, m in found]
    held.rejudge("r2")  # a failed round can be retried
