"""The next round from earlier rounds, and the live round's topics (#235)."""

from typing import List

from orchestrator.application.next_round import (
    TopicTracker,
    keywords,
    slot_asked,
)
from orchestrator.application.ports import Document
from orchestrator.domain.candidate import Round
from orchestrator.domain.interview import InterviewPlan, Slot
from orchestrator.domain.next_round import (
    focus_goal,
    reordered,
    round_from_sheet,
    topic_states,
)
from tests.unit.orchestrator.test_candidate import final


def plan() -> InterviewPlan:
    return InterviewPlan(id="dev", title="Backend", slots=[
        Slot(id="s1", label="Experiencia con Python", level="minimo",
             questions=["¿Cuántos años llevas con Python?"]),
        Slot(id="s2", label="Pruebas automáticas", level="minimo",
             questions=["¿Cómo pruebas tu código?"]),
        Slot(id="s3", label="Bases de datos", level="deseable",
             questions=["¿Qué bases de datos has usado?"]),
        Slot(id="s4", label="Inglés", level="opcional",
             questions=["¿Qué nivel de inglés tienes?"]),
    ])


def test_a_sheet_says_what_was_covered_and_what_is_weak() -> None:
    sheet = {"slots": [
        {"id": "s1", "level": "minimo", "status": "filled", "score": 80},
        {"id": "s2", "level": "minimo", "status": "partial", "score": 40},
        {"id": "s3", "level": "deseable", "status": "filled",
         "score": None},
        {"id": "s4", "level": "opcional", "status": "unknown", "score": 0},
        {"id": "s5", "level": "minimo", "status": "empty", "score": None},
        {"level": "minimo"}, "broken",
    ]}
    assert round_from_sheet(sheet) == (["s1", "s3"], ["s2"])
    assert round_from_sheet({}) == ([], [])


def test_weak_first_then_open_then_covered_newest_round_wins() -> None:
    rounds = [
        Round(session_id="r1", at="x", covered=["s2"], weak=["s1"]),
        Round(session_id="r2", at="y", covered=["s1"], weak=["s2"]),
    ]
    states = topic_states(plan(), rounds)
    assert [(t.slot.id, t.earlier) for t in states] == [
        ("s2", "weak"), ("s3", "open"), ("s4", "open"), ("s1", "covered"),
    ]
    assert [s.id for s in reordered(plan(), states).slots] == [
        "s2", "s3", "s4", "s1"]
    # The agenda still asks minimos first: s2, then s1, then the rest.
    assert [s.id for s in reordered(plan(), states).ordered()] == [
        "s2", "s1", "s3", "s4"]
    topic = states[0].to_topic(True)
    assert topic == {"id": "s2", "name": "Pruebas automáticas",
                     "done": True, "earlier": "weak", "level": "minimo"}


def test_the_goal_of_a_later_round_names_what_is_left() -> None:
    states = topic_states(plan(), [Round(session_id="r1", at="x",
                                         covered=["s1"], weak=["s2"])])
    goal = focus_goal("Backend", states, 1)
    assert goal is not None and goal.startswith("Ronda 2 del puesto")
    assert "por debajo del nivel en la entrevista de IA: Pruebas " \
        "automáticas" in goal
    assert "sin tratar aún: Bases de datos; Inglés" in goal
    english = focus_goal("Backend", states, 1, "en")
    assert english is not None and english.startswith("Round 2")
    assert focus_goal("Backend", states, 0) is None
    done = topic_states(plan(), [Round(session_id="r1", at="x",
                                       covered=["s1", "s2", "s3", "s4"])])
    assert focus_goal("Backend", done, 1) is None
    long = InterviewPlan(id="x", title="T", slots=[
        Slot(id=f"s{n}", label="tema " * 20, questions=["¿Algo?"])
        for n in range(1, 10)])
    cut = focus_goal("T", topic_states(long, []), 3)
    assert cut is not None and len(cut) <= 480 and cut.endswith("…")


def test_the_slot_an_interviewer_asks_about() -> None:
    slots = plan().slots
    assert keywords("¿Cuántos años llevas con Python?") == {
        "cuantos", "llevas", "python"}
    found = slot_asked("Cuéntame cómo pruebas tu código: ¿pruebas "
                       "automáticas?", slots)
    assert found is not None and found.id == "s2"
    english = slot_asked("¿Y el inglés, qué tal?", slots)
    assert english is not None and english.id == "s4"
    assert slot_asked("¿Qué tal el viaje?", slots) is None


def test_the_live_round_marks_topics_and_reports_the_round() -> None:
    published: List[Document] = []
    states = topic_states(plan(), [Round(session_id="r1", at="x",
                                         weak=["s2"])])
    roles = {"S1": "interviewer", "S2": "candidate"}
    track = TopicTracker(plan(), states, roles, published.append)
    track.start()
    assert [t["id"] for t in published[0]["topics"]][0] == "s2"
    track.on_event(final("t0.s0", "S2", "Hice muchas pruebas."))  # no question
    track.on_event(final("t1.s0", "S1", "¿Cómo pruebas tu código, qué "
                                        "pruebas automáticas usas?"))
    track.on_event(final("t2.s0", "S2", "Escribo pruebas unitarias con "
                                        "pytest en cada cambio y las "
                                        "lanzamos en la integración "
                                        "continua del equipo."))
    track.on_event(final("t3.s0", "S1", "¿Qué bases de datos has usado?"))
    track.on_event(final("t4.s0", "S2", "Postgres, un poco."))
    assert track.result() == (["s2"], ["s3"])
    last = published[-1]["topics"]
    assert [t["done"] for t in last if t["id"] == "s2"] == [True]
    assert len(published) == 2


def test_a_live_round_leaves_topics_to_revisit_not_weak() -> None:
    rounds = [Round(session_id="r1", at="x", revisit=["s3"], weak=["s2"],
                    covered=["s1"])]
    states = topic_states(plan(), rounds)
    assert [(t.slot.id, t.earlier) for t in states] == [
        ("s2", "weak"), ("s3", "revisit"), ("s4", "open"), ("s1", "covered"),
    ]
    goal = focus_goal("Backend", states, 1)
    assert goal is not None and "volver a tratar" in goal
    english = focus_goal("Backend", states, 1, "en")
    assert english is not None and "to go back to" in english
