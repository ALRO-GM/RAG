"""The interviewer's agenda and the heuristic slot filler (#208)."""

from datetime import datetime
from typing import List

import pytest

from orchestrator.domain.interview import InterviewPlan, Point, Slot
from orchestrator.domain.interview_agenda import (
    MAX_TURNS,
    Agenda,
    Fill,
    heuristic_fill,
    is_question,
    says_unknown,
    wants_skip,
)

NOW = datetime(2026, 11, 21, 10, 0, 0)


def slot(sid: str, level: str = "minimo", competency: str = "",
         label: str = "") -> Slot:
    return Slot(
        id=sid, label=label or f"tema {sid}", level=level,  # type: ignore
        kind="competency" if competency else "requirement",
        competency=competency or None,
        questions=[f"Pregunta sobre {label or sid}"],
        evidence="experiencia concreta",
    )


def plan(slots: List[Slot], minutes: int = 20) -> InterviewPlan:
    return InterviewPlan(id="plan-1", title="Puesto", minutes=minutes,
                         slots=slots)


def full(sid: str) -> List[Fill]:
    return [Fill(sid, "complete", "cita")]


def part(sid: str) -> List[Fill]:
    return [Fill(sid, "partial", "cita")]


def test_open_then_one_slot_per_answer_minimums_first() -> None:
    agenda = Agenda(plan([slot("s1", "deseable"), slot("s2"), slot("s3")]),
                    0.0)
    first = agenda.step(0.0)
    assert first.action == "open" and first.slot is not None
    assert first.slot.id == "s2"
    agenda.answer("u1", "respuesta larga y concreta", full("s2"))
    assert agenda.step(10.0).slot.id == "s3"  # type: ignore[union-attr]
    agenda.answer("u2", "otra respuesta", full("s3"))
    nxt = agenda.step(20.0)
    assert nxt.action == "ask" and nxt.slot.id == "s1"  # type: ignore
    assert agenda.turns_to_minimums == 2
    agenda.answer("u3", "respuesta", full("s1"))
    assert agenda.step(30.0).action == "close"
    assert agenda.closed_by == "covered"
    assert [s.status for s in agenda.sheet_slots()] == ["filled"] * 3
    assert agenda.step(40.0).action == "close"


def test_one_probe_then_the_clarify_offer_then_a_skip() -> None:
    agenda = Agenda(plan([slot("s1"), slot("s2")]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "pues sí", part("s1"))
    probe = agenda.step(1.0)
    assert probe.action == "probe" and probe.missing
    agenda.answer("u2", "bueno, más o menos", part("s1"))
    clarify = agenda.step(2.0)
    assert clarify.action == "clarify"
    assert clarify.slot is not None and clarify.slot.id == "s1"
    agenda.answer("u3", "Mejor pasamos a la siguiente pregunta", [])
    nxt = agenda.step(3.0)
    assert nxt.action == "ask" and nxt.slot.id == "s2"  # type: ignore
    state = agenda.state["s1"]
    assert state.status == "skipped" and state.probes == 1
    assert [a.utterance for a in state.answers] == ["u1", "u2"]


def test_a_refined_answer_after_the_offer_fills_the_slot() -> None:
    agenda = Agenda(plan([slot("s1")]), 0.0, max_probes=0)
    agenda.step(0.0)
    agenda.answer("u1", "no sabría", [])
    assert agenda.step(1.0).action == "clarify"
    agenda.answer("u2", "Sí: estuve dos años en caja", full("s1"))
    assert agenda.step(2.0).action == "close"
    assert agenda.state["s1"].status == "filled"


def test_not_knowing_the_topic_stops_asking_about_it() -> None:
    agenda = Agenda(plan([slot("s1"), slot("s2")]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "La verdad es que no lo conozco, nunca he trabajado"
                  " con eso", [])
    nxt = agenda.step(1.0)
    assert nxt.slot.id == "s2"  # type: ignore[union-attr]
    assert agenda.state["s1"].status == "unknown"
    assert agenda.state["s1"].probes == 0


def test_a_candidate_question_is_answered_once_per_slot() -> None:
    agenda = Agenda(plan([slot("s1")]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "¿El puesto es de mañana o de tarde?", [])
    assert agenda.step(1.0).action == "answer"
    agenda.answer("u2", "¿Y el sueldo?", [])
    assert agenda.step(2.0).action == "probe"


def test_time_up_closes_and_optional_slots_wait_for_time() -> None:
    agenda = Agenda(plan([slot("s1"), slot("s2", "opcional")], 10), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "respuesta", full("s1"))
    assert agenda.step(6 * 60.0).action == "close"  # 40 % left < 50 %
    late = Agenda(plan([slot("s1"), slot("s2")], 10), 0.0)
    late.step(0.0)
    assert late.step(11 * 60.0).action == "close"
    assert late.closed_by == "time"
    assert late.minutes_left(11 * 60.0) == 0.0


def test_the_interview_closes_after_the_turn_limit() -> None:
    agenda = Agenda(plan([slot("s1")]), 0.0, max_probes=3)
    agenda.turns = MAX_TURNS
    agenda.opened = True
    assert agenda.step(1.0).action == "close"


def test_fills_of_other_slots_are_recorded_too() -> None:
    agenda = Agenda(plan([slot("s1"), slot("s2")]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "respuesta", full("s1") + part("s2") + part("zz"))
    assert agenda.state["s2"].status == "partial"
    assert agenda.step(1.0).slot.id == "s2"  # type: ignore[union-attr]


@pytest.mark.parametrize("text,unknown,skip", [
    ("No lo conozco, la verdad", True, False),
    ("No sé.", True, False),
    ("No tengo experiencia con Kubernetes", True, False),
    ("Prefiero pasar a la siguiente", False, True),
    ("Sigamos, no tengo más que añadir", False, True),
    ("Estuve tres años en una tienda de ropa", False, False),
    ("No sé si me explico: estuve tres años en caja y reponiendo", False,
     False),
])
def test_what_the_candidate_says(text: str, unknown: bool,
                                 skip: bool) -> None:
    assert says_unknown(text) is unknown
    assert wants_skip(text) is skip


def test_question_detection() -> None:
    assert is_question("¿Hay que trabajar los domingos?")
    assert not is_question("Trabajo los domingos.")


def test_heuristic_fill_competency_and_requirement() -> None:
    comm = slot("s1", competency="COMM", label="Comunicación clara")
    caja = slot("s2", label="Experiencia en caja y atención al cliente")
    star = ("Una vez un cliente no entendía la garantía; le expliqué yo "
            "paso a paso con el ticket delante qué cubría y qué no, le "
            "dibujé un esquema y al final se fue contento y volvió a comprar")
    assert heuristic_fill(star, comm, [comm, caja])[0].status == "complete"
    vague = "Yo creo que me comunico bien con la gente en general"
    assert heuristic_fill(vague, comm, [comm])[0].status == "partial"
    years = "Estuve dos años en caja en un supermercado del barrio"
    found = heuristic_fill(years, caja, [comm, caja])
    assert found[0] == Fill("s2", "complete", years)
    assert heuristic_fill("Sí, claro", caja, [caja]) == []
    assert heuristic_fill("No lo conozco", caja, [caja]) == []
    other = ("Ademas tengo experiencia en caja y atencion al cliente, "
             "porque trabaje en una tienda durante varios veranos seguidos")
    assert Fill("s2", "partial", other) in heuristic_fill(
        other, comm, [comm, caja])


# ---- concepts per point (#258) ------------------------------------------

def running(level: str = "minimo", n: int = 3) -> Slot:
    texts = ["Tipos de pisada", "Amortiguación y drop",
             "Cómo elegir la talla", "Plantillas", "Calcetines técnicos"]
    return Slot(
        id="s1", label="Conocimiento de material de running",
        level=level,  # type: ignore[arg-type]
        questions=["¿Qué sabes de zapatillas de running?"],
        points=[Point(id=f"p{i}", text=t)
                for i, t in enumerate(texts[:n], start=1)],
    )


def statuses(agenda: Agenda, sid: str = "s1") -> List[str]:
    return [p.status for p in agenda.state[sid].points]


def test_a_slot_with_points_is_filled_by_its_points_not_by_length() -> None:
    target = running()
    agenda = Agenda(plan([target, slot("s2")]), 0.0)
    assert agenda.step(0.0).slot == target
    text = ("Trabajé tres años en una tienda de deportes y siempre miraba "
            "los tipos de pisada de cada cliente con mucho cuidado.")
    fills = heuristic_fill(text, target, [target])
    assert fills[0].status == "complete"  # long and concrete...
    agenda.answer("u1", text, fills)
    assert statuses(agenda) == ["treated", "pending", "pending"]
    assert agenda.state["s1"].status == "partial"  # ...but two are pending
    probe = agenda.step(1.0)
    assert probe.action == "probe"
    assert probe.points == ("Amortiguación y drop", "Cómo elegir la talla")
    assert probe.treated == ("Tipos de pisada",)
    assert "Amortiguación y drop y Cómo elegir la talla" in probe.missing
    more = ("El drop y la amortiguación según el uso, y para elegir la "
            "talla, medio número más.")
    agenda.answer("u2", more, heuristic_fill(more, target, [target]))
    assert statuses(agenda) == ["treated"] * 3
    assert [s.status for s in agenda.sheet_slots()][0] == "filled"
    assert agenda.state["s1"].points[1].utterance == "u2"
    nxt = agenda.step(2.0)
    assert nxt.action == "ask" and nxt.slot is not None
    assert nxt.slot.id == "s2"


def test_does_not_know_settles_the_points_asked_not_all() -> None:
    target = running(n=5)
    agenda = Agenda(plan([target, slot("s2")]), 0.0, max_probes=2)
    agenda.step(0.0)
    text = ("Sé de tipos de pisada y de la amortiguación y el drop de "
            "cada modelo.")
    agenda.answer("u1", text, heuristic_fill(text, target, [target]))
    probe = agenda.step(1.0)
    assert probe.points == ("Cómo elegir la talla", "Plantillas",
                            "Calcetines técnicos")[:3]
    assert len(probe.points) == 3  # never more than three at once
    agenda.answer("u2", "No lo sé.", heuristic_fill("No lo sé.", target,
                                                    [target]))
    assert statuses(agenda) == ["treated", "treated", "unknown", "unknown",
                                "unknown"]
    assert agenda.state["s1"].status == "partial"  # settled: nothing pending
    assert agenda.step(2.0).slot.id == "s2"  # type: ignore[union-attr]


def test_a_slot_whose_points_are_all_unknown_is_unknown() -> None:
    target = running()
    agenda = Agenda(plan([target, slot("s2")]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "No lo conozco, la verdad.",
                  heuristic_fill("No lo conozco, la verdad.", target,
                                 [target]))
    assert statuses(agenda) == ["unknown"] * 3
    assert agenda.state["s1"].status == "unknown"
    assert agenda.step(1.0).slot.id == "s2"  # type: ignore[union-attr]


def test_one_point_not_known_in_a_sentence_the_rest_pending() -> None:
    target = running()
    agenda = Agenda(plan([target]), 0.0)
    agenda.step(0.0)
    text = "El drop no lo conozco."
    fills = heuristic_fill(text, target, [target])
    assert [(h.point, h.status) for h in fills[0].points] == [
        ("p2", "unknown")]
    agenda.answer("u1", text, fills)
    assert statuses(agenda) == ["pending", "unknown", "pending"]
    probe = agenda.step(1.0)
    assert probe.action == "probe"
    assert probe.points == ("Tipos de pisada", "Cómo elegir la talla")
    assert probe.treated == ()


def test_the_clarification_names_the_points_left_then_moves_on() -> None:
    target = running()
    agenda = Agenda(plan([target, slot("s2")]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "Algo sé de zapatillas", [])
    assert agenda.step(1.0).action == "probe"
    agenda.answer("u2", "Pues lo normal", [])
    clarify = agenda.step(2.0)
    assert clarify.action == "clarify" and len(clarify.points) == 3
    agenda.answer("u3", "Sigamos, mejor", [])
    assert agenda.step(3.0).slot.id == "s2"  # type: ignore[union-attr]
    state = agenda.state["s1"]
    assert state.status == "skipped" and statuses(agenda) == ["pending"] * 3


def test_a_slot_without_points_keeps_the_generic_follow_up() -> None:
    agenda = Agenda(plan([slot("s1")]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", "pues sí", part("s1"))
    probe = agenda.step(1.0)
    assert probe.points == () and probe.missing.startswith("un dato")
    assert agenda.state["s1"].points == []


def test_a_short_answer_still_ticks_the_points_it_names() -> None:
    target = running()
    fills = heuristic_fill("Elegir la talla.", target, [target])
    assert [(f.slot, f.status) for f in fills] == [("s1", "partial")]
    assert [h.point for h in fills[0].points] == ["p3"]
    assert heuristic_fill("Vale.", target, [target]) == []
    english = Agenda(plan([target]).model_copy(update={"language": "en"}),
                     0.0)
    english.step(0.0)
    english.answer("u1", "Algo", [])
    assert english.step(1.0).missing.startswith("what they have not")


def test_points_of_a_slot_not_asked_are_never_ticked() -> None:
    other = running().model_copy(update={"id": "s2"})
    asked = slot("s1", label="Atención al cliente en tienda")
    text = ("Atendía a los clientes de la tienda de material de running: "
            "miraba los tipos de pisada, la amortiguación y el drop de cada "
            "zapatilla, y les ayudaba a elegir la talla.")
    fills = heuristic_fill(text, asked, [asked, other])
    assert {f.slot for f in fills} == {"s1", "s2"}
    assert all(f.points == () for f in fills)
    agenda = Agenda(plan([asked, other]), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", text, fills)
    assert statuses(agenda, "s2") == ["pending"] * 3
    assert agenda.state["s2"].status == "partial"
