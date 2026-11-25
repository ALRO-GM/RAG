"""The concepts of a plan's slot, followed answer by answer (#258)."""

from datetime import datetime

from orchestrator.domain.interview import (
    PlanSource,
    Point,
    SheetPoint,
    fallback_plan,
)
from orchestrator.domain.interview_agenda import Agenda, heuristic_fill
from orchestrator.domain.interview_points import (
    PointHit,
    join_words,
    names_point,
    parts,
    pending,
    point_hits,
    points_status,
    record_hit,
    says_unknown,
)

RUNNING = [
    Point(id="p1", text="Tipos de pisada"),
    Point(id="p2", text="Amortiguación y drop"),
    Point(id="p3", text="Cómo elegir la talla"),
]
STAR = [
    Point(id="p1", text="Situación concreta"),
    Point(id="p2", text="Lo que hizo él o ella"),
    Point(id="p3", text="Resultado"),
]


def states(*statuses: str) -> list[SheetPoint]:
    return [SheetPoint(id=f"p{i}", text=f"c{i}", status=s)  # type: ignore
            for i, s in enumerate(statuses, start=1)]


def test_a_point_is_named_by_its_words_or_a_cue_of_its_kind() -> None:
    assert names_point("Tipos de pisada", "miro si la pisada es pronadora")
    assert names_point("Amortiguación y drop", "la amortiguación y el drop")
    assert not names_point("Amortiguación y drop", "el drop alto")  # both
    assert names_point("Amortiguación y drop", "el drop", loose=True)
    assert not names_point("Tipos de pisada", "hay tipos de clientes")
    assert names_point("Cuánto tiempo", "estuve tres años allí")
    assert names_point("Fecha de incorporación", "podría empezar ya")
    assert names_point("Horario o turnos", "puedo de mañanas y sábados")
    assert names_point("Dónde trabajó y en qué puesto",
                       "trabajé de cajera en un súper")
    assert names_point("Tareas que hacía", "atendiendo en sala")
    assert names_point("Situación concreta", "una vez un cliente vino")
    assert names_point("Lo que hizo él o ella", "le expliqué el cambio")
    assert names_point("Resultado", "al final se resolvió")
    assert not names_point("Resultado", "no sé")
    assert not names_point("de la", "de la tienda")  # no content word


def test_hits_are_read_part_by_part_with_does_not_know() -> None:
    text = ("Conozco los tipos de pisada y cómo elegir la talla, pero el "
            "drop no lo conozco.")
    assert parts(text) == ["Conozco los tipos de pisada y cómo elegir la "
                           "talla", "el drop no lo conozco"]
    hits = point_hits(text, RUNNING)
    assert [(h.point, h.status) for h in hits] == [
        ("p1", "treated"), ("p2", "unknown"), ("p3", "treated")]
    assert hits[1].quote == "el drop no lo conozco"
    told = point_hits(
        "Una vez un cliente no entendía la factura. Se lo expliqué con un "
        "ejemplo y al final quedó contento.", STAR)
    assert [h.status for h in told] == ["treated"] * 3
    assert point_hits("Pues no sé", RUNNING) == []


def test_record_hit_moves_pending_and_unknown_never_back() -> None:
    found = states("pending", "unknown", "treated")
    assert record_hit(found, PointHit("p1", "unknown", "no lo sé"), "u1")
    assert [p.status for p in found][0] == "unknown"
    assert found[0].utterance == "u1"
    assert record_hit(found, PointHit("p1", "treated", "ya sí"), "u2")
    assert [p.status for p in found][0] == "treated"
    assert found[0].quote == "ya sí"
    assert not record_hit(found, PointHit("p3", "unknown", "x"), "u3")
    assert not record_hit(found, PointHit("p2", "unknown", "x"), "u3")
    assert not record_hit(found, PointHit("p9", "treated", "x"), "u3")
    assert found[2].status == "treated"


def test_the_completion_rule_of_a_slot_with_points() -> None:
    assert points_status([]) is None
    assert points_status(states("treated", "pending")) is None
    assert points_status(states("treated", "treated")) == "filled"
    assert points_status(states("unknown", "unknown")) == "unknown"
    assert points_status(states("treated", "unknown")) == "partial"
    assert [p.id for p in pending(states("treated", "pending"))] == ["p2"]


def test_lists_are_joined_as_spoken_and_unknown_is_shared() -> None:
    assert join_words([]) == ""
    assert join_words(["A"]) == "A"
    assert join_words(["A", "B"]) == "A y B"
    assert join_words(["A", "B", "C"]) == "A, B y C"
    assert join_words(["A", "B"], "en") == "A and B"
    assert says_unknown("No lo conozco.") and says_unknown("no sé")
    assert not says_unknown("Sé hacerlo bien")


NON_ANSWERS = [
    ("Lo que hizo él o ella", "Bueno, así es la vida"),
    ("Lo que hizo él o ella", "Aquí estoy"),
    ("Lo que hizo él o ella", "Un café, por favor"),
    ("Situación concreta", "Cuando quieras"),
    ("Cuánto tiempo", "Siempre llego a tiempo"),
    ("Tareas que hacía", "Voy hacia allí"),
    ("Tareas que hacía", "Estoy pensando la respuesta"),
    ("Horario o turnos", "Esta tarde no puedo"),
    ("Resultado", "al final se hace"),
    ("Dónde trabajó y en qué puesto", "Estuve pensando en eso"),
    ("Dónde trabajó y en qué puesto", "No tengo trabajo de eso"),
    ("Dónde trabajó y en qué puesto", "Trabajo de lo que salga"),
    ("Fecha de incorporación", "Me gustaría empezar a aprender"),
    ("Lo que hizo él o ella", "un porqué no te sé dar"),
    ("Cuánto tiempo", "Uf, un año de esos complicados"),
    ("Cuánto tiempo", "Llevo dos semanas buscando"),
]


def test_filler_words_never_tick_a_concept() -> None:
    """Review of #258: missing a concept beats a false tick."""
    for point, said in NON_ANSWERS:
        assert not names_point(point, said), (point, said)
    assert names_point("Lo que hizo él o ella", "lo dejé preparado")
    assert point_hits("Pues no sé qué decirte", STAR) == []


def test_a_vague_star_answer_stays_partial_and_is_probed() -> None:
    plan = fallback_plan("p", "Puesto", PlanSource(), 20, "es",
                         datetime(2026, 11, 25))
    comm = next(s for s in plan.slots if s.competency == "COMM")
    text = "Pues cuando hay mucho jaleo, así es la vida, al final se hace."
    fills = heuristic_fill(text, comm, plan.slots)
    assert [h for f in fills for h in f.points] == []
    agenda = Agenda(plan.model_copy(update={"slots": [comm]}), 0.0)
    agenda.step(0.0)
    agenda.answer("u1", text, fills)
    assert agenda.state[comm.id].status == "partial"
    probe = agenda.step(1.0)
    assert probe.action == "probe" and len(probe.points) == 3


def test_the_documented_limits_of_the_rules() -> None:
    """What docs/live/VOICEBOT_INTERVIEWER.md §4.1 says they miss or tick."""
    assert not names_point("Cómo elegir la talla", "elijo el número")
    assert not names_point("Amortiguación y drop", "la amortiguación")
    # A word of the concept used in another sense still ticks it.
    assert names_point("Tipos de pisada", "dejé una pisada en el suelo mojado")


def test_the_cues_still_read_real_answers() -> None:
    """The tightened cues of the #258 re-review keep the plain answers."""
    assert names_point("Cuánto tiempo", "estuve tres años allí")
    assert names_point("Cuánto tiempo", "llevo dos años trabajando en caja")
    assert names_point("Cuánto tiempo", "tengo cinco años de experiencia")
    assert names_point("Fecha de incorporación", "podría empezar el lunes")
    assert names_point("Fecha de incorporación", "me incorporo de inmediato")
    assert names_point("Dónde trabajó y en qué puesto",
                       "trabajaba de camarera en un bar")
