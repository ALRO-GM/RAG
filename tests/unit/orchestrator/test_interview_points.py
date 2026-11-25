"""The concepts of a plan's slot, followed answer by answer (#258)."""

from orchestrator.domain.interview import Point, SheetPoint
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
    assert names_point("Amortiguación y drop", "el drop alto")
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
