"""Our Spanish dialogue-act set: the text format and its checks."""

import pytest

from src.domain.errors import InvalidInputError
from src.domain.skills.dialogue_act_set import (
    parse_set,
    set_distribution,
    to_transcript_like,
)

GOOD = """\
# comentario
=== neg-01 | negociacion | Renovación
-- fase: apertura
Marta: Bueno, empezamos. | gestion
Marta: ¿Habéis visto la oferta? | pregunta #q1
Luis: Sí, ayer. | respuesta >q1
-- fase: negociación
Luis: Podríamos bajar un cinco por ciento. | propuesta #p1
Marta: No, eso no nos da. | rechazo >p1
Luis: Ajá. | escucha

=== cri-01 | coaching | Objetivos
Ana: ¿Qué quieres conseguir? | pregunta #q
Pablo: Lo reviso yo mañana. | compromiso >q
"""


def test_good_set_is_parsed_into_turns_and_links() -> None:
    first, second = parse_set(GOOD, "good.txt")
    assert first.session.id == "neg-01"
    assert first.session.type == "negociacion"
    assert first.speakers == ["Marta", "Luis"]
    ids = [u.id for u in first.utterances]
    assert ids == ["t0.s0", "t0.s1", "t1.s0", "t1.s1", "t2.s0", "t3.s0"]
    answer = first.utterances[2]
    assert answer.act == "respuesta" and answer.responds_to == "t0.s1"
    assert first.utterances[0].phase == "apertura"
    assert first.utterances[3].phase == "negociación"
    assert first.utterances[4].responds_to == "t1.s1"
    assert second.utterances[1].responds_to == "t0.s0"


def test_distribution_counts_every_category() -> None:
    counts = set_distribution(parse_set(GOOD))
    assert len(counts) == 13
    assert counts["pregunta"] == 2 and counts["opinion"] == 0
    assert sum(counts.values()) == 8


def test_transcript_like_shape() -> None:
    document = to_transcript_like(parse_set(GOOD)[0])
    assert document["synthetic"] is True
    assert document["session"] == {
        "id": "neg-01", "type": "negociacion", "title": "Renovación",
        "language": "es"}
    utterance = document["utterances"][2]  # type: ignore[index]
    assert "line" not in utterance
    assert utterance["responds_to"] == "t0.s1"


HEAD = "=== d | conflicto | T\n"


@pytest.mark.parametrize("text, message", [
    ("A: hola | pregunta\n", "before any"),
    ("=== d | otra | T\n", "unknown conversation type"),
    ("=== d | conflicto\n", "expected '=== id"),
    (HEAD + "A hola | pregunta\n", "expected 'speaker"),
    (HEAD + "A: hola | saludo\n", "unknown label"),
    (HEAD + "A: hola | pregunta extra\n", "unexpected"),
    (HEAD + "A: ¿sí? | pregunta #a\nB: sí | respuesta\n", "needs '>anchor"),
    (HEAD + "A: sí | respuesta >x\n", "points at no earlier"),
    (HEAD + "A: ¿sí? | pregunta #a\nA: sí | respuesta >a\n", "own speaker"),
    (HEAD + "A: Es así. | informacion #a\nB: sí | respuesta >a\n",
     "not to a pregunta"),
    (HEAD + "A: Ajá. | escucha #a\nB: Vale. | aceptacion >a\n",
     "cannot reply to a escucha"),
    (HEAD + "A: a | pregunta #a\nB: b | pregunta #a\n", "defined twice"),
    (HEAD + "-- tipo: x\n", "expected '-- fase"),
    (HEAD + HEAD + "A: a | otro\n", "is empty"),
    (HEAD + "A: a | otro\n" + HEAD, "already defined"),
])
def test_errors_name_the_line(text: str, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        parse_set(text, "x.txt")


def test_error_carries_source_and_line_number() -> None:
    with pytest.raises(InvalidInputError, match=r"^x\.txt:3: "):
        parse_set(HEAD + "A: a | otro\nB: b | nada\n", "x.txt")
