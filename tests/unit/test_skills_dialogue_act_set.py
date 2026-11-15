"""Our Spanish dialogue-act set: the text format and its checks."""

import pytest

from src.domain.errors import InvalidInputError
from src.domain.skills.dialogue_act_set import (
    parse_set,
    set_distribution,
    to_transcript_like,
    weighted_count,
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


WEIGHTED = HEAD + """\
A: Pásame el informe hoy. | peticion #p
B: Vale, te lo mando a las cuatro. | compromiso:70 aceptacion:30 >p
A: ¿Lo tienes? | gestion:60 pregunta:40 #q
B: Sí. | respuesta >q
"""


def test_weighted_labels_are_read_and_the_primary_is_the_act() -> None:
    (dialogue,) = parse_set(WEIGHTED)
    promise = dialogue.utterances[1]
    assert promise.act == "compromiso"
    assert [(x.label, x.weight) for x in promise.labels] == [
        ("compromiso", 70), ("aceptacion", 30)]
    assert promise.responds_to == "t0.s0"
    assert dialogue.utterances[0].labels[0].weight == 100
    assert dialogue.utterances[3].responds_to == "t2.s0"
    counts = set_distribution([dialogue])
    assert counts["compromiso"] == 1 and counts["aceptacion"] == 0
    assert weighted_count([dialogue]) == 2


def test_an_answer_may_reply_to_a_secondary_question() -> None:
    (dialogue,) = parse_set(HEAD + "A: Es así, ¿no? | informacion:60 "
                            "pregunta:40 #a\nB: Sí. | respuesta >a\n")
    assert dialogue.utterances[1].responds_to == "t0.s0"


def test_transcript_like_adds_labels_only_when_weighted() -> None:
    document = to_transcript_like(parse_set(WEIGHTED)[0])
    utterances = document["utterances"]
    assert "labels" not in utterances[0]  # type: ignore[index]
    assert utterances[1]["act"] == "compromiso"  # type: ignore[index]
    assert utterances[1]["labels"] == [  # type: ignore[index]
        {"label": "compromiso", "weight": 70},
        {"label": "aceptacion", "weight": 30}]


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
    (HEAD + "A: a | otro:70 gestion:30 extra\n", "unexpected"),
    (HEAD + "A: a | otro:75 gestion:25\n", "steps of 10"),
    (HEAD + "A: a | otro:70 gestion\n", "add up to 70"),
    (HEAD + "A: a | otro:70 gestion:20\n", "add up to 90"),
    (HEAD + "A: a | gestion:30 otro:70\n", "primary label goes first"),
    (HEAD + "A: a | otro:70 saludo:30\n", "unknown label"),
    (HEAD + "A: Haz esto. | peticion #p\n"
     "B: Vale, lo hago. | compromiso:70 aceptacion:30\n",
     "a aceptacion needs '>anchor"),
    (HEAD + "A: Ajá. | escucha #a\n"
     "B: Vale, lo hago. | compromiso:70 aceptacion:30 >a\n",
     "cannot reply to a escucha"),
])
def test_errors_name_the_line(text: str, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        parse_set(text, "x.txt")


def test_error_carries_source_and_line_number() -> None:
    with pytest.raises(InvalidInputError, match=r"^x\.txt:3: "):
        parse_set(HEAD + "A: a | otro\nB: b | nada\n", "x.txt")
