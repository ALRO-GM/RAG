"""Second layer: reply linking and the events derived from act labels."""

from typing import List, Optional, Sequence, Tuple, Union

import pytest

from src.domain.skills.act_sequence import ActSentence, ActSession
from src.domain.skills.conversation_events import (
    ConversationEvent,
    EventRules,
    deadline,
    derive_events,
    link_replies,
)
from src.domain.skills.dialogue_acts import WeightedLabel

Row = Union[Tuple[str, str, str], Tuple[str, str, str, Optional[int]]]
RULES = EventRules()


def labels(written: str) -> List[WeightedLabel]:
    """Parse ``a:60 b:40`` (a bare label is 100; ``?`` marks doubtful)."""
    found = []
    for token in written.split():
        name, _, weight = token.partition(":")
        doubtful = name.endswith("?")
        found.append(WeightedLabel(
            label=name.rstrip("?"), weight=int(weight or 100),
            doubtful=doubtful,
        ))
    return found


def session(*rows: Row) -> ActSession:
    """Build a session from ``(speaker, labels, text[, reply index])``."""
    sentences = []
    turn = -1
    previous = None
    for order, row in enumerate(rows):
        speaker, written, text = row[0], row[1], row[2]
        target = row[3] if len(row) > 3 else None
        if speaker != previous:
            turn += 1
            previous = speaker
        sentences.append(ActSentence(
            id=f"s{order}", order=order, turn=turn, speaker=speaker,
            text=text, labels=labels(written),
            responds_to=None if target is None else f"s{target}",
            start=float(order),
        ))
    return ActSession(id="t", sentences=sentences)


def events(
    rows: Sequence[Row], rules: EventRules = RULES, kind: str = ""
) -> List[ConversationEvent]:
    """Return the events of ``rows`` (of one ``kind`` if given)."""
    found = derive_events(session(*rows), rules)
    return [e for e in found if not kind or e.type == kind]


def kinds(rows: Sequence[Row], rules: EventRules = RULES) -> List[str]:
    """Return the event types derived from ``rows``, in order."""
    return [e.type for e in events(rows, rules)]


# ----------------------------------------------------------- linking


def test_an_unlinked_answer_goes_to_the_nearest_question_of_another() -> None:
    """Respuesta → the closest earlier pregunta by someone else."""
    s = session(
        ("Ana", "pregunta", "¿Vienes?"),
        ("Ana", "pregunta", "¿Y Luis?"),
        ("Bea", "respuesta", "Sí."),
    )
    (link,) = link_replies(s, RULES)
    assert (link.source, link.target, link.inferred) == ("s2", "s1", True)
    assert link.confidence == 1.0 and link.turns == 1


def test_no_link_to_oneself_or_beyond_the_window() -> None:
    """Own sentences and antecedents too many turns back are skipped."""
    own = session(("Ana", "pregunta", "¿Sí?"), ("Ana", "respuesta", "Sí."))
    assert link_replies(own, RULES) == []
    far = session(
        ("Ana", "pregunta", "¿Sí?"), ("Bea", "informacion", "a"),
        ("Ana", "informacion", "b"), ("Bea", "informacion", "c"),
        ("Ana", "informacion", "d"), ("Bea", "respuesta", "Sí."),
    )
    assert link_replies(far, RULES) == []
    assert len(link_replies(far, EventRules(reply_window=5))) == 1


def test_link_confidence_decays_with_distance() -> None:
    """Each extra turn costs ``link_decay``."""
    s = session(
        ("Ana", "propuesta", "¿Y si…?"), ("Bea", "informacion", "x"),
        ("Ana", "informacion", "y"), ("Bea", "aceptacion", "Vale."),
    )
    (link,) = link_replies(s, RULES)
    assert link.target == "s0" and link.turns == 3
    assert link.confidence == pytest.approx(0.7)
    assert link_replies(s, EventRules(min_link_confidence=0.8)) == []


def test_acceptance_prefers_a_proposal_over_a_nearer_opinion() -> None:
    """The prior of the antecedent's type outweighs one turn."""
    s = session(
        ("Ana", "propuesta", "¿Y si…?"), ("Ana", "opinion", "Creo que…"),
        ("Bea", "informacion", "x"), ("Ana", "opinion", "Mejor."),
        ("Bea", "aceptacion", "Vale."),
    )
    links = link_replies(s, RULES)
    assert [(k.source, k.target) for k in links] == [("s4", "s0")]


def test_given_links_are_kept_and_bad_ones_dropped() -> None:
    """A source link is kept at confidence 1; one to oneself is not."""
    s = session(
        ("Ana", "pregunta", "¿Sí?"), ("Bea", "respuesta", "Sí.", 0),
        ("Bea", "respuesta", "Y no.", 1),
    )
    links = link_replies(s, RULES)
    assert [(k.source, k.inferred, k.confidence) for k in links] == [
        ("s1", False, 1.0)]


def test_a_link_to_an_unknown_sentence_is_ignored() -> None:
    """``responds_to`` pointing nowhere gives no link."""
    s = session(("Ana", "pregunta", "¿Sí?"), ("Bea", "respuesta", "Sí."))
    broken = s.model_copy(update={"sentences": [
        s.sentences[0],
        s.sentences[1].model_copy(update={"responds_to": "nope"}),
    ]})
    assert link_replies(broken, RULES) == []


def test_commitment_takes_a_request_only_if_nobody_else_did() -> None:
    """A compromiso joins a free petición, not one taken by another."""
    s = session(
        ("Ana", "peticion", "¿Me mandas los datos?"),
        ("Bea", "aceptacion", "Sí.", 0),
        ("Bea", "compromiso", "Te los mando hoy."),
        ("Ciro", "compromiso", "Yo reviso el informe."),
    )
    links = {k.source: k.target for k in link_replies(s, RULES)}
    assert links == {"s1": "s0", "s2": "s0"}


def test_label_weight_and_doubtful_labels() -> None:
    """A 40 % question counts; 20 % does not; doubtful only on request."""
    def answered(written: str, rules: EventRules = RULES) -> List[str]:
        return kinds([("Ana", written, "Luis, ¿y tú?"),
                      ("Luis", "respuesta", "Bien.")], rules)
    assert answered("gestion:60 pregunta:40") == ["question_answered"]
    assert answered("gestion:80 pregunta:20") == []
    assert answered("gestion:60 pregunta?:40") == []
    assert answered("gestion:60 pregunta?:40",
                    EventRules(use_doubtful=True)) == ["question_answered"]


# --------------------------------------------------------- questions


def test_question_answered_with_responders_and_latency() -> None:
    """Who answered and after how many turns."""
    (event,) = events([
        ("Ana", "pregunta", "¿Cuándo?"), ("Bea", "escucha", "Mm."),
        ("Ciro", "respuesta", "El lunes.", 0),
    ])
    assert event.type == "question_answered"
    assert event.responders == ["Ciro"] and event.latency_turns == 2
    assert event.sentences == ["s0", "s2"]
    assert event.people == ["Ana", "Ciro"] and event.start == 0.0


def test_unanswered_question_is_an_open_issue() -> None:
    """No reply: unanswered, and still open at the end."""
    found = events([
        ("Ana", "pregunta", "¿Cuándo?"), ("Bea", "informacion", "x"),
        ("Ana", "informacion", "y"), ("Bea", "informacion", "z"),
        ("Ana", "informacion", "w"),
    ])
    assert [e.type for e in found] == ["question_unanswered", "open_issue"]
    assert found[0].confidence == 1.0
    assert found[1].source == found[0].id


def test_a_question_at_the_very_end_is_weakly_unanswered() -> None:
    """Confidence grows with the turns that passed without a reply."""
    (event, _) = events([("Ana", "informacion", "x"),
                         ("Bea", "pregunta", "¿Y?")])
    assert event.type == "question_unanswered"
    assert event.confidence == pytest.approx(1 / 5)


# --------------------------------------------------------- proposals


@pytest.mark.parametrize("replies, expected", [
    ([("Bea", "aceptacion", "Vale.", 0)], "proposal_accepted"),
    ([("Bea", "compromiso", "Lo hago yo.", 0)], "proposal_accepted"),
    ([("Bea", "rechazo", "No.", 0)], "proposal_rejected"),
    ([("Bea", "aceptacion", "Vale.", 0), ("Ciro", "rechazo", "No.", 0)],
     "proposal_contested"),
    ([("Bea", "aceptacion", "Vale.", 0), ("Bea", "rechazo", "Bueno, no.", 0)],
     "proposal_rejected"),
    ([("Bea", "escucha", "Mm.")], "proposal_ignored"),
])
def test_fate_of_a_proposal(replies: List[Row], expected: str) -> None:
    """Accepted, rejected, contested or ignored; the last stance stands."""
    found = events([("Ana", "propuesta", "¿Y si lo probamos?"), *replies],
                   kind=expected)
    assert len(found) == 1


def test_an_ignored_proposal_is_open_unless_a_decision_follows() -> None:
    """A later decision closes the competing proposals."""
    ignored: List[Row] = [("Ana", "propuesta", "Karaoke."),
                          ("Bea", "escucha", "Ajá.")]
    assert "open_issue" in kinds(ignored)
    decided: List[Row] = ignored + [
        ("Ciro", "propuesta", "¿Y una cena?"),
        ("Ana", "aceptacion", "Vale.", 2),
    ]
    assert "open_issue" not in kinds(decided)


def test_a_contested_proposal_stays_open_without_a_decision() -> None:
    """Standing acceptance and rejection and nothing decided after."""
    rows: List[Row] = [
        ("Ana", "propuesta", "¿Y si…?"), ("Bea", "aceptacion", "Sí.", 0),
        ("Ciro", "rechazo", "No.", 0),
    ]
    assert kinds(rows) == ["proposal_contested", "open_issue"]


# --------------------------------------------------------- decisions


def test_an_accepted_proposal_is_a_decision() -> None:
    """Proposal + acceptance and no standing rejection."""
    rows: List[Row] = [("Ana", "propuesta", "¿Y si lo probamos un mes?"),
                       ("Bea", "aceptacion", "Vale.", 0)]
    (decision,) = events(rows, kind="decision")
    assert decision.key == "s0" and decision.sentences == ["s0", "s1"]
    assert events(rows, EventRules(min_acceptances=2), "decision") == []


def test_a_closing_with_outcome_words_is_a_decision_alone() -> None:
    """"Entonces queda…" states a decision even without replies."""
    (decision,) = events([("Ana", "gestion", "Pues queda así: el lunes.")],
                         kind="decision")
    assert decision.key == "s0" and decision.confidence == 1.0


def test_a_rejected_closing_is_no_decision() -> None:
    """A standing rejection of the summary blocks it."""
    rows: List[Row] = [("Ana", "gestion", "Resumiendo: el lunes."),
                       ("Bea", "rechazo", "No, el martes.", 0)]
    assert events(rows, kind="decision") == []


def test_a_gestion_sentence_needs_enough_acceptances() -> None:
    """Without closing words, two acceptances (or all the others)."""
    one: List[Row] = [("Ana", "gestion", "Hacemos la prueba en Valencia."),
                      ("Bea", "aceptacion", "Vale.", 0),
                      ("Ciro", "escucha", "Mm.")]
    assert events(one, kind="decision") == []
    two = one[:2] + [("Ciro", "aceptacion", "Sí.", 0)]
    assert len(events(two, kind="decision")) == 1
    pair: List[Row] = one[:2]
    assert len(events(pair, kind="decision")) == 1


def test_a_turn_giving_question_is_never_a_closing() -> None:
    """``gestion + pregunta`` is answered, not decided."""
    rows: List[Row] = [("Ana", "gestion:60 pregunta:40", "¿Queda claro?"),
                       ("Bea", "aceptacion", "Sí.", 0)]
    assert events(rows, kind="decision") == []


def test_a_closing_absorbs_the_proposals_it_sums_up() -> None:
    """One decision, not two, when a summary follows the acceptance."""
    rows: List[Row] = [
        ("Ana", "propuesta", "¿Y si probamos un mes?"),
        ("Bea", "aceptacion", "Vale.", 0),
        ("Ana", "gestion", "Entonces queda: un mes de prueba."),
        ("Bea", "aceptacion", "Hecho.", 2),
    ]
    (decision,) = events(rows, kind="decision")
    assert decision.key == "s2"
    assert decision.sentences == ["s2", "s0", "s1", "s3"]
    apart = events(rows, EventRules(closing_window=0), "decision")
    assert [e.key for e in apart] == ["s2", "s0"]


# ------------------------------------------------------ action items


def test_an_accepted_request_is_an_action_item_of_the_accepter() -> None:
    """Owner, who asked, and the deadline in the request."""
    (item,) = events([
        ("Ana", "peticion", "Luis, ¿llamas tú esta semana?"),
        ("Luis", "aceptacion", "Sí, claro.", 0),
    ], kind="action_item")
    assert (item.owner, item.by) == ("Luis", "Ana")
    assert item.deadline == "esta semana"
    assert item.what == "Luis, ¿llamas tú esta semana?"


def test_acceptance_and_commitment_of_one_turn_are_one_item() -> None:
    """"Sí, claro. Les llamo el jueves." is a single task."""
    (item,) = events([
        ("Ana", "peticion", "¿Puedes llamar tú?"),
        ("Luis", "aceptacion", "Sí, claro.", 0),
        ("Luis", "compromiso", "Les llamo el jueves por la mañana."),
    ], kind="action_item")
    assert item.sentences == ["s0", "s1", "s2"]
    assert item.what == "Les llamo el jueves por la mañana."
    assert item.deadline == "el jueves por la mañana"


def test_a_given_commitment_link_after_the_turn_is_kept_once() -> None:
    """A commitment in a later turn joins the request it points at."""
    (item,) = events([
        ("Ana", "peticion", "¿Me lo mandas?"),
        ("Luis", "aceptacion", "Sí.", 0), ("Ana", "relacion", "Gracias."),
        ("Luis", "compromiso", "Te lo mando mañana.", 0),
    ], kind="action_item")
    assert item.sentences == ["s0", "s1", "s3"] and item.deadline == "mañana"


def test_a_lone_commitment_is_an_action_item() -> None:
    """The speaker owns it; no deadline when none is said."""
    (item,) = events([("Irene", "compromiso", "Yo preparo la lista.")],
                     kind="action_item")
    assert (item.owner, item.by, item.deadline) == ("Irene", "Irene", None)
    assert item.sentences == ["s0"]


def test_a_rejected_request_is_closed_and_an_ignored_one_open() -> None:
    """No item either way; only the ignored request stays open."""
    rejected: List[Row] = [("Ana", "peticion", "¿Lo haces tú?"),
                           ("Bea", "rechazo", "No puedo.", 0)]
    assert kinds(rejected) == []
    ignored: List[Row] = [("Ana", "peticion", "¿Lo haces tú?"),
                          ("Bea", "informacion", "Hoy no estoy.")]
    (issue,) = events(ignored)
    assert issue.type == "open_issue" and issue.source is None


@pytest.mark.parametrize("text, expected", [
    ("Te lo mando mañana antes de las diez.", "mañana"),
    ("Lo tengo antes del viernes.", "antes del viernes"),
    ("Esta tarde te lo paso.", "Esta tarde"),
    ("La semana que viene te paso tres.", "La semana que viene"),
    ("Lo reviso en dos semanas.", "en dos semanas"),
    ("A finales de marzo.", "A finales de marzo"),
    ("Yo me encargo.", None),
    ("Mándelo a la dirección de la clínica.", None),
    ("Lo cierro a las diez y media.", "a las diez y media"),
    ("Lo quiero para el 5 de marzo.", "para el 5 de marzo"),
])
def test_deadline_expressions(text: str, expected: Optional[str]) -> None:
    """Spanish time expressions are read as the deadline."""
    assert deadline([text]) == expected


def test_events_are_numbered_and_grouped() -> None:
    """Ids follow the order of creation; groups come in a fixed order."""
    found = events([
        ("Ana", "compromiso", "Lo hago yo."),
        ("Bea", "pregunta", "¿Seguro?"),
        ("Ana", "respuesta", "Sí.", 1),
    ])
    assert [(e.id, e.type) for e in found] == [
        ("e1", "question_answered"), ("e2", "action_item")]
