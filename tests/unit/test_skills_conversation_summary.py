"""Second layer: input readers, per-person counts, evaluation, use cases."""

from typing import Any, Dict, List

import pytest

from src.application.skills.conversation_events import (
    analyse,
    evaluate_events,
    person_context,
    pick_session,
    render_events,
)
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError
from src.domain.skills.act_sequence import (
    link_map,
    session_from_json,
    session_from_set,
    sessions_from_json,
    without_links,
)
from src.domain.skills.conversation_events import EventRules, ReplyLink
from src.domain.skills.dialogue_act_set import parse_set, to_transcript_like
from src.domain.skills.event_evaluation import (
    ExpectedEvent,
    ExpectedEvents,
    match,
    score_links,
    score_types,
)
from tests.unit.test_skills_conversation_events import events, session

RULES = EventRules()

SET = """\
=== dec-01 | reunion_decision | Prueba
Marta: ¿Lo habéis mirado? | pregunta #q1
Luis: Sí, ayer. | respuesta >q1
Irene: ¿Y si probamos un mes con Valencia? | propuesta #p1
Marta: Me gusta. | aceptacion >p1
Luis: Vale, de acuerdo. | aceptacion >p1
Luis: Karaoke. | propuesta
Irene: ¿Y el precio? | pregunta
Marta: Luis, ¿llamas tú el jueves? | peticion #r1
Luis: Vale, lo hago. | compromiso:70 aceptacion:30 >r1

=== dec-02 | reunion_decision | Otra
Ana: ¿Vienes? | pregunta #q
Bea: No. | respuesta >q
"""


@pytest.fixture(name="dialogues")
def _dialogues() -> List[Any]:
    """The two small dialogues above, parsed."""
    return parse_set(SET, "set.txt")


# ----------------------------------------------------------- readers


def test_a_set_dialogue_becomes_an_act_session(dialogues: List[Any]) -> None:
    """Ids, turns, order, weighted labels and links are kept."""
    s = session_from_set(dialogues[0])
    assert s.id == "dec-01" and s.synthetic
    assert [x.order for x in s.sentences] == list(range(9))
    last = s.sentences[-1]
    assert last.id == "t7.s0" and last.responds_to == "t6.s0"
    assert last.weight("aceptacion") == 30 and last.weight("rechazo") == 0
    assert s.speakers() == ["Marta", "Luis", "Irene"]
    assert s.sentence("t0.s0").text == "¿Lo habéis mirado?"
    with pytest.raises(KeyError):
        s.sentence("t99.s0")


def test_the_transcript_like_json_reads_back_the_same(
    dialogues: List[Any],
) -> None:
    """``dialogues.json`` (a list) gives the same sessions as the text."""
    documents = [to_transcript_like(d) for d in dialogues]
    read = sessions_from_json(documents, "dialogues.json")
    assert [s.id for s in read] == ["dec-01", "dec-02"]
    assert read[0] == session_from_set(dialogues[0])


def test_the_acts_json_shape_of_the_tagger() -> None:
    """``acts.v1``: weighted labels, times, no turn, extra keys ignored."""
    document: Dict[str, Any] = {
        "schema": "acts.v1", "session": {"id": "demo"},
        "tagger": {"name": "rules", "version": "1"},
        "utterances": [
            {"id": "t0.s0", "speaker": "Ana", "text": "¿Vienes?",
             "start": 1, "end": 2.5, "rest": 3,
             "labels": [{"label": "pregunta", "weight": 90},
                        {"label": "gestion", "weight": 7,
                         "doubtful": True}]},
            {"id": "t1.s0", "speaker": "Bea", "text": "Sí.",
             "act": "respuesta"},
            {"speaker": "Bea", "text": "Claro.", "act": "escucha",
             "turn": 1},
        ],
    }
    s = session_from_json(document)
    assert s.id == "demo" and not s.synthetic
    assert [x.turn for x in s.sentences] == [0, 1, 1]
    assert s.sentences[0].start == 1.0 and s.sentences[0].labels[1].doubtful
    assert s.sentences[2].id == "u2"
    assert sessions_from_json(document) == [s]
    flat = dict(document, session_id="flat")
    del flat["session"]
    assert session_from_json(flat).id == "flat"


@pytest.mark.parametrize("document, message", [
    ({"utterances": []}, "expected 'session.id'"),
    ({"session": {"id": "x"}, "utterances": {}}, "expected 'session.id'"),
    ({"session": {"id": "x"}, "utterances": ["no"]}, "not an object"),
    ({"session": {"id": "x"}, "utterances": [{"speaker": "A"}]},
     "no 'labels' nor 'act'"),
    ({"session": {"id": "x"}, "utterances": [
        {"speaker": "A", "act": "nonsense"}]}, "bad labels"),
    ({"session": {"id": "x"}, "utterances": [
        {"speaker": "A", "labels": [{"label": "pregunta"}]}]}, "bad labels"),
    ({"session": {"id": "x"}, "utterances": [
        {"speaker": "", "act": "pregunta"}]}, "utterance 0"),
    ({"session": {"id": "x"}, "utterances": [
        {"id": "a", "speaker": "A", "act": "pregunta"},
        {"id": "a", "speaker": "B", "act": "respuesta"}]}, "not unique"),
])
def test_bad_acts_documents(document: Dict[str, Any], message: str) -> None:
    """Every malformed document is a clean input error."""
    with pytest.raises(InvalidInputError, match=message):
        session_from_json(document, "acts.json")


def test_a_json_that_is_neither_object_nor_list() -> None:
    """A bare string is refused."""
    with pytest.raises(InvalidInputError, match="object or list"):
        sessions_from_json("x")


def test_without_links_and_link_map(dialogues: List[Any]) -> None:
    """Links can be listed and removed."""
    s = session_from_set(dialogues[0])
    assert link_map(s) == {"t1.s0": "t0.s0", "t3.s0": "t2.s0",
                           "t4.s0": "t2.s0", "t7.s0": "t6.s0"}
    assert link_map(without_links(s)) == {}


# ----------------------------------------------------------- summary


def test_per_person_counts_and_rates(dialogues: List[Any]) -> None:
    """Questions, proposals and their fate, commitments, latency."""
    result = analyse(session_from_set(dialogues[0]), RULES)
    summary = result.summary
    assert summary.links_given == 4 and summary.links_inferred == 0
    assert summary.counts["question_answered"] == 1
    assert summary.counts["proposal_ignored"] == 1
    marta = summary.of("Marta")
    assert (marta.questions_asked, marta.questions_answered) == (1, 1)
    assert marta.answer_rate == 1.0 and marta.proposals_made == 0
    assert marta.acceptance_rate is None and marta.acceptances_given == 1
    luis = summary.of("Luis")
    assert (luis.proposals_made, luis.proposals_ignored) == (1, 1)
    assert luis.ignored_rate == 1.0 and luis.commitments == 1
    assert luis.answers_given == 1 and luis.median_reply_turns == 1.0
    irene = summary.of("Irene")
    assert irene.questions_unanswered == 1 and irene.answer_rate == 0.0
    assert irene.proposals_accepted == 1 and irene.decisions_involved == 1
    assert summary.ignored_by == {"Luis": 1}
    assert summary.median_reply_turns == 1.0
    with pytest.raises(KeyError):
        summary.of("Zoe")


def test_relinking_infers_every_link(dialogues: List[Any]) -> None:
    """``relink`` drops the source links and infers them again."""
    result = analyse(session_from_set(dialogues[0]), RULES, relink=True)
    assert result.relinked and result.summary.links_given == 0
    assert {(k.source, k.target) for k in result.links} >= {
        ("t1.s0", "t0.s0"), ("t3.s0", "t2.s0"), ("t7.s0", "t6.s0")}


def test_render_lists_events_and_people(dialogues: List[Any]) -> None:
    """Every section and the counts-not-ratings line are printed."""
    lines = render_events(analyse(session_from_set(dialogues[0]), RULES))
    text = "\n".join(lines)
    assert lines[0].startswith("Conversation events (given links kept")
    assert "Counts, not ratings" in lines[1]
    for heading in ("Decisions:", "Action items:", "Open issues:",
                    "Questions and proposals:", "Per person:"):
        assert heading in lines
    assert "Luis (asked by Marta): Vale, lo hago.  deadline: el jueves" \
        in text
    assert "Irene → Marta, Luis" in text
    assert "Proposals ignored, by whom they came from: Luis 1" in text
    quiet = render_events(analyse(session(("Ana", "relacion", "Hola.")),
                                  RULES, relink=True))
    assert "every link inferred" in quiet[0] and quiet[0].endswith("none")


def test_person_context_in_both_languages(dialogues: List[Any]) -> None:
    """Plain sentences for the report; nothing for an absent person."""
    summary = analyse(session_from_set(dialogues[0]), RULES).summary
    es = person_context(summary, "Luis", "es")
    assert es == [
        "Respondió a 1 pregunta(s) de otras personas.",
        "Hizo 1 propuesta(s): 0 aceptada(s), 0 rechazada(s), "
        "0 discutida(s), 1 sin respuesta.",
        "Asumió 1 tarea(s).",
        "Recuentos derivados de las etiquetas de actos de diálogo; "
        "contexto, no una valoración.",
    ]
    en = person_context(summary, "Marta", "en")
    assert en[0] == "Asked 1 question(s); 1 got an answer."
    assert en[-1].startswith("Counts derived")
    assert person_context(summary, "Zoe", "es") == []
    assert person_context(summary, "Marta", "fr")[0].startswith("Asked")
    silent = analyse(session(("Ana", "relacion", "Hola.")), RULES).summary
    assert person_context(silent, "Ana", "es") == []


def test_pick_session(dialogues: List[Any]) -> None:
    """By id, the only one, or a clear error."""
    sessions = [session_from_set(d) for d in dialogues]
    assert pick_session(sessions, "dec-02").id == "dec-02"
    assert pick_session(sessions[:1], "").id == "dec-01"
    with pytest.raises(InvalidInputError, match="choose one with"):
        pick_session(sessions, "")
    with pytest.raises(InvalidInputError, match="No session 'x'"):
        pick_session(sessions, "x")
    many = [sessions[0].model_copy(update={"id": f"d{n}"})
            for n in range(8)]
    with pytest.raises(InvalidInputError, match="d5…"):
        pick_session(many, "")


def test_settings_map_onto_the_event_rules() -> None:
    """Every ``event_*`` setting reaches :class:`EventRules`."""
    assert SkillsSettings().event_rules() == EventRules()
    rules = SkillsSettings(event_reply_window=2, event_use_doubtful=True,
                           event_closing_window=0).event_rules()
    assert (rules.reply_window, rules.use_doubtful,
            rules.closing_window) == (2, True, 0)


# -------------------------------------------------------- evaluation


def test_match_prefers_the_key_and_checks_the_owner() -> None:
    """One-to-one, exact keys first, owners must agree."""
    found = events([
        ("Ana", "peticion", "¿Lo haces?"), ("Bea", "aceptacion", "Sí.", 0),
        ("Ciro", "compromiso", "Yo además reviso."),
    ], kind="action_item")
    wanted = [
        ExpectedEvent(type="action_item", sentence="s1", owner="Bea"),
        ExpectedEvent(type="action_item", sentence="s2", owner="Ciro"),
    ]
    pairs = match(wanted, found)
    assert [(e.sentence, d.key) for e, d in pairs] == [("s2", "s2"),
                                                       ("s1", "s0")]
    assert match([ExpectedEvent(type="action_item", sentence="s0",
                                owner="Ana")], found) == []
    assert match([ExpectedEvent(type="decision", sentence="s0")],
                 found) == []


def test_precision_and_recall_per_type() -> None:
    """Derived events of dialogues not in the expectation are ignored."""
    derived = {
        "a": events([("Ana", "pregunta", "¿Y?"),
                     ("Bea", "respuesta", "Sí.", 0)]),
        "b": events([("Ana", "pregunta", "¿Y?")]),
    }
    expected = {"a": [
        ExpectedEvent(type="question_answered", sentence="s0"),
        ExpectedEvent(type="decision", sentence="s1"),
    ]}
    scores = {s.type: s for s in score_types(expected, derived)}
    answered = scores["question_answered"]
    assert (answered.expected, answered.derived, answered.matched) == (
        1, 1, 1)
    assert answered.precision == answered.recall == 1.0
    assert scores["decision"].precision is None
    assert scores["decision"].recall == 0.0
    assert scores["question_unanswered"].derived == 0


def test_link_scores() -> None:
    """Recovered, wrong, missed and spurious inferred links."""
    def link(source: str, target: str) -> ReplyLink:
        return ReplyLink(source=source, target=target, inferred=True,
                         confidence=1.0, turns=1)
    result = score_links(
        {"a": "x", "b": "y", "c": "z"},
        [link("a", "x"), link("b", "x"), link("d", "x")],
    )
    assert (result.recovered, result.wrong, result.missed,
            result.spurious, result.accuracy) == (1, 1, 1, 1, 0.333)


def test_evaluate_events_in_both_modes(dialogues: List[Any]) -> None:
    """Given links, inferred links and link recovery are reported."""
    sessions = [session_from_set(d) for d in dialogues]
    expected = ExpectedEvents(note="mine", dialogues={"dec-02": [
        ExpectedEvent(type="question_answered", sentence="t0.s0")]})
    result = evaluate_events(sessions, expected, RULES)
    assert result.dialogues == ["dec-02"] and result.note == "mine"
    for scores in (result.given_links, result.inferred_links):
        row = next(s for s in scores if s.type == "question_answered")
        assert row.recall == 1.0
    assert result.links.known == 1 and result.links.accuracy == 1.0
    wrong = ExpectedEvents(note="", dialogues={"nope": []})
    with pytest.raises(InvalidInputError, match="nope"):
        evaluate_events(sessions, wrong, RULES)
