"""Coverage of a live round (#286): concepts asked vs treated, topic levels,
the next concept to ask, and what the stored report keeps.

Invented lines and plans only.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from jsonschema import Draft202012Validator

from orchestrator.adapters.outbound.fakes import FakeReasoner
from orchestrator.application.live_judgement import (
    ATTRIBUTION_TEMPLATE,
    LiveJudge,
    replayed_tracker,
)
from orchestrator.application.next_round import TopicTracker
from orchestrator.application.ports import CancelToken, Document
from orchestrator.domain.events import Transcript
from orchestrator.domain.interview import (
    InterviewPlan,
    PointAsked,
    PointCoverage,
    Sheet,
)
from orchestrator.domain.interview_attribution import (
    RoundLine,
    asked_marks,
    merged,
    read_attribution,
)
from orchestrator.domain.interview_coverage import (
    WELL_WORDS,
    TopicFacts,
    TopicRef,
    asks_point,
    counts,
    hand_marks,
    higher,
    is_asking,
    next_proposal,
    question_for,
    topic_level,
    without_words,
)
from orchestrator.domain.next_round import topic_states
from orchestrator.domain.offer_board import interview_view
from tests.unit.orchestrator.test_candidate import final
from tests.unit.orchestrator.test_interview_attribution import (
    the_plan,
    wiring,
)
from tests.unit.orchestrator.test_interview_usecases import Store

ROOT = Path(__file__).resolve().parents[3]
ROLES = {"S1": "interviewer", "S2": "candidate"}
SESSION = "cov-1"

# An invented round on the running shop's plan (attribution_set.json):
# s1 asked and answered at length, s2 asked with no answer, s3 asked and
# answered in part, the rest never reached.
ROUND: List[Tuple[str, str, str]] = [
    ("q1", "S1", "Hola, gracias por venir. ¿Dónde trabajaste antes y "
                 "cuánto tiempo?"),
    ("c1", "S2", "Estuve tres años en una zapatería del barrio, atendiendo "
                 "clientes, reponiendo el género y preparando los pedidos "
                 "que llegaban cada semana al almacén de la tienda."),
    ("q2", "S1", "Cuéntame qué tipos de pisada conoces."),
    ("c2", "S2", "Pues la verdad es que eso no lo tengo muy claro."),
    ("q3", "S1", "¿Y con la caja, cobras con tarjeta y efectivo?"),
    ("c3", "S2", "Sí, cobraba con tarjeta y en efectivo, y al cerrar "
                 "cuadraba la caja cada noche con el cierre."),
]


def transcript(lines: List[Tuple[str, str, str]] = ROUND) -> Transcript:
    return Transcript.model_validate({
        "schema": "transcript.v1",
        "session": {"id": SESSION, "language": "es",
                    "started_at": "2026-11-27T10:00:00Z"},
        "speakers": [{"id": "S1", "role": "interviewer"},
                     {"id": "S2", "role": "candidate"}],
        "utterances": [
            {"id": uid, "turn": n, "index": 0, "speaker": who,
             "start": 5.0 * n, "end": 5.0 * n + 4.0, "text": text}
            for n, (uid, who, text) in enumerate(lines)],
    })


def tracker(plan: Optional[InterviewPlan] = None,
            ) -> Tuple[TopicTracker, List[Document]]:
    plan = plan or the_plan()
    published: List[Document] = []
    track = TopicTracker(plan, topic_states(plan, []), dict(ROLES),
                         published.append)
    track.start()
    return track, published


def say(track: TopicTracker, lines: List[Tuple[str, str, str]]) -> None:
    for n, (uid, who, text) in enumerate(lines):
        track.on_event(final(uid, who, text, seq=n, at=5.0 * n))


# ---- the rules -----------------------------------------------------------

@pytest.mark.parametrize("text,asking", [
    ("¿Qué tipos de pisada conoces?", True),
    ("cuéntame qué tipos de pisada conoces", True),  # no «?» (speech)
    ("Vale, y cómo eliges la talla", True),
    ("La pisada es importante en esta tienda.", False),
    ("Tell me about the cushioning", True),
])
def test_a_line_asks_with_a_question_mark_or_a_question_word(
        text: str, asking: bool) -> None:
    assert is_asking(text) is asking


def test_a_concept_is_asked_when_a_question_names_it() -> None:
    assert asks_point("Tipos de pisada", "¿Qué tipos de pisada conoces?")
    assert asks_point("Cuánto tiempo", "¿Y cuánto tiempo estuviste?")
    assert not asks_point("Tipos de pisada", "La pisada importa mucho.")
    assert not asks_point("Cierre de caja", "¿Qué tal el viaje?")


Given = List[Tuple[Optional[str], Optional[str]]]


def facts(points: Given, **kw: Any) -> TopicFacts:
    """A topic's facts: per concept (status or None, "yes" if asked)."""
    return TopicFacts(points=[
        PointCoverage.model_validate({
            "id": f"p{n}", "status": status or "pending",
            "asked": {"by": "rules"} if asked else None})
        for n, (status, asked) in enumerate(
            ((s, a) for s, a in points), start=1)], **kw)


@pytest.mark.parametrize("given,level", [
    (facts([]), "none"),
    (facts([], named=True), "named"),
    (facts([(None, "yes")]), "named"),                   # asked, no answer
    (facts([(None, "yes"), (None, None)], asked=True, answers=1), "named"),
    (facts([("treated", None), (None, None)], answers=1, words=40),
     "treated"),                                          # not all covered
    (facts([("treated", None), (None, "yes")], answers=1,
           words=WELL_WORDS), "well_treated"),
    (facts([("treated", None), (None, "yes")], answers=1,
           words=WELL_WORDS - 1), "treated"),
    (facts([("treated", None), (None, "yes"), (None, "yes")], answers=1,
           words=99), "treated"),                         # 1 of 3 treated
    (facts([], asked=True, answers=1, words=3), "treated"),
    (facts([], asked=True, answers=2, words=WELL_WORDS, complete=True),
     "well_treated"),
])
def test_the_level_rule(given: TopicFacts, level: str) -> None:
    assert topic_level(given) == level


def test_a_level_never_goes_down_and_treated_counts() -> None:
    assert higher("well_treated", "named") == "well_treated"
    assert higher("none", "treated") == "treated"
    assert [counts(x) for x in ("none", "named", "treated",
                                "well_treated")] == [False, False, True,
                                                     True]


def test_the_next_proposal_goes_point_by_point_then_topic_by_topic(
) -> None:
    topics = [TopicRef("a", "Caja", (("p1", "Cierre de caja"),
                                     ("p2", "Devoluciones"))),
              TopicRef("b", "Inglés", (), "¿Qué nivel de inglés tienes?")]
    found = next_proposal(topics, {}, {})
    assert found is not None and (found.topic, found.point) == ("a", "p1")
    assert found.question == ("Sobre caja: ¿qué me puedes contar de "
                              "cierre de caja?")
    asked = {"a": facts([(None, "yes"), (None, None)])}
    nxt = next_proposal(topics, asked, {"a.p1": 1})
    assert nxt is not None and nxt.point == "p2"  # asked: not again yet
    done = {"a": facts([("treated", None), (None, "yes")])}
    other = next_proposal(topics, done, {"a.p2": 1})
    assert other is not None and other.topic == "b" and other.point is None
    assert other.question == "¿Qué nivel de inglés tienes?"
    end = {**done, "b": TopicFacts(asked=True, answers=1)}
    again = next_proposal(topics, end, {"a.p2": 1})
    assert again is not None and again.again and again.point == "p2"
    assert again.question.startswith("Volviendo a devoluciones")
    assert next_proposal(topics, end, {"a.p2": 2}) is None  # twice: never
    assert next_proposal(topics, end, {"a.p2": 1}, time_left=False) is None
    first = next_proposal(topics, {}, {}, current="b")
    assert first is not None and first.topic == "b"
    assert question_for("Caja", "Devoluciones", language="en") == (
        "On caja: what can you tell me about devoluciones?")


# ---- the live round --------------------------------------------------------

def test_the_round_records_asked_and_treated_and_levels() -> None:
    track, published = tracker()
    say(track, ROUND)
    cover = {c.id: c for c in track.coverage()}
    s1, s2, s3 = cover["s1"], cover["s2"], cover["s3"]
    assert s1.level == "well_treated" and s1.treated == 3
    asked = {p.id: p.asked for p in s1.points}
    assert asked["p1"] is not None and asked["p1"].by == "rules"
    assert asked["p1"].utterance == "q1" and asked["p1"].at == 0.0
    assert asked["p1"].quote.startswith("Hola, gracias")
    # Asked, the answer treated nothing: covered, but not by the candidate.
    p1 = s2.points[0]
    assert p1.asked is not None and p1.status == "pending"
    assert s2.level == "named" and s2.covered == 1 and s2.treated == 0
    assert s3.level == "treated" and s3.asked == 1 and s3.treated == 2
    assert cover["s4"].level == "none" and cover["s4"].asked == 0
    last = published[-1]
    assert last["next"]["topic"] == "s3" and last["next"]["point"] == "p3"
    topic = next(t for t in last["topics"] if t["id"] == "s2")
    assert topic["coverage"] == {"level": "named", "concepts": 3,
                                 "asked": 1, "treated": 0, "covered": 1}
    assert topic["points"][0]["asked"] == {"by": "rules", "at": 10.0}
    assert topic["points"][1]["asked"] is None


def test_a_level_reached_stays_and_a_mention_names_a_topic() -> None:
    track, published = tracker()
    say(track, [("c0", "S2", "Antes de nada: sé algo de inglés básico.")])
    assert track.reached()["s6"] == "named"
    say(track, ROUND)
    before = track.reached()["s1"]
    track.points["s1"][0].status = "pending"  # a later reading
    assert track.reached()["s1"] == before == "well_treated"


def test_the_interviewer_marks_a_concept_by_hand_and_undoes_only_that(
) -> None:
    track, published = tracker()
    say(track, ROUND)
    count = len(published)
    view = track.mark_asked("s3", "p3", True, 42.0)
    assert view["asked"]["by"] == "hand" and view["asked"]["at"] == 42.0
    assert len(published) == count + 1
    assert published[-1]["next"]["topic"] != "s3"
    assert track.mark_asked("s3", "p3", True, 43.0)["asked"]["at"] == 42.0
    assert track.mark_asked("s3", "p3", False, 44.0)["asked"] is None
    # A mark by the rules is never undone by hand.
    assert track.mark_asked("s2", "p1", False, 45.0)["asked"]["by"] == \
        "rules"
    with pytest.raises(KeyError):
        track.mark_asked("s3", "p9", True, 1.0)


def test_the_tick_call_marks_asked_only_by_an_interviewers_line() -> None:
    track, _ = tracker()
    say(track, [("q1", "S1", "Vamos a hablar de zapatillas."),
                ("c1", "S2", "Me encantan, corro todas las semanas.")])
    listed = track.focus_points()
    track.suggested(listed, {"points": [], "asked": [
        {"id": "s2.p2", "utterance": "q1"},
        {"id": "s2.p3", "utterance": "c1"},    # the candidate's: dropped
        {"id": "s2.p1", "utterance": "nope"},  # unknown line: dropped
        {"id": "s9.p1", "utterance": "q1"}]})  # not listed: dropped
    marks = track.asked["s2"]
    assert set(marks) == {"p2"} and marks["p2"].by == "live_model"
    assert marks["p2"].utterance == "q1"


# ---- after the session -----------------------------------------------------

def test_the_model_reads_asked_concepts_from_interviewers_lines() -> None:
    plan = the_plan()
    lines = [RoundLine("q1", "interviewer", "¿Y el calzado, cómo pisa?", 3.0),
             RoundLine("c1", "candidate", "No sé.", 6.0),
             RoundLine("q2", "interviewer", "¿Y la amortiguación?", 9.0)]
    stats: Dict[str, int] = {}
    found = read_attribution({"slots": [{"id": "s2", "answers": [],
                                         "asked": [
        {"id": "p1", "by": "q2"}, {"id": "p1", "by": "[q1]"},
        {"id": "p2", "by": "c1"}, {"id": "p9", "by": "q1"}, "x"]}]},
        plan, lines, stats)
    assert found["s2"].asked == {"p1": "q1"}  # the earliest question
    assert stats["kept"] == 2  # p1 twice; the rest dropped
    joined = merged([found, {"s2": type(found["s2"])(asked={"p1": "q2"})}],
                    ["q1", "c1", "q2"])
    assert joined["s2"].asked == {"p1": "q1"}
    marks = asked_marks(joined, lines)
    assert marks["s2"]["p1"] == PointAsked(
        by="after_session", at=3.0, utterance="q1",
        quote="¿Y el calzado, cómo pisa?")


class Asks(FakeReasoner):
    """The fake reasoner, whose attribution also says s2.p2 was asked by
    q2 (a paraphrase the rules miss) and s4.p1 by nobody valid."""

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        answer = super().reason(request, cancel)
        if request.get("template") == ATTRIBUTION_TEMPLATE:
            data = answer.get("data") or {"slots": []}
            data["slots"] = [*data["slots"],
                             {"id": "s2", "answers": [], "asked": [
                                 {"id": "p2", "by": "q2"},
                                 {"id": "p1", "by": "q3"}]},
                             {"id": "s4", "answers": [], "asked": [
                                 {"id": "p1", "by": "c1"}]}]
            answer["data"] = data
        return answer


def stored(store: Store) -> Dict[str, Any]:
    # The newest of the session (this fake store appends; the real one
    # replaces the file).
    kept: Dict[str, Any] = [x for x in store.results("set-running")
                            if x.get("session_id") == SESSION][-1]
    schema = json.loads((ROOT / "contracts"
                         / "interview.sheet.v1.schema.json").read_text())
    assert not list(Draft202012Validator(schema).iter_errors(kept))
    return kept


def judge(store: Store, reasoner: Any, attribute: bool = True,
          lines: List[Tuple[str, str, str]] = ROUND) -> Optional[Document]:
    plan = the_plan()
    store.save(plan)
    topics = replayed_tracker(transcript(lines), plan)
    return LiveJudge(plan, wiring(store, reasoner), SESSION,
                     lambda m: None, attribute=attribute).judge(topics)


def test_the_stored_report_keeps_every_topic_asked_and_treated() -> None:
    store = Store()
    result = judge(store, Asks())
    assert result is not None and result["coverage"]
    sheet = Sheet.model_validate(stored(store))
    assert sheet.coverage is not None
    cover = {c.id: c for c in sheet.coverage}
    assert list(cover) == [s.id for s in the_plan().slots]  # every topic
    s2 = {p.id: p for p in cover["s2"].points}
    assert s2["p1"].asked is not None and s2["p1"].asked.by == "rules"
    # The model adds the question the rules missed, after the session.
    assert s2["p2"].asked is not None and s2["p2"].asked.by == \
        "after_session" and s2["p2"].asked.utterance == "q2"
    assert s2["p3"].asked is None  # never asked: the interviewer's gap
    assert all(p.asked is None for p in cover["s4"].points)
    assert cover["s1"].level == "well_treated"


def test_a_rejudge_keeps_the_marks_made_by_hand() -> None:
    store = Store()
    judge(store, FakeReasoner(), attribute=False)
    kept = json.loads(json.dumps(stored(store)))
    store.sheets.clear()
    for topic in kept["coverage"]:
        if topic["id"] == "s5":
            topic["points"][0]["asked"] = {"by": "hand", "at": 61.5,
                                           "utterance": None, "quote": ""}
    store.save_result("set-running", kept)
    assert hand_marks(kept) == {"s5": {"p1": PointAsked(by="hand",
                                                        at=61.5)}}
    judge(store, FakeReasoner(), attribute=False)
    again = {c["id"]: c for c in stored(store)["coverage"]}
    assert again["s5"]["points"][0]["asked"]["by"] == "hand"
    assert again["s5"]["level"] == "named"


def test_the_hand_off_coverage_has_no_words() -> None:
    track, _ = tracker()
    say(track, ROUND)
    found = without_words(track.coverage())
    text = json.dumps(found, ensure_ascii=False)
    assert "gracias" not in text and "zapatería" not in text
    assert found[0]["points"][0]["asked"]["utterance"] == "q1"


def test_the_board_figure_does_not_change_with_coverage() -> None:
    store = Store()
    judge(store, Asks())
    sheet = stored(store)
    plan = the_plan()
    with_it = interview_view(plan, sheet)
    without = interview_view(plan, {k: v for k, v in sheet.items()
                                    if k != "coverage"})
    assert with_it.figure == without.figure
    by_id = {p.id: p for p in with_it.points}
    assert by_id["s2"].asked == 2 and by_id["s2"].coverage == "named"
    assert by_id["s4"].asked == 0 and by_id["s4"].coverage == "none"
    assert all(p.asked is None and p.coverage is None
               for p in without.points)
