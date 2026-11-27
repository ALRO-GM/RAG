"""How far each topic of a live round was covered; what to ask next (#286).

Three separate facts make the coverage of a live interview, all about
the conversation, never about how good the answers were:

* a **concept was asked**: the interviewer's question was about it
  (:class:`~orchestrator.domain.interview.PointAsked`). During the
  session the word rules (:func:`asks_point`: a question that names
  the concept, see below), the live tick call (``interview.ticks@2``)
  or the interviewer by hand say so; after the session the model
  reading the whole round (``interview.attribution@2``) adds the ones
  the session missed. A concept asked counts as **covered** even when
  the answer was poor or absent: the question was asked;
* a **concept was treated**: the candidate talked about it (or said
  they do not know it), as decided since #258/#268;
* the **level of a topic** (:func:`topic_level`), which only grows:

  =================  ==================================================
  ``none``           not mentioned yet
  ``named``          either speaker mentioned it
  ``treated``        at least one of its concepts covered (asked, or
                     treated / said unknown by the candidate), or the
                     topic's own completion rule met; a topic without
                     concepts also when it was asked. A concept asked
                     counts even when the answer was poor or absent
  ``well_treated``   every concept covered (asked or treated), at least
                     half of them treated by the candidate, and at
                     least :data:`WELL_WORDS` words of answers to it;
                     a topic without concepts: its completion rule met
                     and :data:`WELL_WORDS` words
  =================  ==================================================

  A topic counts as **covered** in «{done} de {all}» from
  :data:`COVERED_LEVEL` (``treated``) up. The thresholds are a design
  choice for a quiet hint, not measured.

**What to ask next** (:func:`next_proposal`): the first concept not
covered (neither asked nor treated) of the topic being asked, then of
the next topics in the round's order (what earlier rounds left first);
a topic without concepts and not asked yet is proposed with its first
question idea. Once nothing is left, a concept asked once whose answer
did not treat it (absent, off topic or unclear by the live reading) is
proposed **once more**, if the plan's minutes have not run out
(``again``); a concept asked twice is never proposed again.

Plain data and functions: no I/O.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from orchestrator.domain.interview import (
    COVERAGE_LEVELS,
    CoverageLevel,
    PointAsked,
    PointCoverage,
    PointStatus,
    SheetSlot,
    Slot,
    TopicCoverage,
    fold,
)
from orchestrator.domain.interview_points import names_point, parts

WELL_WORDS = 25
"""Words of answers a topic needs to be «bien tratado»: about ten seconds
of speech (a design choice, not measured)."""
COVERED_LEVEL: CoverageLevel = "treated"
"""The lowest level that counts a topic as covered in the page's count."""
MAX_ASKED = 2
"""Times a concept is proposed at most (the second only as ``again``)."""
_ASKING = re.compile(
    r"^(y |e |entonces |vale |bien |muy bien |perfecto |ok |and |so |"
    r"okay )?(cuentame|cuentanos|hablame|explicame|explica|describeme|"
    r"dime|que|como|cual|cuales|cuanto|cuantos|cuanta|cuantas|por que|"
    r"donde|cuando|has|habeis|sabes|conoces|puedes|podrias|tienes|"
    r"tuviste|usaste|usas|tell me|what|how|which|why|where|when|have "
    r"you|do you|did you|can you|could you)\b"
)
"""How a part of an interviewer's line starts when it asks something
even without «?» (speech recognition drops punctuation)."""


def is_asking(text: str) -> bool:
    """Tell whether an interviewer's line asks something.

    A «?» or «¿», or a part of the line (split at ``.``, ``,``, ``;``,
    ``:``, ``!``) starting with a question word or a request («cuéntame»,
    «explícame»…).
    """
    if "?" in text or "¿" in text:
        return True
    return any(_ASKING.match(" ".join(piece.split()))
               for piece in re.split(r"[.,;:!]+", fold(text)))


def asks_point(point: str, text: str) -> bool:
    """Tell whether an interviewer's line asks about the concept ``point``.

    The line asks something (:func:`is_asking`) and one of its parts
    names the concept by the strict rule of the answers
    (:func:`~orchestrator.domain.interview_points.names_point`); a
    concept with no content word («cuánto tiempo») is named by its
    whole words. A paraphrase is left to the model (live tick call,
    after-session attribution) or the interviewer's own mark.
    """
    if not is_asking(text):
        return False
    whole = " ".join(re.findall(r"[a-z0-9+#]+", fold(point)))
    said = " ".join(re.findall(r"[a-z0-9+#]+", fold(text)))
    if whole and re.search(rf"\b{re.escape(whole)}\b", said):
        return True
    return any(names_point(point, piece) for piece in parts(text))


def point_covered(status: PointStatus, asked: Optional[PointAsked]) -> bool:
    """Whether a concept is covered: asked, or treated / said unknown."""
    return asked is not None or status != "pending"


@dataclass
class TopicFacts:
    """What the round knows of one topic, for :func:`topic_level`.

    ``points``: each concept's status and asked mark, in plan order.
    """

    named: bool = False
    asked: bool = False
    answers: int = 0
    words: int = 0
    complete: bool = False
    points: List[PointCoverage] = field(default_factory=list)


def topic_level(facts: TopicFacts) -> CoverageLevel:
    """The level a topic's facts reach now (see the module's table)."""
    total = len(facts.points)
    covered = sum(1 for p in facts.points
                  if point_covered(p.status, p.asked))
    treated = sum(1 for p in facts.points if p.status != "pending")
    full = (covered == total and 2 * treated >= total if total
            else facts.complete)
    if facts.words >= WELL_WORDS and full:
        return "well_treated"
    if covered or facts.complete or (not total and facts.asked):
        return "treated"
    if facts.named or facts.asked:
        return "named"
    return "none"


def higher(old: CoverageLevel, new: CoverageLevel) -> CoverageLevel:
    """The higher of two levels (a level never goes down)."""
    return max(old, new, key=COVERAGE_LEVELS.index)


def counts(level: CoverageLevel) -> bool:
    """Whether a topic at ``level`` counts as covered (``treated`` up)."""
    return COVERAGE_LEVELS.index(level) >= COVERAGE_LEVELS.index(
        COVERED_LEVEL)


def topic_coverage(slot_id: str, facts: TopicFacts,
                   reached: CoverageLevel = "none",
                   label: str = "") -> TopicCoverage:
    """The topic's coverage record, its level at least ``reached``."""
    return TopicCoverage(
        id=slot_id, label=label, level=higher(reached, topic_level(facts)),
        concepts=len(facts.points),
        asked=sum(1 for p in facts.points if p.asked is not None),
        treated=sum(1 for p in facts.points if p.status != "pending"),
        covered=sum(1 for p in facts.points
                    if point_covered(p.status, p.asked)),
        words=facts.words,
        points=list(facts.points),
    )


@dataclass(frozen=True)
class TopicRef:
    """A topic as :func:`next_proposal` reads it: a plan's slot or a
    scripted demo's topic. ``points``: (id, text) in order;
    ``question``: its first question idea (``None``: none)."""

    id: str
    label: str
    points: Sequence[Tuple[str, str]] = ()
    question: Optional[str] = None

    @staticmethod
    def of(slot: Slot) -> "TopicRef":
        """The reference of a plan's slot."""
        return TopicRef(slot.id, slot.label,
                        tuple((p.id, p.text) for p in slot.points),
                        slot.questions[0] if slot.questions else None)


@dataclass(frozen=True)
class Proposal:
    """The next question the interviewer help proposes (#286).

    ``point`` is ``None`` for a topic without concepts (its own first
    question idea). ``again``: a concept asked once whose answer did
    not treat it, proposed a second time.
    """

    topic: str
    topic_name: str
    point: Optional[str]
    point_text: Optional[str]
    question: str
    again: bool = False

    def to_message(self) -> Dict[str, object]:
        """The ``next`` field of the topics message."""
        return {"topic": self.topic, "topic_name": self.topic_name,
                "point": self.point, "text": self.point_text,
                "question": self.question, "again": self.again}


def _lower(text: str) -> str:
    """``text`` with a lower-case first letter, unless it is an acronym."""
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[:1].lower() + text[1:]


def question_for(topic: str, point: str, again: bool = False,
                 language: str = "es") -> str:
    """The proposed question for one concept, using the plan's words."""
    if language[:2] == "en":
        if again:
            return (f"Going back to {_lower(point)}: could you tell me a "
                    "bit more, with a concrete example?")
        return f"On {_lower(topic)}: what can you tell me about " \
               f"{_lower(point)}?"
    if again:
        return (f"Volviendo a {_lower(point)}: ¿me lo puedes concretar un "
                "poco más, con un ejemplo?")
    return f"Sobre {_lower(topic)}: ¿qué me puedes contar de " \
           f"{_lower(point)}?"


def next_proposal(
    topics: Sequence[TopicRef],
    facts: Mapping[str, TopicFacts],
    asked_times: Mapping[str, int],
    current: Optional[str] = None,
    time_left: bool = True,
    language: str = "es",
) -> Optional[Proposal]:
    """The next concept to ask about (see the module's rule), or ``None``.

    ``topics`` in the round's order; ``asked_times``:
    ``"<topic>.<point>"`` → how many questions asked it; ``current``:
    the topic being asked.
    """
    order = list(topics)
    if current is not None:
        order = ([t for t in order if t.id == current]
                 + [t for t in order if t.id != current])
    for topic in order:
        got = facts.get(topic.id, TopicFacts())
        if not topic.points:
            if topic.question and not got.asked and not got.complete \
                    and not got.answers:
                return Proposal(topic.id, topic.label, None, None,
                                topic.question)
            continue
        states = {p.id: p for p in got.points}
        for pid, text in topic.points:
            state = states.get(pid, PointCoverage(id=pid))
            if not point_covered(state.status, state.asked):
                return Proposal(topic.id, topic.label, pid, text,
                                question_for(topic.label, text,
                                             language=language))
    if not time_left:
        return None
    for topic in order:
        states = {p.id: p for p in facts.get(topic.id, TopicFacts()).points}
        for pid, text in topic.points:
            state = states.get(pid, PointCoverage(id=pid))
            times = asked_times.get(f"{topic.id}.{pid}", 0)
            if state.status == "pending" and state.asked is not None \
                    and times <= 1:
                return Proposal(topic.id, topic.label, pid, text,
                                question_for(topic.label, text, True,
                                             language), again=True)
    return None


def without_words(
    coverage: Sequence[TopicCoverage],
) -> List[Dict[str, Any]]:
    """The coverage for the hand-off record: no quotes (ids, times, levels)."""
    found: List[Dict[str, object]] = []
    for topic in coverage:
        item = topic.model_dump(mode="json")
        for point in item["points"]:
            if point.get("asked"):
                point["asked"].pop("quote", None)
        found.append(item)
    return found


def settled(
    slots: Sequence[Slot],
    facts: Mapping[str, TopicFacts],
    reached: Mapping[str, CoverageLevel],
    final: Sequence[SheetSlot],
    texts: Mapping[str, str],
    model_asked: Mapping[str, Mapping[str, PointAsked]],
    hand: Mapping[str, Mapping[str, PointAsked]],
) -> List[TopicCoverage]:
    """The coverage the stored report keeps, after the session (#286).

    From the session's facts: each concept's status becomes the final
    one (``final``: the slots after the model's attribution, or the
    rules'); a concept not asked during the session takes a mark kept
    by hand in an earlier report of the round (``hand``: a rebuild or
    ``rejudge`` replays only the words), else the model's
    (``model_asked``, ``after_session``); answers and words count the
    final answers too. The level is recomputed and never lower than
    the one the session reached, so the report agrees with the page or
    goes past it, never below.
    """
    by_id = {s.id: s for s in final}
    found: List[TopicCoverage] = []
    for slot in slots:
        got = facts.get(slot.id, TopicFacts())
        done = by_id.get(slot.id)
        status = {p.id: p.status for p in (done.points if done else [])}
        said = [a.utterance for a in (done.answers if done else [])]
        words = sum(len(texts.get(u, "").split()) for u in set(said))
        points = []
        for p in slot.points:
            old = next((x for x in got.points if x.id == p.id),
                       PointCoverage(id=p.id))
            mark = (old.asked or hand.get(slot.id, {}).get(p.id)
                    or model_asked.get(slot.id, {}).get(p.id))
            points.append(PointCoverage(
                id=p.id, text=p.text, status=status.get(p.id, old.status),
                asked=mark))
        now = TopicFacts(
            named=got.named, asked=got.asked or any(
                x.asked for x in points),
            answers=max(got.answers, len(set(said))),
            words=max(got.words, words),
            complete=got.complete or (done is not None
                                      and done.status == "filled"),
            points=points)
        found.append(topic_coverage(slot.id, now,
                                    reached.get(slot.id, "none"),
                                    slot.label))
    return found


def hand_marks(sheet: Mapping[str, object]
               ) -> Dict[str, Dict[str, PointAsked]]:
    """The marks made by hand in a stored report's coverage (#286)."""
    marks: Dict[str, Dict[str, PointAsked]] = {}
    raw = sheet.get("coverage")
    for item in raw if isinstance(raw, list) else []:
        try:
            topic = TopicCoverage.model_validate(item)
        except ValueError:
            continue
        for point in topic.points:
            if point.asked is not None and point.asked.by == "hand":
                marks.setdefault(topic.id, {})[point.id] = point.asked
    return marks
