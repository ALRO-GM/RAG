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
proposed **once more**, once the candidate has answered that question
and the answer did not treat it (off topic, unclear or empty by the
live reading), if the plan's minutes have not run out (``again``); a
concept asked twice is never proposed again.

Plain data and functions: no I/O.
"""

import math
import re
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple,
)

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

WELL_WORDS = 25
"""Words of answers a topic needs to be «bien tratado»: about ten seconds
of speech (a design choice, not measured)."""
COVERED_LEVEL: CoverageLevel = "treated"
"""The lowest level that counts a topic as covered in the page's count."""
MAX_ASKED = 2
"""Times a concept is proposed at most (the second only as ``again``)."""
_REQUEST = re.compile(
    r"^(y |e |entonces |vale |bien |muy bien |perfecto |ok |and |so |"
    r"okay |now |ahora )?(cuentame|cuentanos|hablame|hablanos|explicame|"
    r"explicanos|describeme|describenos|dime|tell me|talk me through|"
    r"explain|describe|walk me through)\b"
)
"""How a request starts («cuéntame…», «háblame de…»): a clause that asks
even without «?»."""
_INTERROGATIVE = re.compile(
    r"^(y |e |entonces |vale |bien |muy bien |and |so |okay )?(que|como|"
    r"cual|cuales|cuanto|cuantos|cuanta|cuantas|por que|para que|donde|"
    r"cuando|quien|has|habeis|hiciste|sabes|conoces|puedes|podrias|"
    r"tienes|tuviste|usaste|usas|trabajaste|estas|estuviste|te|what|how|"
    r"which|why|where|when|who|have you|do you|did you|can you|could you|"
    r"are you|were you)\b"
)
"""How a clause ending in «?» without «¿» starts when it is a question
(speech recognition may drop the «¿»)."""
_TAG = re.compile(
    r"^(vale|no|verdad|de acuerdo|cierto|si|ok|okay|eh|vale no|me "
    r"explico|entiendes|sabes|me sigues|right|ok then|isn t it|don t you|"
    r"yes)$"
)
"""Tag questions («¿vale?», «¿no?», «¿verdad?»): they confirm a
statement, they do not ask about a concept."""
_NOT_ASKING = re.compile(
    r"\b(no (vamos a|voy a|hace falta|te voy a|te pregunto|hablaremos|"
    r"entramos|tocamos|preguntare)|sin entrar en|dejamos de lado|"
    r"let s not|we won t|i won t ask)\b"
)
"""A clause that says a topic will not be discussed."""
_CLAUSE_END = re.compile(r"[.;:!]+")


def _words(text: str) -> str:
    """``text`` folded to its words, single-spaced."""
    return " ".join(re.findall(r"[a-z0-9+#]+", fold(text)))


def question_clauses(text: str) -> List[str]:
    """The clauses of an interviewer's line that ask something (#286).

    Conservative: a clause between «¿» and «?»; else a sentence ending
    in «?» whose last comma part starts with a question word (speech
    recognition may drop the «¿»); else a clause that starts with a
    request («cuéntame…», «háblame de…»). Tag questions («¿vale?»,
    «¿no?») and clauses saying a topic will not be discussed are not
    questions. A statement with a question word («como te decía…»,
    «cuando entres…») never is: a missed question is left to the model
    and the interviewer's button, a false one would hide a gap.
    """
    found: List[str] = []
    rest = text
    for match in re.finditer(r"¿([^?¿]*)\?", text):
        found.append(match.group(1))
        rest = rest.replace(match.group(0), " . ")
    for sentence in re.findall(r"[^.;:!?¿]*\?", rest):
        last = sentence.split(",")[-1]
        if _INTERROGATIVE.match(_words(last)):
            found.append(last.rstrip("?"))
    rest = re.sub(r"[^.;:!?¿]*\?", " . ", rest)
    for sentence in _CLAUSE_END.split(rest):
        for piece in sentence.split(","):
            if _REQUEST.match(_words(piece)):
                found.append(sentence[sentence.index(piece):])
                break
    return [c for c in found
            if _words(c) and not _TAG.match(_words(c))
            and not _NOT_ASKING.search(_words(c))]


def is_asking(text: str) -> bool:
    """Tell whether an interviewer's line asks something."""
    return bool(question_clauses(text))


_FILLER = frozenset(
    "tipo tipos forma formas manera cosas cosa nivel sobre entre desde "
    "hasta para segun algun alguna alguno todos todas cada otro otra "
    "otros otras esta este estos estas bien mucho mucha algo todo "
    "with from that this what which your have".split()
)
"""Words of a concept that do not identify it."""
_GENERIC = frozenset(
    "traba usar hacer tener saber donde cuant tiemp como cuand quien "
    "exper puest empre clien perso equip proye cosas vez veces anos "
    "dias semana work years time".split()
)
"""Stems shared with common verbs and nouns of any interview («trabajar»,
«experiencia», «tiempo»): alone they never identify a concept."""


def asked_stems(point: str) -> List[str]:
    """The stems a question must name to ask about ``point``.

    Words of four letters or more (question words included: «dónde»,
    «cuánto» help identify «Dónde trabajó», «Cuánto tiempo»), filler
    left out, first five letters.
    """
    words = re.findall(r"[a-z0-9+#]{4,}", fold(point))
    return list(dict.fromkeys(w[:5] for w in words if w not in _FILLER))


def point_score(point: str, clause: str) -> float:
    """How well a question clause names ``point`` (0: it does not).

    All the concept's stems (one or two), or at least half of them and
    two or more (three or more stems), and at least one that is not
    generic unless every stem is named; the share of stems named.
    """
    stems = asked_stems(point)
    if not stems:
        return 0.0
    heard = re.findall(r"[a-z0-9+#]{3,}", fold(clause))
    found = [s for s in stems if any(w.startswith(s) for w in heard)]
    if len(found) == len(stems):
        if len(stems) == 1 and stems[0] in _GENERIC:
            return 0.0
        return 1.0
    if len(stems) > 2 and len(found) >= max(2, math.ceil(len(stems) / 2)) \
            and any(s not in _GENERIC for s in found):
        return len(found) / len(stems)
    return 0.0


def asks_point(point: str, text: str) -> bool:
    """Tell whether an interviewer's line asks about the concept ``point``.

    The concept is named inside a question clause
    (:func:`question_clauses`) by its distinctive words
    (:func:`point_score`). A paraphrase is left to the model (live tick
    call, after-session attribution) or the interviewer's own mark.
    """
    return any(point_score(point, c) > 0 for c in question_clauses(text))


@dataclass(frozen=True)
class AskedMatch:
    """The concepts one interviewer line asks, in one topic.

    ``clear``: the topic was the only best match (it may move the topic
    being asked); a tie broken in favour of the topic being asked is
    not clear.
    """

    topic: str
    points: Tuple[str, ...]
    clear: bool


def asked_concepts(
    text: str, topics: Sequence["TopicRef"], current: Optional[str] = None,
) -> Optional[AskedMatch]:
    """The concepts a line asks, in the topic it matches best (#286).

    Each concept named in a question clause gets a score
    (:func:`point_score`); the topic with the best concept wins. A tie
    between topics goes to the topic being asked (``current``) when it
    is one of them, else nothing is marked (ambiguous: plans repeat
    concepts across topics).
    """
    clauses = question_clauses(text)
    if not clauses:
        return None
    best: Dict[str, float] = {}
    hits: Dict[str, List[str]] = {}
    for topic in topics:
        for pid, ptext in topic.points:
            score = max(point_score(ptext, c) for c in clauses)
            if score > 0:
                hits.setdefault(topic.id, []).append(pid)
                best[topic.id] = max(best.get(topic.id, 0.0), score)
    if not best:
        return None
    top = max(best.values())
    leaders = [t for t, v in best.items() if v == top]
    if len(leaders) == 1:
        return AskedMatch(leaders[0], tuple(hits[leaders[0]]), True)
    if current in leaders:
        return AskedMatch(str(current), tuple(hits[str(current)]), False)
    return None


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
    answered: Set[str] = field(default_factory=set)
    """Concepts the candidate answered after their question was asked
    (the «again» rule)."""


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
                    and times <= 1 and pid in facts.get(
                        topic.id, TopicFacts()).answered:
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
            for mark in [point.get("asked"), *point.get("undone", [])]:
                if mark:
                    mark.pop("quote", None)
        found.append(item)
    return found


def settled(
    slots: Sequence[Slot],
    facts: Mapping[str, TopicFacts],
    reached: Mapping[str, CoverageLevel],
    final: Sequence[SheetSlot],
    texts: Mapping[str, str],
    model_asked: Mapping[str, Mapping[str, PointAsked]],
    earlier: Optional[Mapping[str, TopicCoverage]] = None,
) -> List[TopicCoverage]:
    """The coverage the stored report keeps, after the session (#286).

    From the session's facts: each concept's status becomes the final
    one (``final``: the slots after the model's attribution, or the
    rules'); a concept not asked during the session takes the mark of an
    earlier report or hand-off of the same round (``earlier``: a rebuild
    or ``rejudge`` replays only the words, so marks by hand or by the
    live model would be lost), else the model's (``model_asked``,
    ``after_session``) unless the interviewer undid a mark on that same
    line. Answers and words count the final answers too. The level is
    recomputed and never lower than the one the session (or the earlier
    report) reached.
    """
    by_id = {s.id: s for s in final}
    before = earlier or {}
    found: List[TopicCoverage] = []
    for slot in slots:
        got = facts.get(slot.id, TopicFacts())
        done = by_id.get(slot.id)
        old_topic = before.get(slot.id)
        old_points = {p.id: p for p in (old_topic.points
                                        if old_topic else [])}
        status = {p.id: p.status for p in (done.points if done else [])}
        said = [a.utterance for a in (done.answers if done else [])]
        words = sum(len(texts.get(u, "").split()) for u in set(said))
        points = []
        for p in slot.points:
            now = next((x for x in got.points if x.id == p.id),
                       PointCoverage(id=p.id))
            then = old_points.get(p.id, PointCoverage(id=p.id))
            undone = [*now.undone, *(u for u in then.undone
                                     if u not in now.undone)]
            gone = {u.utterance for u in undone if u.utterance}
            model = model_asked.get(slot.id, {}).get(p.id)
            if model is not None and model.utterance in gone:
                model = None
            mark = now.asked or then.asked or model
            points.append(PointCoverage(
                id=p.id, text=p.text, status=status.get(p.id, now.status),
                asked=mark, undone=undone))
        floor = higher(reached.get(slot.id, "none"),
                       old_topic.level if old_topic else "none")
        merged = TopicFacts(
            named=got.named, asked=got.asked or any(
                x.asked for x in points),
            answers=max(got.answers, len(set(said))),
            words=max(got.words, words, old_topic.words if old_topic
                      else 0),
            complete=got.complete or (done is not None
                                      and done.status == "filled"),
            points=points)
        found.append(topic_coverage(slot.id, merged, floor, slot.label))
    return found


def earlier_coverage(sheet: Mapping[str, object]
                     ) -> Dict[str, TopicCoverage]:
    """A stored report's or hand-off's coverage by topic id (#286)."""
    found: Dict[str, TopicCoverage] = {}
    raw = sheet.get("coverage")
    for item in raw if isinstance(raw, list) else []:
        try:
            topic = TopicCoverage.model_validate(item)
        except ValueError:
            continue
        found.setdefault(topic.id, topic)
    return found
