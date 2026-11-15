"""Second layer: conversation events derived from sequences of acts.

No single sentence is a decision or an unanswered question: these are
read from the first-layer labels of several sentences and the reply
links between them (``docs/skills/CONVERSATION.md`` §2 and §5.1, layer
2). The rules, their parameters and Spanish examples are in
``docs/skills/CONVERSATION_EVENTS.md``; in short:

1. **Reply linking.** A respuesta, aceptación, rechazo or compromiso
   with no link is attached to the most plausible earlier sentence of
   another speaker within ``reply_window`` turns: each candidate scores
   ``prior(type) × (1 - link_decay × (distance - 1))`` and the best one
   above ``min_link_confidence`` wins; the score is the link's
   confidence. Given links are kept with confidence 1.
2. **Events** per first-pair part: a pregunta is answered or
   unanswered; a propuesta is accepted, rejected, contested (standing
   acceptance and rejection) or ignored (no acceptance nor rejection);
   an accepted petición or any compromiso is an action item (owner,
   sentences, deadline expression); an accepted proposal or a
   gestión closing is a decision; an open issue is an unanswered
   question, a petición nobody took up, or an ignored or contested
   proposal with no decision after it.

Every event lists the sentences that support it, the people involved,
its turn and time, and a confidence: the smallest of the weights of the
labels it rests on and the confidences of the links it uses. It is a
heuristic score, not a calibrated probability. All of it is counting
over labels: no model is called and nothing here rates anybody.
"""

import re
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.act_sequence import ActSentence, ActSession

EventType = Literal[
    "question_answered", "question_unanswered", "proposal_accepted",
    "proposal_rejected", "proposal_contested", "proposal_ignored",
    "decision", "action_item", "open_issue",
]

EVENT_TYPES: Tuple[EventType, ...] = (
    "question_answered", "question_unanswered", "proposal_accepted",
    "proposal_rejected", "proposal_contested", "proposal_ignored",
    "decision", "action_item", "open_issue",
)
"""Every event type, in the order reports list them."""

ACCEPTING: Tuple[str, ...] = ("aceptacion", "compromiso")
"""Labels that take up a proposal or a request."""

REPLYING: Tuple[str, ...] = ("respuesta", "aceptacion", "rechazo",
                             "compromiso")
"""Labels whose sentence is linked to an antecedent when it has none."""

NOT_SUBSTANTIVE: Tuple[str, ...] = ("escucha", "otro", "relacion")
"""Labels a reply is never attached to."""

LINK_PRIORS: Dict[str, Dict[str, float]] = {
    "respuesta": {"pregunta": 1.0, "gestion": 0.5},
    "aceptacion": {"propuesta": 1.0, "peticion": 1.0, "closing": 1.0,
                   "opinion": 0.6, "informacion": 0.5, "gestion": 0.5,
                   "pregunta": 0.3},
    "rechazo": {"propuesta": 1.0, "peticion": 1.0, "closing": 1.0,
                "opinion": 0.6, "informacion": 0.6, "gestion": 0.4,
                "pregunta": 0.3},
    "compromiso": {"peticion": 1.0, "pregunta": 0.6},
}
"""How plausible each kind of antecedent is for each reply label.

``closing`` is a gestión sentence that states an outcome
(:data:`CLOSING`). A compromiso is only linked to a petición or a
pregunta nobody else has taken up yet.
"""

CLOSING = re.compile(
    r"\b(resumiendo|en resumen|queda|quedamos|acordamos|acordado|"
    r"decidido|cerrado|lo dicho)\b",
    re.IGNORECASE,
)
"""Words with which a gestión sentence states what was decided."""

_DAYS = r"(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
_MONTHS = (r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
           r"septiembre|setiembre|octubre|noviembre|diciembre)")
_WHEN = (r"(\d+|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
         r"once|doce|mediod[ií]a|fin de \w+|final de \w+|"
         r"pr[oó]xim[oa] \w+|reuni[oó]n)")
DEADLINE = re.compile(
    r"\b("
    r"(antes de(l| las?)?|para (el|la|las)|hasta (el|la|las)|a (las?)) "
    r"(" + _DAYS + "|" + _MONTHS + "|" + _WHEN + r")"
    r"( de " + _MONTHS + r"| y media| por la (mañana|tarde))?"
    r"|pasado mañana|mañana( por la (mañana|tarde|noche)| a primera hora)?"
    r"|hoy( mismo)?|esta (tarde|mañana|noche|semana)|este (mes|año)"
    r"|(el|la) (semana|mes) que viene|la próxima semana"
    r"|(el|este|esta|el próximo) " + _DAYS + r"( por la (mañana|tarde))?"
    r"|(el )?(fin|final) de (mes|semana|año)"
    r"|en (dos|tres|cuatro|cinco|diez|quince|\d+) (días|semanas|meses)"
    r"|(en|el \d+ de|a principios de|a finales de) " + _MONTHS +
    r"|ahora mismo|enseguida|cuanto antes"
    r")\b",
    re.IGNORECASE,
)
"""Spanish time expressions read as an action item's deadline."""


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EventRules(_Frozen):
    """Parameters of the second layer (mirrored in ``SkillsSettings``).

    - ``reply_window``: how many speaker turns back a missing reply link
      is searched for, and how many turns a pregunta or propuesta waits
      for a reply before an unanswered/ignored event is fully confident.
    - ``link_decay``: confidence lost per extra turn of distance.
    - ``min_link_confidence``: weakest link inferred at all.
    - ``min_label_weight``: a weighted label counts when it carries at
      least this % (so ``gestion:60 pregunta:40`` is also a question).
    - ``use_doubtful``: whether tagger labels marked doubtful count.
    - ``min_acceptances``: acceptances by other speakers a proposal
      needs to become a decision.
    - ``closing_acceptances``: acceptances that make a gestión sentence
      without closing words a decision (capped by the other speakers).
    - ``closing_window``: turns after an accepted proposal within which
      a gestión closing absorbs it into one decision.
    """

    reply_window: int = Field(default=4, ge=1)
    link_decay: float = Field(default=0.15, ge=0.0, le=1.0)
    min_link_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    min_label_weight: int = Field(default=30, ge=1, le=100)
    use_doubtful: bool = False
    min_acceptances: int = Field(default=1, ge=1)
    closing_acceptances: int = Field(default=2, ge=1)
    closing_window: int = Field(default=12, ge=0)


class ReplyLink(_Frozen):
    """A sentence replying to an earlier one of another speaker."""

    source: str
    target: str
    inferred: bool
    confidence: float = Field(ge=0.0, le=1.0)
    turns: int = Field(ge=0)


class ConversationEvent(_Frozen):
    """One derived event and what supports it.

    ``key`` is the sentence that defines it (the question, the
    proposal, the petición or compromiso, the gestión closing);
    ``sentences`` starts with it and adds the replies it rests on.
    ``by`` started it; ``responders`` replied; ``owner``, ``what`` and
    ``deadline`` describe an action item; ``latency_turns`` is how many
    turns the first reply took; ``source`` is the event an open issue
    restates.
    """

    id: str
    type: EventType
    key: str
    sentences: List[str]
    by: str
    responders: List[str] = []
    people: List[str]
    turn: int
    order: int
    start: Optional[float] = None
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    latency_turns: Optional[int] = None
    owner: Optional[str] = None
    what: Optional[str] = None
    deadline: Optional[str] = None
    source: Optional[str] = None


class _View:
    """Lookups over a session under a set of rules."""

    def __init__(self, session: ActSession, rules: EventRules) -> None:
        """Index the sentences of ``session``."""
        self.session = session
        self.rules = rules
        self.by_id = {s.id: s for s in session.sentences}
        self.last_turn = max((s.turn for s in session.sentences), default=0)

    def has(self, sentence: ActSentence, *labels: str) -> bool:
        """Return whether ``sentence`` carries one of ``labels``."""
        return self.weight(sentence, *labels) > 0

    def weight(self, sentence: ActSentence, *labels: str) -> int:
        """Return the largest counted weight among ``labels`` (0 if none).
        """
        best = 0
        for item in sentence.labels:
            if item.label not in labels:
                continue
            if item.doubtful and not self.rules.use_doubtful:
                continue
            if item.weight >= self.rules.min_label_weight:
                best = max(best, item.weight)
        return best

    def is_closing(self, sentence: ActSentence) -> bool:
        """Return whether a gestión sentence states an outcome."""
        return (self.has(sentence, "gestion")
                and not self.has(sentence, "pregunta")
                and CLOSING.search(sentence.text) is not None)

    def kinds(self, sentence: ActSentence) -> List[str]:
        """Return the antecedent kinds ``sentence`` can play."""
        kinds = [item.label for item in sentence.labels
                 if self.has(sentence, item.label)]
        if self.is_closing(sentence):
            kinds.append("closing")
        return kinds


def deadline(texts: Iterable[str]) -> Optional[str]:
    """Return the first time expression found in ``texts``, if any."""
    for text in texts:
        found = DEADLINE.search(text)
        if found:
            return found.group(0).strip()
    return None


def _reply_label(view: _View, sentence: ActSentence) -> Optional[str]:
    """Return the heaviest reply label of ``sentence``, if any."""
    best: Optional[str] = None
    weight = 0
    for label in REPLYING:
        current = view.weight(sentence, label)
        if current > weight:
            best, weight = label, current
    return best


def link_replies(session: ActSession, rules: EventRules) -> List[ReplyLink]:
    """Return the reply links of ``session``: given, then inferred.

    A given link (``responds_to``) is kept as it is. A sentence with a
    reply label and no link gets the best-scoring antecedent (module
    docstring); a compromiso only takes a petición or pregunta that no
    other speaker has taken up yet.
    """
    view = _View(session, rules)
    links: List[ReplyLink] = []
    taken: Dict[str, Set[str]] = {}
    for sentence in session.sentences:
        if sentence.responds_to is not None:
            target = view.by_id.get(sentence.responds_to)
            if target is None or target.speaker == sentence.speaker:
                continue
            links.append(ReplyLink(
                source=sentence.id, target=target.id, inferred=False,
                confidence=1.0, turns=max(0, sentence.turn - target.turn),
            ))
            taken.setdefault(target.id, set()).add(sentence.speaker)
            continue
        label = _reply_label(view, sentence)
        if label is None:
            continue
        found = _antecedent(view, sentence, label, taken)
        if found is None:
            continue
        target, score = found
        links.append(ReplyLink(
            source=sentence.id, target=target.id, inferred=True,
            confidence=round(score, 3), turns=sentence.turn - target.turn,
        ))
        taken.setdefault(target.id, set()).add(sentence.speaker)
    return links


def _antecedent(
    view: _View,
    sentence: ActSentence,
    label: str,
    taken: Dict[str, Set[str]],
) -> Optional[Tuple[ActSentence, float]]:
    """Return the best earlier sentence for ``sentence`` and its score."""
    rules = view.rules
    priors = LINK_PRIORS[label]
    best: Optional[Tuple[ActSentence, float]] = None
    for candidate in view.session.sentences[:sentence.order]:
        distance = sentence.turn - candidate.turn
        if candidate.speaker == sentence.speaker or distance < 1:
            continue
        if distance > rules.reply_window:
            continue
        if label == "compromiso" and taken.get(candidate.id, set()) - {
            sentence.speaker
        }:
            continue
        kinds = [k for k in view.kinds(candidate) if k not in NOT_SUBSTANTIVE]
        prior = max((priors.get(k, 0.0) for k in kinds), default=0.0)
        score = prior * max(0.0, 1.0 - rules.link_decay * (distance - 1))
        if score < rules.min_link_confidence or score <= 0:
            continue
        if best is None or score >= best[1]:
            best = (candidate, score)
    return best


class _Builder:
    """Derive the events of one session."""

    def __init__(
        self, session: ActSession, rules: EventRules, links: List[ReplyLink]
    ) -> None:
        """Index the links by antecedent."""
        self.view = _View(session, rules)
        self.rules = rules
        self.links = {link.source: link for link in links}
        self.replies: Dict[str, List[ReplyLink]] = {}
        for link in links:
            self.replies.setdefault(link.target, []).append(link)
        self.events: List[ConversationEvent] = []

    def sentence(self, sentence_id: str) -> ActSentence:
        """Return a sentence by id."""
        return self.view.by_id[sentence_id]

    def add(
        self,
        kind: EventType,
        key: ActSentence,
        sentences: List[str],
        confidence: float,
        responders: Optional[List[str]] = None,
        latency_turns: Optional[int] = None,
        owner: Optional[str] = None,
        what: Optional[str] = None,
        deadline: Optional[str] = None,
        source: Optional[str] = None,
    ) -> ConversationEvent:
        """Record an event defined by ``key``."""
        ordered = [key.id] + sorted(
            {s for s in sentences if s != key.id},
            key=lambda s: self.sentence(s).order,
        )
        people = list(dict.fromkeys(
            [key.speaker] + [self.sentence(s).speaker for s in ordered]
        ))
        event = ConversationEvent(
            id=f"e{len(self.events) + 1}", type=kind, key=key.id,
            sentences=ordered, by=key.speaker,
            responders=list(responders or []), people=people,
            turn=key.turn, order=key.order, start=key.start, text=key.text,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            latency_turns=latency_turns, owner=owner, what=what,
            deadline=deadline, source=source,
        )
        self.events.append(event)
        return event

    def stances(self, target: ActSentence) -> Tuple[
        Dict[str, ReplyLink], Dict[str, ReplyLink]
    ]:
        """Return the standing acceptances and rejections, per speaker.

        A speaker's last stance on ``target`` is the one that stands.
        """
        last: Dict[str, Tuple[str, ReplyLink]] = {}
        for link in self.replies.get(target.id, []):
            reply = self.sentence(link.source)
            accept = self.view.weight(reply, *ACCEPTING)
            reject = self.view.weight(reply, "rechazo")
            if accept == reject == 0:
                continue
            last[reply.speaker] = (
                "accept" if accept >= reject else "reject", link
            )
        accepted = {p: link for p, (s, link) in last.items() if s == "accept"}
        rejected = {p: link for p, (s, link) in last.items() if s == "reject"}
        return accepted, rejected

    def waited(self, key: ActSentence) -> float:
        """Return how much of the reply window passed after ``key``."""
        after = self.view.last_turn - key.turn
        return min(1.0, (after + 1) / (self.rules.reply_window + 1))

    def label_confidence(self, key: ActSentence, *labels: str) -> float:
        """Return the weight of the defining label as a fraction."""
        return self.view.weight(key, *labels) / 100

    def questions(self) -> None:
        """Answered and unanswered questions."""
        for key in self.view.session.sentences:
            if not self.view.has(key, "pregunta"):
                continue
            base = self.label_confidence(key, "pregunta")
            links = self.replies.get(key.id, [])
            if links:
                first = min(links, key=lambda k: self.sentence(k.source).order)
                self.add(
                    "question_answered", key, [k.source for k in links],
                    min([base] + [k.confidence for k in links]),
                    responders=list(dict.fromkeys(
                        self.sentence(k.source).speaker for k in links)),
                    latency_turns=first.turns,
                )
            else:
                self.add("question_unanswered", key, [key.id],
                         base * self.waited(key))

    def proposals(self) -> List[Tuple[ConversationEvent, ActSentence]]:
        """Fate of every proposal; return the accepted ones."""
        accepted_events: List[Tuple[ConversationEvent, ActSentence]] = []
        for key in self.view.session.sentences:
            if not self.view.has(key, "propuesta"):
                continue
            base = self.label_confidence(key, "propuesta")
            accepted, rejected = self.stances(key)
            used = list(accepted.values()) + list(rejected.values())
            confidence = min([base] + [k.confidence for k in used])
            kind: EventType
            if accepted and rejected:
                kind = "proposal_contested"
            elif accepted:
                kind = "proposal_accepted"
            elif rejected:
                kind = "proposal_rejected"
            else:
                kind = "proposal_ignored"
                confidence = base * self.waited(key)
            first = min(
                used, key=lambda k: self.sentence(k.source).order,
            ) if used else None
            event = self.add(
                kind, key, [k.source for k in used], confidence,
                responders=list(dict.fromkeys(
                    self.sentence(k.source).speaker for k in sorted(
                        used, key=lambda k: self.sentence(k.source).order))),
                latency_turns=first.turns if first else None,
            )
            if kind == "proposal_accepted" and (
                len(accepted) >= self.rules.min_acceptances
            ):
                accepted_events.append((event, key))
        return accepted_events

    def decisions(
        self, accepted: List[Tuple[ConversationEvent, ActSentence]]
    ) -> None:
        """Decisions: gestión closings, then accepted proposals left over.
        """
        others = max(1, len(self.view.session.speakers()) - 1)
        needed = min(self.rules.closing_acceptances, others)
        absorbed: Set[str] = set()
        for key in self.view.session.sentences:
            if not self.view.has(key, "gestion") or self.view.has(
                key, "pregunta"
            ):
                continue
            agree, rejected = self.stances(key)
            if rejected:
                continue
            if not (self.view.is_closing(key) or len(agree) >= needed):
                continue
            support = [k.source for k in agree.values()]
            for event, proposal in accepted:
                if proposal.id in absorbed:
                    continue
                gap = key.turn - proposal.turn
                if (proposal.order < key.order
                        and gap <= self.rules.closing_window):
                    absorbed.add(proposal.id)
                    support += event.sentences
            self.add(
                "decision", key, support,
                min([self.label_confidence(key, "gestion")]
                    + [k.confidence for k in agree.values()]),
                responders=list(agree),
            )
        for event, proposal in accepted:
            if proposal.id not in absorbed:
                self.add("decision", proposal, event.sentences,
                         event.confidence, responders=event.responders)

    def action_items(self) -> None:
        """Accepted requests and commitments, one item per owner."""
        used: Set[str] = set()
        for key in self.view.session.sentences:
            if not self.view.has(key, "peticion"):
                continue
            accepted, _ = self.stances(key)
            for owner, link in accepted.items():
                promises = [
                    k.source for k in self.replies.get(key.id, [])
                    if self.sentence(k.source).speaker == owner
                ]
                promises += [
                    s.id for s in self.view.session.sentences
                    if s.speaker == owner and s.order > key.order
                    and self.view.has(s, "compromiso")
                    and s.id not in used
                    and self.links.get(s.id) is None
                    and s.turn == self.sentence(link.source).turn
                ]
                used.update(promises)
                commitment = [
                    p for p in promises
                    if self.view.has(self.sentence(p), "compromiso")
                ]
                texts = [self.sentence(p).text for p in commitment] + [
                    self.sentence(p).text for p in promises] + [key.text]
                self.add(
                    "action_item", key, promises,
                    min(self.label_confidence(key, "peticion"),
                        link.confidence),
                    responders=[owner], owner=owner,
                    what=(self.sentence(commitment[0]).text if commitment
                          else key.text),
                    deadline=deadline(texts),
                )
        for key in self.view.session.sentences:
            if key.id in used or not self.view.has(key, "compromiso"):
                continue
            given = self.links.get(key.id)
            support = [key.id] + ([given.target] if given else [])
            confidence = self.label_confidence(key, "compromiso")
            if given:
                confidence = min(confidence, given.confidence)
            self.add(
                "action_item", key, support, confidence, owner=key.speaker,
                what=key.text, deadline=deadline([key.text]),
            )

    def requests_open(self) -> List[ActSentence]:
        """Return the peticiones nobody accepted nor rejected."""
        return [
            key for key in self.view.session.sentences
            if self.view.has(key, "peticion")
            and not any(self.stances(key))
        ]

    def open_issues(self) -> None:
        """What is unresolved at the end of the session."""
        decided = [e.order for e in self.events if e.type == "decision"]
        pending: List[Tuple[ActSentence, Optional[ConversationEvent]]] = []
        for event in list(self.events):
            stays = event.type == "question_unanswered"
            if event.type in ("proposal_ignored", "proposal_contested"):
                stays = not any(order > event.order for order in decided)
            if stays:
                pending.append((self.sentence(event.key), event))
        pending += [(key, None) for key in self.requests_open()]
        pending.sort(key=lambda item: item[0].order)
        for key, origin in pending:
            confidence = (origin.confidence if origin
                          else self.label_confidence(key, "peticion")
                          * self.waited(key))
            self.add("open_issue", key, [key.id], confidence,
                     source=origin.id if origin else None)


def derive_events(
    session: ActSession,
    rules: EventRules,
    links: Optional[List[ReplyLink]] = None,
) -> List[ConversationEvent]:
    """Return the conversation events of ``session``.

    ``links`` defaults to :func:`link_replies` of the session. Events
    come grouped by kind (questions, proposals, decisions, action
    items, open issues), each group in spoken order.
    """
    builder = _Builder(
        session, rules,
        link_replies(session, rules) if links is None else links,
    )
    builder.questions()
    builder.decisions(builder.proposals())
    builder.action_items()
    builder.open_issues()
    return builder.events
