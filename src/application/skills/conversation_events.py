"""Use cases of the second layer: analyse a session, evaluate the rules.

:func:`analyse` links the replies of one :class:`ActSession`, derives
its events (``docs/skills/CONVERSATION_EVENTS.md``) and counts them per
person; :func:`evaluate_events` measures the rules against a
hand-written expectation, once with the links of the source and once
with every link removed and inferred again. :func:`render_events` and
:func:`person_context` turn the result into plain text for the CLI and
the person report. No model is called.
"""

from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from src.domain.errors import InvalidInputError
from src.domain.skills.act_sequence import ActSession, link_map, without_links
from src.domain.skills.conversation_events import (
    ConversationEvent,
    EventRules,
    ReplyLink,
    derive_events,
    link_replies,
)
from src.domain.skills.conversation_summary import (
    PersonEvents,
    SessionEvents,
    summarise,
)
from src.domain.skills.event_evaluation import (
    ExpectedEvents,
    LinkScore,
    TypeScore,
    score_links,
    score_types,
)


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ConversationAnalysis(_Frozen):
    """The second layer of one session: links, events and counts."""

    session_id: str
    synthetic: bool
    relinked: bool
    rules: EventRules
    links: List[ReplyLink]
    events: List[ConversationEvent]
    summary: SessionEvents


class EventEvaluation(_Frozen):
    """Precision and recall of the rules against an expectation.

    ``given_links`` uses the links of the source; ``inferred_links``
    removes them and infers every link again; ``links`` compares those
    inferred links with the source's.
    """

    note: str
    dialogues: List[str]
    rules: EventRules
    given_links: List[TypeScore]
    inferred_links: List[TypeScore]
    links: LinkScore


def analyse(
    session: ActSession, rules: EventRules, relink: bool = False
) -> ConversationAnalysis:
    """Return the links, events and counts of ``session``.

    With ``relink`` the source's reply links are ignored and every link
    is inferred.
    """
    source = without_links(session) if relink else session
    links = link_replies(source, rules)
    events = derive_events(source, rules, links)
    return ConversationAnalysis(
        session_id=session.id, synthetic=session.synthetic, relinked=relink,
        rules=rules, links=links, events=events,
        summary=summarise(source, links, events),
    )


def pick_session(sessions: List[ActSession], session_id: str) -> ActSession:
    """Return the session ``session_id``, or the only one there is."""
    if not session_id:
        if len(sessions) == 1:
            return sessions[0]
        raise InvalidInputError(
            f"The file holds {len(sessions)} sessions; choose one with "
            "--session (" + ", ".join(s.id for s in sessions[:6])
            + ("…" if len(sessions) > 6 else "") + ").")
    for session in sessions:
        if session.id == session_id:
            return session
    raise InvalidInputError(f"No session {session_id!r} in the file.")


def evaluate_events(
    sessions: List[ActSession], expected: ExpectedEvents, rules: EventRules
) -> EventEvaluation:
    """Score the rules on the dialogues of ``expected``.

    Raises :class:`~src.domain.errors.InvalidInputError` when one of
    them is not among ``sessions``.
    """
    by_id = {s.id: s for s in sessions}
    missing = sorted(set(expected.dialogues) - set(by_id))
    if missing:
        raise InvalidInputError(
            f"Expected events for unknown dialogue(s): {', '.join(missing)}."
        )
    given: Dict[str, List[ConversationEvent]] = {}
    inferred: Dict[str, List[ConversationEvent]] = {}
    known: Dict[str, str] = {}
    relinked: List[ReplyLink] = []
    for dialogue in expected.dialogues:
        session = by_id[dialogue]
        given[dialogue] = analyse(session, rules).events
        again = analyse(session, rules, relink=True)
        inferred[dialogue] = again.events
        known.update({f"{dialogue}/{s}": f"{dialogue}/{t}"
                      for s, t in link_map(session).items()})
        relinked += [
            link.model_copy(update={
                "source": f"{dialogue}/{link.source}",
                "target": f"{dialogue}/{link.target}",
            })
            for link in again.links
        ]
    return EventEvaluation(
        note=expected.note, dialogues=list(expected.dialogues), rules=rules,
        given_links=score_types(expected.dialogues, given),
        inferred_links=score_types(expected.dialogues, inferred),
        links=score_links(known, relinked),
    )


def _fmt(value: Optional[float]) -> str:
    """Format an optional rate."""
    return "n/a" if value is None else f"{value:.2f}"


def _person_line(p: PersonEvents) -> str:
    """Return one speaker's counts on one line."""
    reply = (
        "" if p.median_reply_turns is None
        else f"  replies after {p.median_reply_turns:g} turn(s)"
    )
    return (
        f"  {p.speaker:<16} questions {p.questions_answered}/"
        f"{p.questions_asked} answered  answered others {p.answers_given}  "
        f"proposals {p.proposals_made} (accepted {p.proposals_accepted}, "
        f"rejected {p.proposals_rejected}, contested "
        f"{p.proposals_contested}, ignored {p.proposals_ignored})  "
        f"commitments {p.commitments}{reply}"
    )


def _event_line(event: ConversationEvent) -> str:
    """Return one event on one line."""
    who = event.by
    if event.type == "action_item":
        who = f"{event.owner}" + (
            "" if event.owner == event.by else f" (asked by {event.by})")
    elif event.responders:
        who += " → " + ", ".join(event.responders)
    extra = f"  deadline: {event.deadline}" if event.deadline else ""
    text = event.what or event.text
    return (f"  [{event.key}] {event.type:<19} {who}: {text}{extra}  "
            f"(confidence {event.confidence:.2f})")


def render_events(analysis: ConversationAnalysis) -> List[str]:
    """Return the events and counts of a session as text lines."""
    summary = analysis.summary
    counts = {k: v for k, v in summary.counts.items() if v}
    mode = (
        "every link inferred" if analysis.relinked
        else "given links kept, missing ones inferred"
    )
    found = ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"
    lines = [
        f"Conversation events ({mode}; {summary.links_given} given, "
        f"{summary.links_inferred} inferred links): {found}",
        "Counts, not ratings; derived from act labels by fixed rules.",
    ]
    for kind in ("decision", "action_item", "open_issue"):
        chosen = [e for e in analysis.events if e.type == kind]
        if chosen:
            lines.append(f"{kind.replace('_', ' ').capitalize()}s:")
            lines += [_event_line(e) for e in chosen]
    others = [
        e for e in analysis.events
        if e.type not in ("decision", "action_item", "open_issue")
    ]
    if others:
        lines.append("Questions and proposals:")
        others.sort(key=lambda e: e.order)
        lines += [_event_line(e) for e in others]
    lines.append("Per person:")
    lines += [_person_line(p) for p in summary.per_person]
    if summary.ignored_by:
        lines.append("Proposals ignored, by whom they came from: " + ", ".join(
            f"{p} {n}" for p, n in summary.ignored_by.items()
        ))
    return lines


CONTEXT_TEXTS: Mapping[str, Mapping[str, str]] = {
    "es": {
        "questions": "Hizo {asked} pregunta(s); {answered} recibieron "
                     "respuesta.",
        "answers": "Respondió a {n} pregunta(s) de otras personas.",
        "proposals": "Hizo {made} propuesta(s): {accepted} aceptada(s), "
                     "{rejected} rechazada(s), {contested} discutida(s), "
                     "{ignored} sin respuesta.",
        "commitments": "Asumió {n} tarea(s).",
        "note": "Recuentos derivados de las etiquetas de actos de diálogo; "
                "contexto, no una valoración.",
    },
    "en": {
        "questions": "Asked {asked} question(s); {answered} got an answer.",
        "answers": "Answered {n} question(s) from other people.",
        "proposals": "Made {made} proposal(s): {accepted} accepted, "
                     "{rejected} rejected, {contested} contested, "
                     "{ignored} with no reply.",
        "commitments": "Took on {n} task(s).",
        "note": "Counts derived from the dialogue-act labels; context, "
                "not a rating.",
    },
}
"""Wording of the person-report context lines, per language."""


def person_context(
    summary: SessionEvents, participant: str, language: str
) -> List[str]:
    """Return the context sentences about ``participant`` for the report.

    Empty when the participant does not appear in the session; the last
    line says these are counts, not ratings.
    """
    words = CONTEXT_TEXTS.get(language, CONTEXT_TEXTS["en"])
    try:
        p = summary.of(participant)
    except KeyError:
        return []
    lines = []
    if p.questions_asked:
        lines.append(words["questions"].format(
            asked=p.questions_asked, answered=p.questions_answered))
    if p.answers_given:
        lines.append(words["answers"].format(n=p.answers_given))
    if p.proposals_made:
        lines.append(words["proposals"].format(
            made=p.proposals_made, accepted=p.proposals_accepted,
            rejected=p.proposals_rejected,
            contested=p.proposals_contested, ignored=p.proposals_ignored))
    if p.commitments:
        lines.append(words["commitments"].format(n=p.commitments))
    return lines + [words["note"]] if lines else []
