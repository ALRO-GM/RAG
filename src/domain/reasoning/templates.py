"""Prompt templates of the reasoner, identified as ``name@version``.

A template builds the system and user prompts of a role from the
request and parses the model's reply into the response's fields. A
template never changes once released: a new wording is a new version.
Plain data and functions: no I/O.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.domain.errors import InvalidInputError
from src.domain.reasoning.roles import (
    MAX_SUGGESTION_CHARS,
    SIGNALS,
    ReasonRequest,
    Reply,
    Role,
    Suggestion,
)
from src.domain.reasoning.suggestion_grounding import ground

DEFAULT_MAX_OUTPUT_TOKENS = 120
"""Output cap of a live template: long outputs caused the latency tails."""


@dataclass(frozen=True)
class Parsed:
    """What a template read from the model's reply."""

    suggestions: Tuple[Suggestion, ...] = ()
    reply: Optional[Reply] = None


@dataclass(frozen=True)
class Template:
    """One versioned prompt of one role."""

    id: str
    role: Role
    system: Callable[[ReasonRequest], str]
    user: Callable[[ReasonRequest], str]
    parse: Callable[[str, ReasonRequest], Parsed]
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


def language_name(language: Optional[str]) -> str:
    """Return the language the reply is asked in (Spanish by default)."""
    return {"en": "English", "es": "Spanish"}.get(
        (language or "es")[:2].lower(), "the language of the conversation"
    )


def window_text(request: ReasonRequest) -> str:
    """Return the window as ``id speaker: text`` lines."""
    roles = request.context.speaker_roles
    return "\n".join(
        f"[{line.id}] {line.speaker}"
        + (f" ({roles[line.speaker]})" if line.speaker in roles else "")
        + f": {line.text}"
        for line in request.context.window
    )


def chunks_text(request: ReasonRequest) -> str:
    """Return the retrieved chunks as ``[id] text`` lines."""
    return "\n".join(f"[{r.id}] {r.text}" for r in request.retrieved)


def json_object(reply: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object of ``reply``, or ``None``."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(reply[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _conversational_system(request: ReasonRequest) -> str:
    return (
        "You are a voice assistant on a phone call. Answer the last "
        f"thing the user said in {language_name(request.language)}, in "
        "one or two short spoken sentences, using only the knowledge "
        "chunks given. If they do not cover it, say so and offer to find "
        "out. Reply with one JSON object: "
        '{"text": "...", "cites": ["chunk id", ...]}.'
    )


def _conversational_user(request: ReasonRequest) -> str:
    return (
        f"Knowledge:\n{chunks_text(request) or '(none)'}\n\n"
        f"Conversation:\n{window_text(request)}\n\nJSON:"
    )


def _conversational_parse(reply: str, request: ReasonRequest) -> Parsed:
    data = json_object(reply)
    text = str(data.get("text") or "").strip() if data else reply.strip()
    if not text:
        return Parsed()
    known = {r.id for r in request.retrieved}
    raw: Any = data.get("cites") if data else []
    cites: List[str] = [
        c for c in (raw if isinstance(raw, list) else [])
        if isinstance(c, str) and c in known
    ]
    return Parsed(reply=Reply(text=text, cites=cites))


CONVERSATIONAL_V1 = Template(
    id="conversational.voicebot@1", role="conversational",
    system=_conversational_system, user=_conversational_user,
    parse=_conversational_parse,
)

DEFAULT_MAX_SUGGESTIONS = 1

KINDS = ("question", "follow_up", "answer", "reminder", "caution")

SHAPE = (
    'Reply with one JSON object and nothing else: {"suggestions": '
    '[{"kind": "question|follow_up|answer|reminder|caution", '
    '"text": "...", "cites": ["chunk id"], "about": ["utterance id"]%s}]}. '
    "No other field, no explanation: the whole reply must stay under 100 "
    "tokens. Use only ids shown in brackets; if nothing is worth saying, "
    'reply {"suggestions": []}.'
)


def _text(value: object, limit: int = MAX_SUGGESTION_CHARS) -> str:
    """Return ``value`` as one trimmed line cut to ``limit`` characters."""
    return " ".join(str(value or "").split())[:limit]


def _strings(value: object) -> List[str]:
    """Return the strings of a JSON list (anything else is empty)."""
    return [v for v in value if isinstance(v, str)] if isinstance(
        value, list
    ) else []


def read_suggestions(
    reply: str, request: ReasonRequest, allow_signals: bool
) -> Parsed:
    """Read a suggester's reply into grounded suggestions.

    Only the contract's fields are read: anything else the model adds
    (a rating, a score, a label of the person) is dropped here, and
    ``signals`` are read only when ``allow_signals`` (support). Items
    of an unknown kind are skipped; those citing ids the request did
    not carry are dropped by :func:`ground`.
    """
    data = json_object(reply)
    items: Any = data.get("suggestions") if data else None
    found: List[Suggestion] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("kind") not in KINDS:
            continue
        text = _text(item.get("text"))
        if not text:
            continue
        signals = [
            s for s in _strings(item.get("signals")) if s in SIGNALS
        ] if allow_signals else []
        phase = item.get("script_phase")
        found.append(Suggestion.model_validate({
            "kind": item["kind"], "text": text,
            "rationale": _text(item.get("rationale")) or None,
            "cites": _strings(item.get("cites")),
            "about": _strings(item.get("about")),
            "signals": signals,
            "script_phase": phase if isinstance(phase, str) else None,
        }))
    return Parsed(suggestions=tuple(ground(found, request, allow_signals)))


def _suggester_user(request: ReasonRequest) -> str:
    ctx = request.context
    session = ctx.session
    head = ""
    if session and (session.type or session.goal):
        head = f"Conversation: {session.type or ''} {session.goal or ''}\n"
    phase = ""
    if ctx.script_phase or ctx.covered_phases:
        phase = (
            f"Current phase: {ctx.script_phase or '?'}; covered: "
            f"{', '.join(ctx.covered_phases) or 'none'}\n"
        )
    return (
        f"{head}{phase}Knowledge:\n{chunks_text(request) or '(none)'}\n\n"
        f"Recent conversation:\n{window_text(request)}\n\n"
        f"Focus: {ctx.focus or 'the last utterance'}\n\nJSON:"
    )


def _how_many(request: ReasonRequest) -> int:
    limit = request.constraints.max_suggestions
    return DEFAULT_MAX_SUGGESTIONS if limit is None else limit


def _interview_system(request: ReasonRequest) -> str:
    return (
        "You assist an interviewer during a live job interview. Suggest "
        f"at most {_how_many(request)} short things for the interviewer to "
        "ask next (a follow-up on what the candidate just said, or the "
        "next question of the script), in "
        f"{language_name(request.language)}, each under 20 words, grounded "
        "in the utterances and knowledge chunks given. Never evaluate, "
        "rate, score, rank or label the candidate, never describe their "
        "emotions, stress, confidence or personality, and add no field "
        "beyond those listed. " + SHAPE % ""
    )


def _interview_parse(reply: str, request: ReasonRequest) -> Parsed:
    return read_suggestions(reply, request, allow_signals=False)


def _support_system(request: ReasonRequest) -> str:
    return (
        "You assist a customer-support agent during a live call. Suggest "
        f"at most {_how_many(request)} short things for the agent to say "
        f"or ask next, in {language_name(request.language)}, each under "
        "30 words, grounded in the utterances and knowledge chunks given; "
        "do not invent policies, prices or deadlines. Also label what the "
        "CUSTOMER says in their words, never what they feel and never "
        "from the voice: signals from "
        f"{', '.join(SIGNALS)}, each backed by the customer's utterance "
        "ids in about (at most two signals). Add no field beyond those "
        "listed. "
        + SHAPE % ', "signals": ["complaint"]'
    )


def _support_parse(reply: str, request: ReasonRequest) -> Parsed:
    return read_suggestions(reply, request, allow_signals=True)


SUGGESTER_INTERVIEW_V1 = Template(
    id="suggester.interview@1", role="suggester",
    system=_interview_system, user=_suggester_user, parse=_interview_parse,
)
SUGGESTER_SUPPORT_V1 = Template(
    id="suggester.support@1", role="suggester",
    system=_support_system, user=_suggester_user, parse=_support_parse,
)

_REGISTRY: Dict[str, Template] = {
    t.id: t for t in (
        CONVERSATIONAL_V1, SUGGESTER_INTERVIEW_V1, SUGGESTER_SUPPORT_V1,
    )
}


def get_template(template_id: str, role: Role) -> Template:
    """Return the template ``template_id`` for ``role``.

    Raises :class:`~src.domain.errors.InvalidInputError` when it does
    not exist or belongs to another role.
    """
    template = _REGISTRY.get(template_id)
    if template is None or template.role != role:
        known = sorted(t for t, v in _REGISTRY.items() if v.role == role)
        raise InvalidInputError(
            f"Unknown {role} template {template_id!r}; known: "
            f"{', '.join(known)}."
        )
    return template
