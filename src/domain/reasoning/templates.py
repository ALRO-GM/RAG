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
    ReasonRequest,
    Reply,
    Role,
    Suggestion,
)

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

_REGISTRY: Dict[str, Template] = {CONVERSATIONAL_V1.id: CONVERSATIONAL_V1}


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
