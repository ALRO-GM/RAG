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
    AssistMode,
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
    data: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Template:
    """One versioned prompt of one role."""

    id: str
    role: Role
    system: Callable[[ReasonRequest], str]
    user: Callable[[ReasonRequest], str]
    parse: Callable[[str, ReasonRequest], Parsed]
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


SUPPORT_MAX_OUTPUT_TOKENS = 200
"""Cap of ``suggester.support@1``: its JSON (suggestion, ids and signals)
needs about 105 tokens and a longer Spanish sentence reached the 120 of
the interview template, which cut the reply (``OutputTooLongError``)."""


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


SIGNAL_DEFINITIONS = (
    "Definitions, each with a customer line that shows it: complaint = "
    "reports a concrete fault or a bad experience (\"The invoice has "
    "the wrong amount\"); stated_frustration = the customer says in "
    "words that they are annoyed, tired or fed up, whatever the topic "
    "(\"I'm sick of explaining this again\"); urgency = needs it "
    "solved by a time (\"I need it before 5 pm\"); churn_risk = says "
    "they may cancel or leave (\"I'll switch provider\"); confusion = "
    "does not understand a step, term or policy (\"What does pending "
    "mean?\"); satisfaction = thanks or says it worked (\"Perfect, "
    "thanks\"). A fault without annoyance words is only complaint; "
    "annoyance words are stated_frustration even beside a fault."
)
"""Short definitions and one example per signal of ``suggester.support@1``
(#196): without them the model never used ``stated_frustration``."""


def _support_system(request: ReasonRequest) -> str:
    return (
        "You assist a customer-support agent during a live call. Suggest "
        f"at most {_how_many(request)} short things for the agent to say "
        f"or ask next, in {language_name(request.language)}, each under "
        "25 words, grounded in the utterances and knowledge chunks given; "
        "do not invent policies, prices or deadlines. Also label what the "
        "CUSTOMER says in their words, never what they feel and never "
        "from the voice: signals from "
        f"{', '.join(SIGNALS)}, each backed by the customer's utterance "
        "ids in about (at most two signals). " + SIGNAL_DEFINITIONS
        + " Add no field beyond those listed, and no rationale. "
        + SHAPE % ', "signals": ["urgency"]'
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
    max_output_tokens=SUPPORT_MAX_OUTPUT_TOKENS,
)

# ---- @2: two assistance modes (LV-16) ----------------------------------

PINCELADA_WORDS = 8
"""Most words of a ``pincelada`` hint; longer ones are cut here."""
GUIDED_WORDS = 35
"""Asked upper bound of a ``guiado`` sentence (the contract's 300
characters still apply)."""
MODES_MAX_OUTPUT_TOKENS = 240
"""Cap of the ``@2`` templates: a ``guiado`` sentence is about 30
words more than a ``pincelada`` hint, on top of ``support@1``'s 200."""

MODE_RULES = (
    "The request's Mode line says how to phrase each text. pincelada: a "
    f"hint of at most {PINCELADA_WORDS} words saying where to steer "
    "(a topic or a move, not a full sentence, e.g. \"Ofrecer cambio de "
    "fecha de entrega\"). guiado: the exact sentence or question the "
    "%s can read aloud word for word, polite, in the second person to "
    f"the %s, under {GUIDED_WORDS} words."
)


def mode_of(request: ReasonRequest) -> AssistMode:
    """Return the request's assistance mode (``pincelada`` by default)."""
    return request.constraints.mode or "pincelada"


def _modes_user(request: ReasonRequest) -> str:
    """The ``@2`` user prompt: the ``@1`` one plus the mode and count.

    Both live here, not in the system prompt, so switching the mode in
    a session keeps the warm process (keyed by the system prompt).
    """
    mode = mode_of(request)
    how = ("a hint of at most 8 words" if mode == "pincelada"
           else "the exact sentence to say aloud")
    return (
        f"Mode: {mode} ({how}). At most {_how_many(request)} "
        f"suggestion(s).\n" + _suggester_user(request)
    )


def _short(parsed: Parsed, request: ReasonRequest) -> Parsed:
    """Cut ``pincelada`` texts to :data:`PINCELADA_WORDS` words."""
    if mode_of(request) != "pincelada":
        return parsed
    cut = []
    for suggestion in parsed.suggestions:
        words = suggestion.text.split()
        if len(words) > PINCELADA_WORDS:
            text = " ".join(words[:PINCELADA_WORDS]).rstrip(",;:") + "…"
            suggestion = suggestion.model_copy(update={"text": text})
        cut.append(suggestion)
    return Parsed(suggestions=tuple(cut), reply=parsed.reply)


def _interview_modes_system(request: ReasonRequest) -> str:
    return (
        "You assist an interviewer during a live job interview. Suggest "
        "what the interviewer could ask next (a follow-up on what the "
        "candidate just said, or the next question of the script), in "
        f"{language_name(request.language)}, grounded in the utterances "
        "and knowledge chunks given. "
        + MODE_RULES % ("interviewer", "candidate")
        + " Never evaluate, rate, score, rank or label the candidate, "
        "never describe their emotions, stress, confidence or "
        "personality, and add no field beyond those listed. " + SHAPE % ""
    )


def _interview_modes_parse(reply: str, request: ReasonRequest) -> Parsed:
    return _short(read_suggestions(reply, request, allow_signals=False),
                  request)


def _support_modes_system(request: ReasonRequest) -> str:
    return (
        "You assist a customer-support agent during a live call. Suggest "
        "what the agent could say or ask next, in "
        f"{language_name(request.language)}, grounded in the utterances "
        "and knowledge chunks given; do not invent policies, prices or "
        "deadlines. " + MODE_RULES % ("agent", "customer")
        + " Also label what the CUSTOMER says in their words, never what "
        "they feel and never from the voice: signals from "
        f"{', '.join(SIGNALS)}, each backed by the customer's utterance "
        "ids in about (at most two signals). " + SIGNAL_DEFINITIONS
        + " Add no field beyond those listed, and no rationale. "
        + SHAPE % ', "signals": ["urgency"]'
    )


def _support_modes_parse(reply: str, request: ReasonRequest) -> Parsed:
    return _short(read_suggestions(reply, request, allow_signals=True),
                  request)


SUGGESTER_INTERVIEW_V2 = Template(
    id="suggester.interview@2", role="suggester",
    system=_interview_modes_system, user=_modes_user,
    parse=_interview_modes_parse, max_output_tokens=MODES_MAX_OUTPUT_TOKENS,
)
SUGGESTER_SUPPORT_V2 = Template(
    id="suggester.support@2", role="suggester",
    system=_support_modes_system, user=_modes_user,
    parse=_support_modes_parse, max_output_tokens=MODES_MAX_OUTPUT_TOKENS,
)

# ---- the AI interviewer (#208) ------------------------------------------

INTERVIEWER_MAX_OUTPUT_TOKENS = 160
"""Cap of ``interviewer@1``: one JSON reply of at most 45 spoken words."""
PLANNER_MAX_OUTPUT_TOKENS = 2400
"""Cap of ``interview.planner@1``: up to 10 slots with their questions."""
SLOTS_MAX_OUTPUT_TOKENS = 160
"""Cap of ``interview.slots@1``: at most three slots with short quotes."""
OFFER_CHARS = 12000
"""Longest job offer the planner reads (the rest is cut)."""
ROLE_TYPES = (
    "tecnico", "comercio", "atencion_cliente", "hosteleria", "almacen",
    "oficina", "otro",
)
"""Kinds of job of an interview plan (tags ``puesto_*`` of the corpus)."""


def _payload_text(request: ReasonRequest, key: str, limit: int) -> str:
    """Return ``payload[key]`` as text cut to ``limit`` characters."""
    value = request.payload.get(key)
    return str(value)[:limit] if isinstance(value, (str, int, float)) else ""


def _payload_list(request: ReasonRequest, key: str) -> List[Any]:
    """Return ``payload[key]`` when it is a list (else empty)."""
    value = request.payload.get(key)
    return value if isinstance(value, list) else []


def _interviewer_system(request: ReasonRequest) -> str:
    return (
        "You are an AI interviewer holding a job interview by voice, in "
        f"{language_name(request.language)}. Each turn gives you the step "
        "the interview plan decided: ACTION (open, ask, probe, clarify, "
        "answer, reask, hint or close), the topic, question ideas and "
        "the evidence wanted. Say it in one or two short spoken "
        "sentences, at most 45 words, natural and polite. open: greet, "
        "say you are an AI "
        "interviewer, how many minutes it takes, and ask the first "
        "question. ask: a brief neutral acknowledgement, then one "
        "question on the topic. "
        "probe: ask for the concrete part that is missing (what they did "
        "themselves, a real example, the result), without repeating the "
        "whole question. answer: answer the candidate's question briefly "
        "from the job summary or the chunks (or say the team will answer "
        "it later), then ask the step's question. clarify: say "
        "the answer on this topic was not fully clear and ask whether "
        "they want to add something or go on to the next question "
        "(\"No me ha quedado del todo claro esto; ¿quieres matizarlo o "
        "pasamos a la siguiente pregunta?\"). close: thank them, say "
        "the team will contact them about next steps and say goodbye, "
        "with no question. reask: the candidate said nothing; say there is "
        "no hurry and ask the same question again in other, simpler "
        "words. hint: the candidate stopped in the middle of the answer "
        "(the answer so far is given); give one brief idea of how they "
        "could go on with this topic, then ask a choice question "
        "(\"¿Te referías a ..., o prefieres hablar de ...?\"), without "
        "judging what they said. Ask exactly one question per turn. "
        "Never give a score, a rating, feedback on how well they "
        "answered or whether they will pass; never praise or criticise "
        "an answer. Never ask "
        "about age, family, pregnancy, health, religion, politics, union "
        "membership, sexual orientation, origin or other protected "
        "topics; if the candidate brings one up, say it is not needed and "
        "go on. Reply with one JSON object: "
        '{"text": "...", "cites": ["chunk id", ...]}.'
    )


def _interviewer_user(request: ReasonRequest) -> str:
    ideas = [str(q)[:200] for q in _payload_list(request, "questions")[:3]]
    lines = [
        f"Job: {_payload_text(request, 'title', 120) or '?'}",
        f"Job summary: {_payload_text(request, 'summary', 400) or '?'}",
        f"Minutes: {_payload_text(request, 'minutes', 8) or '?'} in total, "
        f"{_payload_text(request, 'minutes_left', 8) or '?'} left",
        f"ACTION: {_payload_text(request, 'action', 12) or 'ask'}",
        f"Topic: {_payload_text(request, 'topic', 120) or '-'}",
        "Question ideas: " + (" | ".join(ideas) or "-"),
        f"Evidence wanted: {_payload_text(request, 'evidence', 240) or '-'}",
    ]
    missing = _payload_text(request, "missing", 240)
    if missing:
        lines.append(f"Missing: {missing}")
    so_far = _payload_text(request, "answer_so_far", 400)
    if so_far:
        lines.append(f"Answer so far: {so_far}")
    return (
        "\n".join(lines)
        + f"\n\nKnowledge:\n{chunks_text(request) or '(none)'}\n\n"
        f"Conversation:\n{window_text(request) or '(none yet)'}\n\nJSON:"
    )


INTERVIEWER_V1 = Template(
    id="interviewer@1", role="conversational",
    system=_interviewer_system, user=_interviewer_user,
    parse=_conversational_parse,
    max_output_tokens=INTERVIEWER_MAX_OUTPUT_TOKENS,
)


def _planner_system(request: ReasonRequest) -> str:
    return (
        "You prepare a structured job interview from a job offer. Write "
        f"in {language_name(request.language)}. Read the offer and the "
        "interview-method chunks and return one JSON object and nothing "
        'else: {"title": "job title", "role_type": "'
        + "|".join(ROLE_TYPES) + '", "summary": "what the job is, '
        'at most 300 characters", "slots": [{"label": "short topic", '
        '"kind": "requirement|competency", "competency": "competency id '
        'or null", "level": "minimo|deseable|opcional", "phase": '
        '"experience|technical|behavioural", "questions": ["one to three '
        'interview questions"], "evidence": "what a complete answer '
        'shows", "weight": 3}]}. weight is how much the slot matters for '
        "this job, from 1 (minor) to 5 (decisive). Make 6 to 10 slots: "
        "the offer's required "
        "experience, knowledge, certificates and conditions (shifts, "
        "availability) as requirement slots with level minimo; its "
        "nice-to-have items as deseable; the two to four competencies "
        "of the list this job needs most as competency slots with "
        "behavioural STAR questions (\"Cuéntame una vez en la que...\") "
        "and the competency id; anything else useful as opcional. "
        "Questions are open, one idea each, under 160 characters. Never "
        "make a slot or a question about age, sex, family, pregnancy, "
        "health or disability, religion, politics, union membership, "
        "sexual orientation, nationality or origin, even if the offer "
        "mentions them. No other field, no explanation."
    )


def _planner_user(request: ReasonRequest) -> str:
    competencies = "\n".join(
        f"- {c.get('id')}: {c.get('name')}"
        for c in _payload_list(request, "competencies")
        if isinstance(c, dict)
    )
    hint = _payload_text(request, "role_type", 20)
    offer = _payload_text(request, "offer", OFFER_CHARS)
    return (
        f"Competencies (id: name):\n{competencies or '(none)'}\n\n"
        f"Interview-method chunks:\n{chunks_text(request) or '(none)'}"
        f"\n\nTime budget: {_payload_text(request, 'minutes', 8) or '?'}"
        " minutes"
        + (f"\nKind of job (given): {hint}" if hint else "")
        + f"\n\nJob offer:\n<<<\n{offer}\n>>>\n\nJSON:"
    )


def _structured_parse(reply: str, request: ReasonRequest) -> Parsed:
    return Parsed(data=json_object(reply))


INTERVIEW_PLANNER_V1 = Template(
    id="interview.planner@1", role="structured",
    system=_planner_system, user=_planner_user, parse=_structured_parse,
    max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS,
)


def _slots_system(request: ReasonRequest) -> str:
    return (
        "You track the coverage of a job interview. Given the plan's "
        "slots (id, topic, evidence wanted), the interviewer's last "
        "question and the candidate's answer, say which slots the "
        "answer gives evidence for: complete when it gives a concrete "
        "fact or a real first-person example with what they did (and the "
        "result, for a competency), partial when it is on topic but "
        "vague, generic, hypothetical or misses that. Do not judge how "
        "good the answer is. Reply with one JSON object only: "
        '{"slots": [{"id": "slot id", "status": "complete|partial", '
        '"quote": "the candidate\'s own words, at most 20"}]} with at '
        'most three slots, or {"slots": []} when it covers none.'
    )


def _slots_user(request: ReasonRequest) -> str:
    rows = []
    for slot in _payload_list(request, "slots")[:16]:
        if isinstance(slot, dict):
            rows.append(
                f"[{str(slot.get('id'))[:12]}] {str(slot.get('label'))[:120]}"
                f" — evidence: {str(slot.get('evidence') or '-')[:200]}"
            )
    return (
        "Slots:\n" + ("\n".join(rows) or "(none)")
        + f"\nAsked about: {_payload_text(request, 'target', 12) or '-'}"
        + f"\n\nInterviewer: {_payload_text(request, 'question', 600)}"
        + f"\nCandidate: {_payload_text(request, 'answer', 2000)}"
        + "\n\nJSON:"
    )


INTERVIEW_SLOTS_V1 = Template(
    id="interview.slots@1", role="structured",
    system=_slots_system, user=_slots_user, parse=_structured_parse,
    max_output_tokens=SLOTS_MAX_OUTPUT_TOKENS,
)

REPORT_MAX_OUTPUT_TOKENS = 2400
"""Cap of ``interview.report@1``: a score and a note per slot, a summary."""


def _report_system(request: ReasonRequest) -> str:
    return (
        "You help a recruiter read a job interview held by an AI "
        "interviewer. For each slot of the plan, rate the candidate's "
        "evidence from 0 to 100 RELATIVE TO WHAT THIS OFFER REQUIRES: 100 "
        "means the answers show the level the job asks for (a junior "
        "job asks for junior level, not an expert), 50 partly, 0 no "
        "evidence or they said they do not know it. Base it only on the "
        "candidate's quotes; set above to true when the evidence clearly "
        "exceeds what the offer requires; confidence, from 0 to 1, says "
        "how much evidence there is. Also list skills the candidate "
        "showed that the offer does not ask for but are useful for this "
        "job, each with a short quote, and write a neutral summary for "
        "the recruiter (at most 600 characters) in "
        f"{language_name(request.language)}. Describe evidence, never "
        "personality, emotions, health or protected characteristics. "
        'Reply with one JSON object only: {"slots": [{"id": "slot id", '
        '"score": 0, "above": false, "confidence": 0.5, "notes": "at '
        'most 200 characters"}], "other_skills": [{"label": "...", '
        '"quote": "..."}], "summary": "..."}.'
    )


def _report_user(request: ReasonRequest) -> str:
    rows = []
    for slot in _payload_list(request, "slots")[:16]:
        if not isinstance(slot, dict):
            continue
        said = slot.get("quotes")
        quotes = " | ".join(
            str(q)[:300] for q in (said if isinstance(said, list) else [])[:4]
        )
        rows.append(
            f"[{str(slot.get('id'))[:12]}] {str(slot.get('label'))[:120]}"
            f" (level {str(slot.get('level'))[:10]}, status "
            f"{str(slot.get('status'))[:10]}) - evidence wanted: "
            f"{str(slot.get('evidence') or '-')[:200]}\n  quotes: "
            f"{quotes or '(none)'}"
        )
    return (
        f"Job: {_payload_text(request, 'title', 120) or '?'}\n"
        f"Job summary: {_payload_text(request, 'summary', 400) or '?'}\n\n"
        "Slots:\n" + ("\n".join(rows) or "(none)")
        + "\n\nCandidate's other words:\n"
        + (_payload_text(request, 'other', 3000) or '(none)')
        + "\n\nJSON:"
    )


INTERVIEW_REPORT_V1 = Template(
    id="interview.report@1", role="structured",
    system=_report_system, user=_report_user, parse=_structured_parse,
    max_output_tokens=REPORT_MAX_OUTPUT_TOKENS,
)


_REGISTRY: Dict[str, Template] = {
    t.id: t for t in (
        CONVERSATIONAL_V1, SUGGESTER_INTERVIEW_V1, SUGGESTER_SUPPORT_V1,
        SUGGESTER_INTERVIEW_V2, SUGGESTER_SUPPORT_V2, INTERVIEWER_V1,
        INTERVIEW_PLANNER_V1, INTERVIEW_SLOTS_V1, INTERVIEW_REPORT_V1,
    )
}


def template_role(template_id: str) -> Role:
    """Return the role of the template ``template_id``.

    Raises :class:`~src.domain.errors.InvalidInputError` when it does
    not exist.
    """
    template = _REGISTRY.get(template_id)
    if template is None:
        raise InvalidInputError(f"Unknown template {template_id!r}.")
    return template.role


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
