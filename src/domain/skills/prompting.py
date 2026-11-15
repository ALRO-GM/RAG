"""Prompt and reply format of the competency judge."""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from src.domain.skills.grounding import Judgement, ProposedEvidence
from src.domain.skills.rubric import Competency, Scale
from src.domain.skills.transcript import Transcript, Turn

_INTEGER = re.compile(r"-?[0-9]+")

LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}

SYSTEM_PROMPT = (
    "You are a trained assessor of behavioural competencies. You rate "
    "one competency of one participant from the transcript excerpts "
    "of a group work session, following the observe-record-classify-"
    "evaluate method.\n"
    "Rules:\n"
    "1. Use ONLY what the participant said in the numbered turns. "
    "Other speakers' turns are context.\n"
    "2. Evidence is a short verbatim quote of the participant, copied "
    "word for word from one turn, classified under exactly one "
    "indicator id of the rubric.\n"
    "3. Most turns are NOT evidence. Cite a turn only when it clearly "
    "shows the behaviour its indicator describes; citing few turns, or "
    "none, is a normal result.\n"
    "4. Judge the behaviour, not the person: ignore accent, grammar, "
    "fluency, vocabulary, gender, age, role or how much they spoke.\n"
    "5. Do not infer intentions, personality or what happened outside "
    "the excerpts.\n"
    "6. Rate with the anchors only if at least {min_incidents} "
    "different turns give evidence; otherwise the rating is null. "
    "Absence of evidence is not evidence of a low level.\n"
    "7. Reply with one JSON object and nothing else:\n"
    '{{"evidence": [{{"turn": <turn number>, "quote": "<verbatim '
    'words>", "indicator": "<indicator id>"}}], "rating": <integer '
    '{minimum}-{maximum} or null>, "rationale": "<one or two '
    'sentences in {language}>"}}'
)


PromptStyle = Literal["standard", "guided", "precise"]
"""How the judge is asked: ``standard`` (Claude), or one of the two
variants for small local models measured in #104: ``guided`` (walk the
turns, more quotes) and ``precise`` (walk the turns, name what each
quote does before filing it)."""

MAX_QUOTES = 6
"""Most evidence items the guided prompt asks for in one reply."""

GUIDED_SYSTEM_PROMPT = (
    "You are a trained assessor of behavioural competencies. You read "
    "the turns of one participant of a group work session and find "
    "the ones that show the indicators of one competency.\n"
    "How to work:\n"
    "1. The participant's turns are numbered, like [12]. Lines marked "
    "(context) are other speakers; they only help to understand.\n"
    "2. Go through the numbered turns one by one. For each, ask: does "
    "this turn show the behaviour of one of the indicators? If it "
    "does, add one evidence item with the turn number, the indicator "
    "id and a quote. Leave out turns that show none.\n"
    "3. The quote is the part of the turn that shows the behaviour: "
    "one sentence or clause, usually 5 to 25 words, copied character "
    "by character from that turn, in the language of the transcript. "
    "Do not translate, summarise, correct or join words from "
    "different places.\n"
    "4. A turn may show two indicators: cite each. Give at most "
    "{max_quotes} items, the clearest first.\n"
    "5. Judge the behaviour, not the person: ignore accent, grammar, "
    "fluency, vocabulary, gender, age, role or how much they spoke. "
    "Do not infer intentions, personality or what happened outside "
    "the excerpts.\n"
    "6. Rate with the anchors only if at least {min_incidents} "
    "different turns give evidence; otherwise the rating is null. "
    "Absence of evidence is not evidence of a low level.\n"
    "7. Reply with one JSON object and nothing else, like this "
    "example (an invented meeting and an invented competency whose "
    "indicators are X.P1 (+) asks the others for their view before "
    "deciding, X.P2 (+) proposes a concrete next step):\n"
    "{example_turns}\n"
    "{example_reply}"
)

PRECISE_SYSTEM_PROMPT = (
    "You are a trained assessor of behavioural competencies. You read "
    "the turns of one participant of a group work session and find "
    "the ones that show the indicators of one competency.\n"
    "How to work:\n"
    "1. The participant's turns are numbered, like [12]. Lines marked "
    "(context) are other speakers: never quote them or cite them as "
    "the participant's; they only help to understand.\n"
    "2. Go through the numbered turns one by one. For each, say to "
    "yourself what the participant's words do (ask, propose, inform, "
    "agree, summarise, offer, apologise...). Cite the turn only if "
    "that is exactly what one indicator describes. A turn that is on "
    "the topic of the competency but does something else is not "
    "evidence. Most turns fit no indicator; few items, or none, is a "
    "normal result.\n"
    "3. Each evidence item has, in this order, the turn number, "
    "\"why\" (a few words saying what the words do, in the terms of "
    "the indicator), the quote and the indicator id.\n"
    "4. The quote is the part of the turn that does it: one sentence "
    "or clause, usually 5 to 25 words, copied character by character "
    "from that turn, in the language of the transcript. Do not "
    "translate, summarise, correct or join words from different "
    "places. Give at most {max_quotes} items.\n"
    "5. Judge the behaviour, not the person: ignore accent, grammar, "
    "fluency, vocabulary, gender, age, role or how much they spoke. "
    "Do not infer intentions, personality or what happened outside "
    "the excerpts.\n"
    "6. Rate with the anchors only if at least {min_incidents} "
    "different turns give evidence; otherwise the rating is null. "
    "Absence of evidence is not evidence of a low level.\n"
    "7. Reply with one JSON object and nothing else, like this "
    "example (an invented meeting and an invented competency whose "
    "indicators are X.P1 (+) asks the others for their view before "
    "deciding, X.P2 (+) proposes a concrete next step; turn 8 does "
    "neither, so it is left out):\n"
    "{example_turns}\n"
    "{example_reply}"
)

_WHY = (
    "asks the others' view before deciding",
    "proposes a concrete next step with a date",
)

_EXAMPLES = {
    "es": (
        "[3] Marta: Antes de cerrarlo, ¿qué opináis vosotros del "
        "horario nuevo? Yo no lo veo claro.\n"
        "(context) Luis: A mí los lunes me viene mal.\n"
        "[5] Marta: Vale. Entonces lo probamos dos semanas con los "
        "martes y el viernes que viene lo revisamos.\n"
        "[8] Marta: Hoy hace mucho calor aquí dentro.",
        "¿qué opináis vosotros del horario nuevo?",
        "lo probamos dos semanas con los martes y el viernes que viene "
        "lo revisamos",
    ),
    "en": (
        "[3] Marta: Before we settle it, what do you all think of the "
        "new timetable? I am not sure about it.\n"
        "(context) Luis: Mondays are bad for me.\n"
        "[5] Marta: Fine. Then we try Tuesdays for two weeks and review "
        "it next Friday.\n"
        "[8] Marta: It is very hot in here today.",
        "what do you all think of the new timetable?",
        "we try Tuesdays for two weeks and review it next Friday",
    ),
}


def _guided_example(
    scale: Scale, min_incidents: int, language: str, why: bool = False,
) -> Tuple[str, str]:
    """Return the example turns and reply of the guided prompt.

    The example is in the transcript's language (English when there is
    none for it): a small model copies the quoting it is shown. Its
    rating follows rule 6 with the session's ``min_incidents``;
    ``why`` adds the reason of each item (the ``precise`` style).
    """
    turns, first, second = _EXAMPLES.get(language, _EXAMPLES["en"])
    rating: Optional[int] = None
    if min_incidents <= 2:
        rating = (scale.minimum + scale.maximum) // 2
    items: List[Dict[str, Any]] = [
        {"turn": 3, "quote": first, "indicator": "X.P1"},
        {"turn": 5, "quote": second, "indicator": "X.P2"},
    ]
    if why:
        items = [
            {"turn": item["turn"], "why": reason, "quote": item["quote"],
             "indicator": item["indicator"]}
            for item, reason in zip(items, _WHY)
        ]
    reply = json.dumps({
        "evidence": items,
        "rating": rating,
        "rationale": "<one or two sentences>",
    }, ensure_ascii=False)
    return turns, reply


def system_prompt(
    scale: Scale,
    min_incidents: int,
    language: str,
    style: PromptStyle = "standard",
) -> str:
    """Return the system prompt for a scale and an output language.

    ``guided`` and ``precise`` are the variants for small local models
    (#104): both walk the model through the turns one by one, cap the
    quotes at :data:`MAX_QUOTES` and show one example reply; ``precise``
    also asks what each quote does before it is filed under an
    indicator (a ``why`` field, ignored by the parser). The reply
    format is otherwise the same, so :func:`parse_judgement` reads all.
    """
    if style in ("guided", "precise"):
        turns, reply = _guided_example(
            scale, min_incidents, language, why=style == "precise"
        )
        template = (
            PRECISE_SYSTEM_PROMPT if style == "precise"
            else GUIDED_SYSTEM_PROMPT
        )
        rules = template.format(
            max_quotes=MAX_QUOTES,
            min_incidents=min_incidents,
            example_turns=turns,
            example_reply=reply,
        )
        return (
            f"{rules}\n"
            f"Write the rationale in "
            f"{LANGUAGE_NAMES.get(language, language)}; the rating is "
            f"an integer {scale.minimum}-{scale.maximum} or null."
        )
    return SYSTEM_PROMPT.format(
        min_incidents=min_incidents,
        minimum=scale.minimum,
        maximum=scale.maximum,
        language=LANGUAGE_NAMES.get(language, language),
    )


def format_rubric(competency: Competency, scale: Scale) -> str:
    """Render a competency as the rubric section of the prompt."""
    lines = [
        f"Competency: {competency.name} ({competency.id})",
        f"Definition: {competency.definition}",
        "Indicators:",
    ]
    for indicator in competency.indicators:
        sign = "+" if indicator.polarity == "positive" else "-"
        lines.append(f"  {indicator.id} ({sign}) {indicator.text}")
    lines.append("Rating anchors:")
    for point in scale.points():
        anchor = competency.anchors.get(point) or scale.labels.get(point)
        if anchor:
            lines.append(f"  {point}: {anchor}")
    if competency.limits:
        lines.append(f"Do not infer: {competency.limits}")
    return "\n".join(lines)


def format_turns(
    transcript: Transcript, candidates: Sequence[Turn], participant: str
) -> str:
    """Render the candidate turns, each after the turn it replies to."""
    shown: set[int] = set()
    lines: List[str] = []
    for turn in sorted(candidates, key=lambda t: t.index):
        before = turn.index - 1
        if before >= 0 and before not in shown:
            context = transcript.turns[before]
            if context.speaker != participant:
                lines.append(
                    f"(context) {context.speaker}: {context.text}"
                )
        if turn.index not in shown:
            lines.append(f"[{turn.index}] {turn.speaker}: {turn.text}")
            shown.add(turn.index)
    return "\n".join(lines)


def user_prompt(
    transcript: Transcript,
    candidates: Sequence[Turn],
    participant: str,
    competency: Competency,
    scale: Scale,
) -> str:
    """Return the user turn: rubric, excerpts and the task."""
    return (
        f"{format_rubric(competency, scale)}\n\n"
        f"Participant to assess: {participant}\n\n"
        f"Transcript excerpts:\n"
        f"{format_turns(transcript, candidates, participant)}\n\n"
        "JSON:"
    )


def _as_int(value: Any) -> Optional[int]:
    """Coerce a JSON value to an integer when it is one."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str) and _INTEGER.fullmatch(value.strip()):
        return int(value.strip())
    return None


def parse_judgement(reply: str) -> Judgement:
    """Parse the judge's reply, tolerating text around the JSON object.

    A reply that is not a JSON object yields an empty judgement with
    ``parse_error`` set, which the caller turns into "insufficient
    evidence" rather than an invented rating.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start or reply.lstrip().startswith("["):
        return Judgement(parse_error=True)
    try:
        data = json.loads(reply[start:end + 1])
    except ValueError:
        return Judgement(parse_error=True)
    if not isinstance(data, dict):
        return Judgement(parse_error=True)
    items = data.get("evidence")
    evidence: List[ProposedEvidence] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        evidence.append(ProposedEvidence(
            turn=_as_int(item.get("turn")),
            quote=str(item.get("quote") or ""),
            indicator=str(item.get("indicator") or ""),
        ))
    return Judgement(
        evidence=evidence,
        rating=_as_int(data.get("rating")),
        rationale=" ".join(str(data.get("rationale") or "").split()),
    )
