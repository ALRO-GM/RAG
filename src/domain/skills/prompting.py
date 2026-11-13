"""Prompt and reply format of the competency judge."""

import json
import re
from typing import Any, List, Optional, Sequence

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


def system_prompt(scale: Scale, min_incidents: int, language: str) -> str:
    """Return the system prompt for a scale and an output language."""
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
