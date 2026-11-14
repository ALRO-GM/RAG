"""Prompt and reply format of the sentence classifier (call 1 of 2).

``THEORY.md`` asks for classification and evaluation in separate model
calls, so that an overall impression of a person cannot leak into what
counts as evidence. This is the first call: one sentence, its context,
the indicators of the competencies it may bear on; the answer is one
indicator or none. No rating, and nothing about the person.
"""

import json
from typing import List, Optional, Sequence, Tuple

from src.domain.skills.prompting import LANGUAGE_NAMES
from src.domain.skills.rubric import Competency
from src.domain.skills.speech import Sentence

CONFIRM_SYSTEM_PROMPT = (
    "You code observable behaviour in the transcript of a group work "
    "session. You are given ONE target sentence, the sentences said "
    "just before it, and a list of behavioural indicators.\n"
    "Rules:\n"
    "1. Decide whether the TARGET sentence, read in its context, "
    "clearly shows exactly one of the listed indicators.\n"
    "2. Most sentences show none of them: answer null unless the "
    "behaviour is clearly there. Sharing a topic or a word with an "
    "indicator is not enough.\n"
    "3. Code what was said and done in the sentence, not the person: "
    "ignore accent, grammar, fluency, gender, age or role, and do not "
    "infer intentions.\n"
    "4. The context sentences are never coded; they only tell you what "
    "the target replies to.\n"
    "5. Reply with one JSON object and nothing else:\n"
    '{{"indicator": "<one indicator id from the list>" or null, '
    '"reason": "<one short sentence in {language}>"}}'
)


def confirm_system_prompt(language: str) -> str:
    """Return the classifier's system prompt for an output language."""
    return CONFIRM_SYSTEM_PROMPT.format(
        language=LANGUAGE_NAMES.get(language, language)
    )


def format_indicators(competencies: Sequence[Competency]) -> str:
    """Render the indicators of ``competencies``, grouped and signed."""
    lines: List[str] = []
    for competency in competencies:
        lines.append(f"{competency.name} ({competency.id}):")
        for indicator in competency.indicators:
            sign = "+" if indicator.polarity == "positive" else "-"
            lines.append(f"  {indicator.id} ({sign}) {indicator.text}")
    return "\n".join(lines)


def confirm_user_prompt(
    context: Sequence[Sentence],
    target: Sentence,
    competencies: Sequence[Competency],
) -> str:
    """Return the user turn: indicators, context, target, task.

    When the transcript says who spoke in the turn before the target
    (``previous_speaker``), the target line says so too; otherwise the
    prompt is unchanged.
    """
    before = "\n".join(f"{s.speaker}: {s.text}" for s in context)
    said = f"said by {target.speaker}"
    if target.previous_speaker:
        said += f", after {target.previous_speaker} spoke"
    return (
        f"Indicators:\n{format_indicators(competencies)}\n\n"
        f"Context (not coded):\n{before or '(start of the session)'}\n\n"
        f"TARGET sentence, {said}:\n{target.text}\n\n"
        "JSON:"
    )


def parse_confirmation(reply: str) -> Tuple[Optional[str], str, bool]:
    """Parse the classifier's reply: ``(indicator id, reason, ok)``.

    ``ok`` is False when the reply is not a JSON object with an
    ``indicator`` key; the caller then records the sentence as
    unparsed instead of guessing. ``"none"``, ``"null"`` and an empty
    string mean no indicator, as ``null`` does.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None, "", False
    try:
        data = json.loads(reply[start:end + 1])
    except ValueError:
        return None, "", False
    if not isinstance(data, dict) or "indicator" not in data:
        return None, "", False
    value = data.get("indicator")
    reason = " ".join(str(data.get("reason") or "").split())
    if value is None or isinstance(value, bool):
        return None, reason, True
    text = str(value).strip()
    if text.lower() in ("", "none", "null"):
        return None, reason, True
    return text, reason, True
