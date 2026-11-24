"""Interview questions better avoided, and whether they were rephrased.

LV-17 (#232): when the interviewer asks about a protected
characteristic (age, family plans, health, religion, origin...), the
RAG's check of KN-4's rules returns the categories the sentence touches
with why and a lawful alternative. The live page labels that line
«Mejor evitar» and shows the hint; it is a hint about the question,
never a verdict and never anything about the candidate.

A flag reads ``rephrased`` («Reformulado») when the interviewer asks
another question (a sentence with ``?`` or ``¿``) without that category
within :data:`REPHRASE_WINDOW_S` seconds after it. Checks may answer out
of order, so :class:`AvoidLedger` decides on stream times, not on the
order results arrive. Plain data and rules: no I/O.
"""

from typing import (
    Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple,
)

from pydantic import BaseModel, Field, ValidationError

REPHRASE_WINDOW_S = 30.0
"""Seconds after a flagged line in which a clean question counts as its
rephrasing (a display choice, not a measured value)."""

FlagState = Literal["flagged", "rephrased"]


class AvoidFlag(BaseModel):
    """One category a line of the interviewer touched."""

    utterance: str
    at: float
    category: str = Field(min_length=1)
    title: str = Field(min_length=1)
    why: str = Field(min_length=1)
    alternative: str = Field(min_length=1)
    state: FlagState = "flagged"
    rephrased_by: Optional[str] = None
    en: Optional[Dict[str, str]] = None
    """English texts of a prepared (demo) hint, if any."""


def is_question(text: str) -> bool:
    """Tell whether a sentence reads as a question (``?`` or ``¿``)."""
    return "?" in text or "¿" in text


def parse_hints(answer: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return the well-formed hints of a check's answer (others dropped)."""
    found: List[Dict[str, Any]] = []
    for hint in answer.get("hints") or []:
        if not isinstance(hint, Mapping):
            continue
        fields = {k: hint.get(k) for k in
                  ("category", "title", "why", "alternative")}
        if all(isinstance(v, str) and v for v in fields.values()):
            found.append({k: str(v) for k, v in fields.items()})
    return found


class AvoidLedger:
    """The flags of one session and the clean questions that may rephrase them.

    Args:
        window: Seconds a clean question may come after a flag.
    """

    def __init__(self, window: float = REPHRASE_WINDOW_S) -> None:
        """Start with no flags."""
        self.window = window
        self.flags: List[AvoidFlag] = []
        self._clean: List[Tuple[str, float]] = []

    def record(
        self, utterance: str, at: float, text: str,
        hints: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Take the check of one line; return whether a flag changed."""
        changed = False
        categories = set()
        for hint in hints:
            try:
                flag = AvoidFlag.model_validate({
                    **hint, "utterance": utterance, "at": at,
                    "state": "flagged", "rephrased_by": None,
                })
            except ValidationError:
                continue
            if any(f.utterance == utterance and f.category == flag.category
                   for f in self.flags):
                continue
            categories.add(flag.category)
            self.flags.append(flag)
            changed = True
            for later, when in self._clean:
                changed |= self._rephrase(flag, later, when)
        if is_question(text):
            self._clean.append((utterance, at))
            for flag in self.flags:
                if flag.category not in categories:
                    changed |= self._rephrase(flag, utterance, at)
        self.flags.sort(key=lambda f: (f.at, f.utterance, f.category))
        return changed

    def _rephrase(self, flag: AvoidFlag, utterance: str, at: float) -> bool:
        """Mark ``flag`` rephrased by a clean question, if it fits."""
        if flag.state == "rephrased" or utterance == flag.utterance:
            return False
        if not 0 < at - flag.at <= self.window:
            return False
        if any(f.utterance == utterance and f.category == flag.category
               for f in self.flags):
            return False
        flag.state = "rephrased"
        flag.rephrased_by = utterance
        return True

    def drop(self, utterance: str) -> bool:
        """Forget a line's flags (it was not the interviewer's after all)."""
        before = len(self.flags)
        self.flags = [f for f in self.flags if f.utterance != utterance]
        self._clean = [c for c in self._clean if c[0] != utterance]
        for flag in self.flags:
            if flag.rephrased_by == utterance:
                flag.state, flag.rephrased_by = "flagged", None
        return len(self.flags) != before

    def view(self) -> List[Dict[str, Any]]:
        """Return every flag as JSON, oldest first."""
        return [f.model_dump(mode="json") for f in self.flags]
