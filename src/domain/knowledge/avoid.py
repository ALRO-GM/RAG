"""Questions better avoided in an interview: the hint the live page shows.

When an interviewer asks about a protected characteristic (age, family
plans, health, religion, origin...), the live page labels that line
«Mejor evitar» and shows why and a lawful way to ask about what the job
really needs (LV-17, #232). The categories, patterns and texts come from
the knowledge file of KN-4 (``knowledge/interview/forbidden-questions.json``,
read by :mod:`src.domain.knowledge.interview_rules`); this module is only
the shape of one hint, in one language. A hint is about the question,
never a verdict and never about the candidate. Plain data: no I/O.
"""

from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

AVOID_LANGUAGES = ("es", "en")
"""Languages the hints are written in."""
MAX_QUESTION = 2000
"""Longest sentence checked (an interviewer's final is far shorter)."""


class AvoidHint(BaseModel):
    """One category a question touches, in one language.

    ``category`` is the rules file's id (``age``, ``health``...);
    ``title`` names it, ``why`` says in one or two sentences why it is
    better not asked and ``alternative`` how to ask about the job
    instead (or that there is no question to ask).
    """

    model_config = ConfigDict(frozen=True)

    category: str = Field(min_length=1)
    title: str = Field(min_length=1)
    why: str = Field(min_length=1)
    alternative: str = Field(min_length=1)


def language_of(language: str) -> str:
    """Return ``language`` when hints exist in it, else Spanish."""
    return language if language in AVOID_LANGUAGES else "es"


def in_language(texts: Mapping[str, str], language: str) -> str:
    """Pick ``language`` from a ``{"es", "en"}`` text, else the Spanish."""
    return texts.get(language) or texts.get("es") or ""
