"""Interview rules of the knowledge folder (KN-4, #234).

Two plain-data files describe what an interviewer, human or AI, must not
ask under Spanish and EU law and which few questions every interview
asks once: ``knowledge/interview/forbidden-questions.json``
(``interview.forbidden.v1``) and ``knowledge/interview/core-questions.json``
(``interview.core.v1``). This module is their schema and the one rule to
read them: a hint, never a verdict, and **not legal advice**. No I/O:
callers pass the parsed JSON.

Patterns are Python regular expressions applied to the question after
:func:`fold` (lower case, accents removed, typographic apostrophes
turned into ``'``), per language (``es``, ``en``).
"""

import re
import unicodedata
from typing import Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

LANGUAGES = ("es", "en")
MAX_CORE_QUESTIONS = 6
"""Core questions are kept few on purpose (the interview is not saturated)."""


class Text(BaseModel):
    """A text in both languages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    es: str = Field(min_length=1)
    en: str = Field(min_length=1)


class Patterns(BaseModel):
    """Regular expressions per language, matched on the folded question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    es: List[str] = Field(min_length=1)
    en: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _compile(self) -> "Patterns":
        """Refuse a pattern that is not a valid regular expression."""
        for pattern in (*self.es, *self.en):
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"bad pattern {pattern!r}: {error}") from error
        return self


class Examples(BaseModel):
    """Example questions per language (each one must match its patterns)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    es: List[str] = Field(min_length=1)
    en: List[str] = Field(min_length=1)


class Basis(BaseModel):
    """One legal basis: a source, an article, its URL and the exact words.

    ``verified`` is true only when the article's wording was read in the
    official text; otherwise the item is *unverified* and ``note`` says
    what was not checked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    quote: Optional[str] = None
    verified: bool
    note: Optional[str] = None

    @model_validator(mode="after")
    def _evidence(self) -> "Basis":
        """An unverified basis must say what was not checked."""
        if not self.verified and not self.note:
            raise ValueError(f"unverified basis {self.ref!r} needs a note")
        return self


class Exemption(BaseModel):
    """A legitimate exception (work permit, minimum age, essential duty)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    es: str = Field(min_length=1)
    en: str = Field(min_length=1)
    basis: List[Basis]


class Source(BaseModel):
    """A source of the research log: where and how it was read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    id: Optional[str]
    url: str = Field(pattern=r"^https://")
    kind: str
    text_read: bool
    how_read: str


class Category(BaseModel):
    """One category of forbidden or to-avoid questions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    category: str
    title: Text
    patterns: Patterns
    examples: Examples
    why: Text
    legal_basis: List[Basis] = Field(min_length=1)
    lawful_alternative: Text
    exceptions: List[Exemption]
    verification: str = Field(pattern=r"^(verified|partly_verified)$")
    status_note: Optional[Text] = None

    def matches(self, folded: str) -> bool:
        """Tell whether an already folded question matches a pattern."""
        return any(
            re.search(p, folded)
            for lang in LANGUAGES for p in getattr(self.patterns, lang)
        )


class AiLimit(BaseModel):
    """A limit the EU AI Act, the GDPR or Law 15/2022 puts on the product."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    limit: Text
    status: str
    why: Text
    basis: List[Basis] = Field(min_length=1)
    product_rule: Text


class ForbiddenRules(BaseModel):
    """The parsed ``interview.forbidden.v1`` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: str = Field(alias="schema", pattern=r"^interview\.forbidden\.v1$")
    version: int = Field(ge=1)
    language_of_patterns: str
    disclaimer: Text
    checked_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    pattern_rules: str
    sources: Dict[str, Source]
    categories: List[Category] = Field(min_length=1)
    ai_limits: List[AiLimit]

    @model_validator(mode="after")
    def _consistent(self) -> "ForbiddenRules":
        """Unique ids and every basis names a listed source."""
        ids = [c.id for c in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError("category ids must be unique")
        bases = [b for c in self.categories for b in c.legal_basis]
        bases += [b for c in self.categories for e in c.exceptions
                  for b in e.basis]
        bases += [b for a in self.ai_limits for b in a.basis]
        missing = {b.source for b in bases} - set(self.sources)
        if missing:
            raise ValueError(f"sources not listed: {sorted(missing)}")
        return self

    def hits(self, question: str) -> List[str]:
        """Return the ids of the categories a question matches (a hint)."""
        folded = fold(question)
        return [c.id for c in self.categories if c.matches(folded)]


class ProfileField(BaseModel):
    """Where an answer goes in the candidate profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(pattern=r"^[a-z_]+(\.[a-z_]+)+$")
    type: str
    unit: Optional[str] = None
    optional: bool = False
    max_chars: Optional[int] = None


class CoreQuestion(BaseModel):
    """A question asked once per candidate whatever the role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    order: int = Field(ge=1)
    phase: str
    question: Text
    profile_fields: List[ProfileField] = Field(min_length=1)
    why: Text
    follow_up: Text
    legal_note: Text
    related_forbidden: List[str] = Field(default_factory=list)
    legal_basis: List[str] = Field(default_factory=list)


class CoreQuestions(BaseModel):
    """The parsed ``interview.core.v1`` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: str = Field(alias="schema", pattern=r"^interview\.core\.v1$")
    version: int = Field(ge=1)
    max_questions: int = Field(ge=1, le=MAX_CORE_QUESTIONS)
    disclaimer: Text
    rules: Text
    questions: List[CoreQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _few_and_unique(self) -> "CoreQuestions":
        """At most ``max_questions``; unique ids; orders 1..n in a row."""
        if len(self.questions) > self.max_questions:
            raise ValueError("too many core questions")
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("core question ids must be unique")
        if [q.order for q in self.questions] != list(
                range(1, len(self.questions) + 1)):
            raise ValueError("orders must be 1..n in file order")
        return self

    def profile_paths(self) -> List[str]:
        """Return every profile path the core questions fill."""
        return [f.path for q in self.questions for f in q.profile_fields]


def fold(text: str) -> str:
    """Lower-case ``text``, drop accents and straighten apostrophes."""
    straight = text.replace("’", "'").replace("‘", "'")
    parts = unicodedata.normalize("NFKD", straight.lower())
    return "".join(ch for ch in parts if not unicodedata.combining(ch))


def parse_forbidden(data: Mapping[str, object]) -> ForbiddenRules:
    """Validate the parsed forbidden-questions JSON."""
    return ForbiddenRules.model_validate(data)


def parse_core(data: Mapping[str, object]) -> CoreQuestions:
    """Validate the parsed core-questions JSON."""
    return CoreQuestions.model_validate(data)


def core_hits(
    rules: ForbiddenRules, core: CoreQuestions
) -> Dict[str, List[str]]:
    """Return, per core question id, the forbidden categories it matches.

    Both languages of the question are checked; an empty result for every
    id is the invariant the tests keep.
    """
    found: Dict[str, List[str]] = {}
    for q in core.questions:
        texts: Sequence[str] = (q.question.es, q.question.en)
        hits = sorted({h for t in texts for h in rules.hits(t)})
        found[q.id] = hits
    return found
