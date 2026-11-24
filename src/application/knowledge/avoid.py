"""Use case: tell whether an interviewer's sentence is better avoided.

The orchestrator sends each final sentence of the interviewer
(``POST /knowledge/interview/avoid``); the answer lists the categories
of protected questions it touches, with why and a lawful alternative
(:class:`~src.domain.knowledge.avoid.AvoidHint`). The rules live behind
:class:`ForbiddenRules`: the adapter reads the KN-4 knowledge file
through its own module, so this use case never sees patterns. A match
is a hint for the interviewer, never a verdict, and says nothing about
the candidate.
"""

from typing import List, Mapping, Protocol

from src.domain.errors import InvalidInputError
from src.domain.knowledge.avoid import MAX_QUESTION, AvoidHint, language_of


class ForbiddenRules(Protocol):
    """The questions better avoided in an interview (KN-4's file)."""

    def hints(self, question: str, language: str) -> List[AvoidHint]:
        """Return the categories ``question`` touches, in ``language``.

        Raises:
            NotFoundError: There are no rules to check with.
        """
        ...


class AvoidQuestions:
    """Check one sentence against the rules.

    Args:
        rules: The rules of questions better avoided.
    """

    def __init__(self, rules: ForbiddenRules) -> None:
        """Keep the rules (read by the adapter when first needed)."""
        self.rules = rules

    def check(self, text: str, language: str = "es") -> Mapping[str, object]:
        """Return ``{"language", "hints": [...]}`` for one sentence.

        ``InvalidInputError`` for an empty or too long text;
        ``NotFoundError`` (from the rules) when there are none.
        """
        question = text.strip()
        if not question:
            raise InvalidInputError("text is empty")
        if len(question) > MAX_QUESTION:
            raise InvalidInputError(
                f"text is longer than {MAX_QUESTION} characters"
            )
        chosen = language_of(language)
        hints = self.rules.hints(question, chosen)
        return {
            "language": chosen,
            "hints": [hint.model_dump() for hint in hints],
        }
