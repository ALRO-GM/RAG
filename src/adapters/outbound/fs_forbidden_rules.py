"""The interview rules of KN-4 read from the knowledge folder.

``knowledge/interview/forbidden-questions.json`` (``interview.forbidden.v1``)
lists the categories of questions an interviewer should not ask, with
patterns, why and a lawful alternative. Its schema and matching rule are
the domain module :mod:`src.domain.knowledge.interview_rules` (#234);
this adapter only reads the file, hands the parsed JSON to that module
and turns its hits into :class:`~src.domain.knowledge.avoid.AvoidHint`.

The module is imported by name when the file is first read, so a
checkout without it (or without the file) answers ``NotFoundError``
instead of failing at start-up. The file is read again when it changes.
"""

import importlib
import json
import threading
from pathlib import Path
from types import ModuleType
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple,
)

from src.domain.errors import NotFoundError
from src.domain.knowledge.avoid import AvoidHint, in_language

RULES_MODULE = "src.domain.knowledge.interview_rules"
"""The module that parses and matches the rules file (KN-4, #234)."""


class _Text(Protocol):
    """A text in both languages, as the rules module gives it."""

    @property
    def es(self) -> str:
        """The Spanish text."""
        ...

    @property
    def en(self) -> str:
        """The English text."""
        ...


class _Category(Protocol):
    """One category of the parsed rules (only what is shown)."""

    @property
    def id(self) -> str:
        """The category id."""
        ...

    @property
    def title(self) -> _Text:
        """Its name."""
        ...

    @property
    def why(self) -> _Text:
        """Why it is better not asked."""
        ...

    @property
    def lawful_alternative(self) -> _Text:
        """How to ask about the job instead."""
        ...


class _Rules(Protocol):
    """The parsed rules file: its categories and the match."""

    @property
    def categories(self) -> Sequence[_Category]:
        """Every category, in file order."""
        ...

    def hits(self, question: str) -> List[str]:
        """Return the ids of the categories ``question`` matches."""
        ...


def _both(text: _Text) -> Dict[str, str]:
    """Return a two-language text as a mapping."""
    return {"es": text.es, "en": text.en}


class FileForbiddenRules:
    """The rules of questions better avoided, read from ``path``.

    Args:
        path: The ``interview.forbidden.v1`` JSON file.
        load_module: Imports the rules module (replaced in tests).
    """

    def __init__(
        self,
        path: Path,
        load_module: Callable[[str], ModuleType] = importlib.import_module,
    ) -> None:
        """Keep where the file is; nothing is read yet."""
        self.path = Path(path)
        self.load_module = load_module
        self._cached: Optional[Tuple[float, _Rules]] = None
        self._lock = threading.Lock()

    def _rules(self) -> _Rules:
        """Return the parsed rules, read again when the file changed."""
        try:
            stamp = self.path.stat().st_mtime
        except OSError as exc:
            raise NotFoundError(
                f"No interview rules file at {self.path}."
            ) from exc
        with self._lock:
            if self._cached is not None and self._cached[0] == stamp:
                return self._cached[1]
            try:
                module = self.load_module(RULES_MODULE)
            except ImportError as exc:
                raise NotFoundError(
                    "The interview rules module is not installed."
                ) from exc
            parse: Callable[[Any], _Rules] = getattr(
                module, "parse_forbidden"
            )
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                rules = parse(data)
            except (OSError, ValueError) as exc:
                raise NotFoundError(
                    f"Unreadable interview rules {self.path.name}: "
                    f"{type(exc).__name__}"
                ) from exc
            self._cached = (stamp, rules)
            return rules

    def hints(self, question: str, language: str) -> List[AvoidHint]:
        """Return the categories ``question`` touches, in ``language``."""
        rules = self._rules()
        found = set(rules.hits(question))
        return [
            AvoidHint(
                category=c.id,
                title=in_language(_both(c.title), language),
                why=in_language(_both(c.why), language),
                alternative=in_language(
                    _both(c.lawful_alternative), language
                ),
            )
            for c in rules.categories if c.id in found
        ]
