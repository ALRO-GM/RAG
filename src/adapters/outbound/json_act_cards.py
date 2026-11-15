"""Read the theory corpus of the dialogue acts from JSON files (#94).

``<directory>/categories.json`` holds ``{"cards": [...]}`` (one
:class:`~src.domain.skills.act_retrieval.CategoryCard` per category)
and ``<directory>/sequences.json`` holds ``{"sequences": [...]}``.
Other keys (``about``) are notes for the people who edit the cards.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.skills.act_retrieval import TheoryCorpus

DEFAULT_THEORY_DIR = "knowledge/dialogue_acts/theory"
"""Where the cards live in the repository."""


def _read(path: Path) -> Dict[str, Any]:
    """Return one JSON object file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"No theory file {path}")
    except (OSError, ValueError) as exc:
        raise InvalidInputError(f"Cannot read {path}: {exc}")
    if not isinstance(data, dict):
        raise InvalidInputError(f"{path} must hold a JSON object")
    return data


class JsonActCards:
    """Load and fingerprint the theory cards of a directory."""

    def __init__(self, directory: str = DEFAULT_THEORY_DIR):
        """Remember the directory; nothing is read yet."""
        self.directory = Path(directory)

    def load(self) -> TheoryCorpus:
        """Return the validated corpus."""
        cards = _read(self.directory / "categories.json")
        sequences = _read(self.directory / "sequences.json")
        try:
            return TheoryCorpus(cards=cards.get("cards", []),
                                sequences=sequences.get("sequences", []))
        except ValidationError as exc:
            raise InvalidInputError(
                f"Invalid theory cards in {self.directory}: {exc}")

    def version(self) -> str:
        """Return a short hash of both files (changes when a card does)."""
        digest = hashlib.sha256()
        for name in ("categories.json", "sequences.json"):
            try:
                digest.update((self.directory / name).read_bytes())
            except OSError as exc:
                raise NotFoundError(f"Cannot read {name}: {exc}")
        return f"theory-{digest.hexdigest()[:10]}"
