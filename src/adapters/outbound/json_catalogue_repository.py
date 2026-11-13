"""Catalogue repository adapter: rubrics as ``<name>.<language>.json``."""

from pathlib import Path
from typing import List, Tuple

from pydantic import ValidationError

from src.domain.errors import (
    CorruptedDataError,
    NotFoundError,
    RagError,
    short_reason,
)
from src.domain.skills.rubric import RubricCatalogue


class JsonCatalogueRepository:
    """Read rubric catalogues from a directory of JSON files."""

    def __init__(self, directory: str = "knowledge/rubrics"):
        """Read catalogues under ``directory``."""
        self.directory = Path(directory)

    def available(self) -> List[Tuple[str, str]]:
        """Return the (name, language) pairs found in the directory."""
        found: List[Tuple[str, str]] = []
        for file in sorted(self.directory.glob("*.*.json")):
            name, language = file.name[:-5].rsplit(".", 1)
            found.append((name, language))
        return found

    def load(self, name: str, language: str) -> RubricCatalogue:
        """Return the catalogue ``name`` in ``language``.

        ``name`` may also be the path of a catalogue file.
        """
        file = Path(str(name))
        if not file.is_file():
            file = self.directory / f"{name}.{language[:2].lower()}.json"
        if not file.is_file():
            known = ", ".join(f"{n} ({lang})" for n, lang in self.available())
            raise NotFoundError(
                f"Catalogue {name!r} is not available in {language!r}. "
                f"Available: {known or 'none'}."
            )
        try:
            return RubricCatalogue.model_validate_json(
                file.read_text(encoding="utf-8")
            )
        except (ValidationError, ValueError) as exc:
            raise CorruptedDataError(
                f"Invalid catalogue {file}: {short_reason(exc)}"
            )
        except OSError as exc:
            raise RagError(f"Cannot read {file}: {exc}")
